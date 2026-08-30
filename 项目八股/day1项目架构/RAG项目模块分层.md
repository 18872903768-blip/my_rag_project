# RAG 项目模块分层

本文基于当前仓库的真实源码梳理项目分层，目标是建立全局认知，不展开算法公式和参数细节。

> 范围说明：`*_old.py` 和 `rag_modules_old/` 是历史基线，不属于当前运行链路；`eval/`、`tests/` 属于离线验证，也不参与在线 Query。

## 一、在线运行模块分层

### 1. 服务 / API 层

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `api/app.py` | `create_app()`、`ask()`、`ask_stream()`、`search()`、`traces()`、`healthz()` | 用 FastAPI 将 `RecipeRAGSystem` 封装成同步问答、SSE、纯检索、Trace 查询和健康检查接口，并管理内存会话历史。 |
| `api/schemas.py` | `AskRequest`、`SearchRequest`、`AskResponse`、`SearchResponse`、`TraceResponse` | 定义 HTTP 请求和响应的数据结构、字段约束及默认值。 |
| `api/__init__.py` | 无类 / 函数 | API 包标记文件，仅声明这是位于 RAG 内核外部的轻量 HTTP 层。 |

API 层不实现检索算法，核心调用关系是：

```text
api.app.ask()
→ RecipeRAGSystem.ask_question() / ask_agent()
```

### 2. 系统编排层

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `main.py` | `RecipeRAGSystem`、`initialize_system()`、`build_knowledge_base()`、`retrieve()`、`ask_question()`、`ask_agent()`、`main()` | 创建各模块、选择 Milvus/FAISS 后端，并编排建库、检索、Classic 问答和 Agent 问答的完整调用顺序。 |

这是全项目的中心文件。

`RecipeRAGSystem` 自己不负责文档切块、向量搜索、融合、Prompt 生成或 Agent 节点执行；它负责把这些模块装配起来：

```text
RecipeRAGSystem
├── DataPreparationModule
├── IndexConstructionModule
├── VectorStoreBackend
├── RetrievalOptimizationModule
├── GenerationIntegrationModule
└── RecipeAgent
```

### 3. 数据准备层

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `rag_modules/data_preparation.py` | `DataPreparationModule`、`load_documents()`、`chunk_documents()`、`_enhance_metadata()`、`get_parent_documents()` | 将 Markdown 菜谱加载成父文档，补充分类、难度和权限元数据，再切成可检索子块，并提供子块到父文档的回溯。 |
| `rag_modules/image_ingestion.py` | `ImageIngestionModule`、`ingest_images()`、`_caption_image()`、`_build_chunk()` | 将菜谱图片转换为带 caption、`image_path` 和 `parent_id` 的图片子块，并缓存视觉模型结果。 |

数据准备层的输出统一是 LangChain `Document`：

```python
Document(
    page_content=str,
    metadata=dict,
)
```

- 父文档用于生成。
- 文本子块和图片子块用于检索。

### 4. 索引 / 向量库层

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `rag_modules/index_construction.py` | `IndexConstructionModule`、`setup_embeddings()`、`build_vector_index()`、`save_index()`、`load_index()`、`build_manifest()` | 初始化 Embedding 模型，并负责 FAISS 索引的构建、保存、语料指纹校验和安全加载。 |
| `rag_modules/vector_store.py` | `VectorStoreBackend`、`FaissBackend`、`MilvusVectorStore`、`build_filter_expr()`、`combine_expr()` | 定义统一向量后端接口，并实现 FAISS 适配器以及 Milvus 建库、元数据存储、过滤和检索。 |

两者分工：

```text
index_construction.py
├── Embedding 模型初始化
├── FAISS 原生索引
└── 索引 manifest

vector_store.py
├── 后端抽象接口
├── FaissBackend 适配器
└── MilvusVectorStore
```

`IndexConstructionModule` 仍被 Milvus 路径使用，因为 `main.py` 会通过它取得 Embedding 实例。

### 5. Retrieval 层

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `rag_modules/retrieval_optimization.py` | `RetrievalOptimizationModule`、`hybrid_search()`、`metadata_filtered_search()`、`_rrf_rerank()`、`tokenize_chinese()` | 对上层提供统一检索入口，根据后端选择服务端或客户端混合检索，并负责候选池、过滤、融合及可选重排编排。 |

它位于业务编排和底层向量库之间：

```text
RecipeRAGSystem.retrieve()
        ↓
RetrievalOptimizationModule
        ↓
MilvusVectorStore / FaissBackend
```

职责边界：

- `RetrievalOptimizationModule` 决定检索流程。
- `MilvusVectorStore` 负责真正向 Milvus 发起搜索。
- FAISS 路径的本地稀疏检索和客户端融合位于 `RetrievalOptimizationModule`。
- Milvus 路径的混合搜索与融合位于 `MilvusVectorStore.hybrid_search()`。

### 6. Reranker 层

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `rag_modules/reranker.py` | `RerankerModule`、`_load_model()`、`rerank()` | 可选地加载 Cross-Encoder，对混合检索得到的候选文档重新打分和排序。 |

它由 `main.py` 创建后注入 Retrieval 层：

```text
RecipeRAGSystem._build_reranker()
        ↓
RetrievalOptimizationModule(reranker=...)
        ↓
_apply_rerank()
```

当前 `rerank_enabled=False`，因此它属于“已实现但默认不进入主链”的实验模块。

### 7. Generation 层

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `rag_modules/generation_integration.py` | `GenerationIntegrationModule`、`query_router()`、`query_rewrite()`、`_build_context()`、`generate_basic_answer()`、`generate_step_by_step_answer()`、`format_citations()` | 负责 Classic 管线的 Query 路由与改写，并把父文档构造成 Context，通过 Prompt 调用 LLM 生成答案和来源附录。 |

该层内部包含三类职责：

```text
查询理解
├── query_router()
└── query_rewrite()

上下文与 Prompt
└── _build_context()

答案生成与来源
├── generate_basic_answer()
├── generate_step_by_step_answer()
├── generate_list_answer()
└── format_citations()
```

其中 `generate_list_answer()` 是规则生成，不调用 LLM。

### 8. Agent 层

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `rag_modules/agentic_rag.py` | `AgentState`、`RecipeAgent`、`build_agent_tools()`、`_route_node()`、`_act_node()`、`_reflect_node()`、`_generate_node()`、`_build_graph()` | 用固定 LangGraph 状态机让 LLM 在菜谱检索、图片检索、元数据浏览和直接回答之间选择，并提供有限重写和分级降级。 |

Agent 层不替代 Retrieval 和 Generation，而是调用它们：

```text
RecipeAgent
├── route：选择工具
├── act：调用 RetrievalOptimizationModule
├── reflect：空结果时改写一次
└── generate：调用 GenerationIntegrationModule
```

因此：

```text
Agent 层 = 动态流程控制层
Retrieval 层 = 检索执行层
Generation 层 = 答案生成层
```

## 二、横切辅助层

### 1. 配置与领域配置

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `config.py` | `RAGConfig`、`from_env()`、`from_dict()`、`to_dict()` | 集中保存路径、后端、模型、Top-K、上下文、重排和多模态等运行参数，并从环境变量加载。 |
| `rag_modules/domain_config.py` | `DomainConfig`、`RECIPE_DOMAIN`、`get_domain()`、`set_domain()`、`default_internal_categories()` | 集中保存菜谱领域的分类词汇、Prompt、Agent 工具描述、拒答话术和内部类目。 |

区别：

```text
config.py
→ 运行参数：模型、路径、数量、开关

domain_config.py
→ 领域语义：分类、Prompt、工具描述、用户话术
```

`domain_config.py` 同时被数据层、Generation 层和 Agent 层使用，因此是横切模块。

### 2. Observability

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `rag_modules/observability.py` | `QueryTrace`、`event()`、`finish()`、`append_trace()`、`load_traces()`、`tracing_status()` | 为每次 Query 记录结构化事件、耗时、答案和错误，并持久化到 JSONL。 |

它观察流程，但不改变检索或生成结果。

### 3. 认证

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `api/auth.py` | `Principal`、`AuthError`、`authenticate()`、`mint_token()`、`get_jwt_secret()` | 验证服务间 JWT，并从 Token 中提取可信角色，防止请求体自行提升权限。 |
| `api/mint_token.py` | `main()` | 提供命令行工具，为测试或服务调用生成 JWT。 |

认证层产生 `Principal`，随后 `main.py` 将角色转换成检索层的 `visibility` 过滤条件。

### 4. 包装与懒加载

| 文件 | 核心类 / 函数 | 职责 |
|---|---|---|
| `rag_modules/__init__.py` | `__getattr__()` | 对四个基础模块执行懒加载，避免仅建索引时提前引入 LLM 依赖。 |

当前 `main.py` 大多直接从具体模块导入类，因此该文件主要承担包级兼容和懒加载职责。

## 三、离线评测层

这些文件不会在正常用户 Query 中自动运行。

| 文件 | 核心函数 | 职责 |
|---|---|---|
| `eval/generate_golden_set.py` | `build_items()`、`main()` | 根据真实语料元数据生成并写出种子 Golden Set。 |
| `eval/run_eval.py` | `evaluate_item()`、`main()` | 直接调用 `RecipeRAGSystem.retrieve()`，计算检索命中、排序、过滤和权限隔离结果。 |
| `eval/run_judge.py` | `parse_judge()`、`main()` | 分别运行 Classic 和 Agent 问答，再通过 LLM Judge 评价答案忠实性和相关性。 |

依赖方向：

```text
eval/
  ↓
RecipeRAGSystem
  ↓
现有 RAG 内核
```

RAG 内核不会反向依赖 `eval/`。

## 四、测试与运维辅助代码

### 1. 运维脚本

| 文件 | 核心函数 | 职责 |
|---|---|---|
| `scripts/download_lfs_images.py` | `is_lfs_pointer()`、`download()`、`main()` | 一次性扫描 Git LFS 图片指针并尝试下载真实图片，不属于 RAG 运行链。 |

### 2. 测试文件

| 文件 | 主要验证对象 |
|---|---|
| `tests/conftest.py` | 设置 pytest 项目导入路径。 |
| `tests/test_config.py` | `RAGConfig` 的默认路径和环境变量路径解析。 |
| `tests/test_data_preparation.py` | `DataPreparationModule` 的加载、切块、稳定 ID 和父文档回溯。 |
| `tests/test_image_ingestion.py` | `ImageIngestionModule` 的 caption、缓存、错误跳过和 LFS 指针识别。 |
| `tests/test_index_construction.py` | `IndexConstructionModule` 的 manifest、FAISS 保存加载和完整性校验。 |
| `tests/test_vector_store.py` | 过滤表达式、Milvus Schema、建库复用和文档过滤。 |
| `tests/test_retrieval_optimization.py` | 中文分词、混合检索、融合和候选池行为。 |
| `tests/test_reranker.py` | `RerankerModule` 及其与 Retrieval 的集成。 |
| `tests/test_agentic_rag.py` | Agent 路由、重写、降级和会话历史。 |
| `tests/test_domain_config.py` | `DomainConfig` 的换域能力和领域词汇替换。 |
| `tests/test_citations.py` | `format_citations()` 的来源去重和空结果行为。 |
| `tests/test_observability.py` | `QueryTrace` 的事件记录和 JSONL 读写。 |
| `tests/test_permissions.py` | `visibility` 标记和角色权限过滤。 |
| `tests/test_api.py` | FastAPI、JWT、SSE、会话、搜索和 Trace 接口。 |

这些测试通过 Fake LLM、Fake Embedding 和 Fake Milvus 隔离外部依赖，不属于生产模块。

## 五、历史基础版文件

以下文件明确不属于当前活跃运行链：

```text
main_old.py
config_old.py
rag_modules_old/
```

| 文件 | 历史类 / 函数 | 职责 |
|---|---|---|
| `main_old.py` | 旧 `RecipeRAGSystem`、旧 `main()` | 基础版单后端、固定管线编排。 |
| `config_old.py` | 旧 `RAGConfig` | 基础版少量静态配置。 |
| `rag_modules_old/data_preparation.py` | 旧 `DataPreparationModule` | 基础版文档加载与切块。 |
| `rag_modules_old/index_construction.py` | 旧 `IndexConstructionModule` | 基础版 FAISS 索引。 |
| `rag_modules_old/retrieval_optimization.py` | 旧 `RetrievalOptimizationModule` | 基础版检索组合。 |
| `rag_modules_old/generation_integration.py` | 旧 `GenerationIntegrationModule` | 基础版 Query 路由、改写和答案生成。 |
| `rag_modules_old/__init__.py` | 包导出 | 基础版模块导出。 |

`pyproject.toml` 明确将这些文件排除在当前 Ruff 和 mypy 检查之外，因此应在“基础版 vs 升级版”阶段阅读，而不是当前架构主线。

## 六、核心文件优先级

### 第一优先级：必须掌握

1. **`main.py`**
   - 核心对象：`RecipeRAGSystem`
   - 原因：决定所有模块什么时候创建、怎么连接，以及 Query 的真实执行顺序。

2. **`rag_modules/data_preparation.py`**
   - 核心对象：`DataPreparationModule`
   - 原因：父文档、子块、元数据和父子回溯决定检索对象和生成对象为什么不同。

3. **`rag_modules/retrieval_optimization.py`**
   - 核心对象：`RetrievalOptimizationModule`
   - 原因：统一 Retrieval 入口，连接上层业务和底层 Milvus/FAISS。

4. **`rag_modules/vector_store.py`**
   - 核心对象：`VectorStoreBackend`、`MilvusVectorStore`、`FaissBackend`
   - 原因：决定检索最终在哪里执行，以及两个后端为什么不是完全等价的。

5. **`rag_modules/generation_integration.py`**
   - 核心对象：`GenerationIntegrationModule`
   - 原因：解释检索结果如何变成 Context、Prompt、Answer 和来源引用。

### 第二优先级：理解升级能力

6. **`rag_modules/agentic_rag.py`**
   - 理解 Agent 如何复用 Retrieval 和 Generation，以及 Harness 如何限制 LLM。

7. **`rag_modules/index_construction.py`**
   - 理解 Embedding 初始化、FAISS 索引和索引一致性。

8. **`api/app.py`**
   - 理解 RAG 内核如何常驻并通过 HTTP/SSE 对外提供服务。

### 第三优先级：横切能力

9. **`config.py` + `rag_modules/domain_config.py`**
   - 理解运行参数与领域语义如何解耦。

10. **`rag_modules/observability.py` + `eval/`**
    - 理解如何发现 badcase，以及如何证明升级有效。

## 七、纯文本架构图

```text
┌─────────────────────────────────────────────────────────────────────┐
│                           调用方 / 用户                              │
│                    CLI                     HTTP/SSE                  │
└─────────────────────┬─────────────────────────┬─────────────────────┘
                      │                         │
                      │                  ┌──────▼──────────────┐
                      │                  │ 服务 / API 层        │
                      │                  │ api/app.py           │
                      │                  │ api/schemas.py       │
                      │                  │ api/auth.py          │
                      │                  └──────┬──────────────┘
                      │                         │
                      └──────────────┬──────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ 系统编排层            │
                          │ main.py               │
                          │ RecipeRAGSystem       │
                          └───────┬───────┬──────┘
                                  │       │
              ┌───────────────────┘       └──────────────────────┐
              │                                                  │
     ┌────────▼──────────┐                              ┌────────▼─────────┐
     │ 建库 / 数据链路    │                              │ Query / 问答链路 │
     └────────┬──────────┘                              └────────┬─────────┘
              │                                                  │
     ┌────────▼─────────────────┐                       ┌────────▼──────────────┐
     │ 数据准备层                │                       │ Classic 或 Agent       │
     │ data_preparation.py      │                       │                       │
     │ image_ingestion.py       │                       │ Classic: main.py      │
     │                          │                       │ Agent: agentic_rag.py │
     │ Markdown → 父文档 → 子块 │                       └────────┬──────────────┘
     │ 图片 → caption 子块      │                                │
     └────────┬─────────────────┘                                │
              │                                                  │
     ┌────────▼──────────────────────┐                  ┌────────▼────────────────┐
     │ 索引 / 向量库层               │                  │ Retrieval 层             │
     │ index_construction.py         │◄─────────────────┤ retrieval_optimization.py│
     │ vector_store.py               │                  │ hybrid/filtered search   │
     │                               │                  └────────┬────────────────┘
     │ Embedding                     │                           │
     │ FAISS 或 Milvus               │                           │
     └───────────────────────────────┘                  ┌────────▼──────────────┐
                                                      │ 可选 Reranker 层       │
                                                      │ reranker.py           │
                                                      └────────┬──────────────┘
                                                               │
                                                      ┌────────▼──────────────┐
                                                      │ 父文档回溯            │
                                                      │ get_parent_documents │
                                                      └────────┬──────────────┘
                                                               │
                                                      ┌────────▼──────────────┐
                                                      │ Generation 层         │
                                                      │ generation_           │
                                                      │ integration.py        │
                                                      │ Context → Prompt      │
                                                      │ → LLM → Citation      │
                                                      └────────┬──────────────┘
                                                               │
                                                      ┌────────▼──────────────┐
                                                      │ 最终 Answer           │
                                                      └───────────────────────┘


横切所有层：
────────────────────────────────────────────────────────────────────────
config.py                运行配置
domain_config.py         领域词汇、Prompt、工具定义、话术
observability.py         query_id、事件、耗时、错误、JSONL Trace
api/auth.py              JWT 身份与角色
────────────────────────────────────────────────────────────────────────

离线验证：
eval/  ────────────────► 调用 RecipeRAGSystem 做检索与答案评测
tests/ ────────────────► 对各层执行隔离单元测试
```

## 八、一句话全局认知

```text
API 接收请求
→ main.py 编排
→ Retrieval 找子块
→ data_preparation 回溯父文档
→ Generation 构造上下文并生成
→ Agent 可选择性地控制上述流程
→ Observability 记录全过程
```

本文没有标注为“推断”的职责；上述分层均能从当前源码的导入关系、类定义和函数调用中直接确认。
