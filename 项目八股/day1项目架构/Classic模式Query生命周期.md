# Classic 模式 Query 生命周期

本文基于当前仓库真实源码，完整追踪一次 Classic 模式下的用户 Query，从 FastAPI 接收问题开始，直到最终 Answer 返回。

本文以同步接口 `POST /api/ask`、`pipeline="classic"` 为主线，同时标出 `list/detail/general`、无结果、FAISS/Milvus、同步/流式等条件分支。

## A. 真实函数调用链

### 0. 服务启动：RAG 已在请求前完成初始化

**文件**

`api/app.py`

**函数**

`create_app()` 内的 `lifespan()`

```text
create_app()
→ lifespan()
→ RecipeRAGSystem()
→ initialize_system(load_generation=True)
→ build_knowledge_base()
→ app.state.rag = system
```

因此用户请求到达时，通常已经存在：

```python
app.state.rag: RecipeRAGSystem
rag.retrieval_module: RetrievalOptimizationModule
rag.generation_module: GenerationIntegrationModule
rag.backend: MilvusVectorStore | FaissBackend
```

重要条件：

> FAISS 或 Milvus 在服务启动建库时已经选定，不是每条 Query 临时选择。

---

### 1. API 接收 Classic 请求

**文件**

`api/app.py`

**函数**

`ask()`

HTTP 请求：

```http
POST /api/ask
Content-Type: application/json
```

请求体示例：

```json
{
  "query": "简易红烧肉怎么做",
  "pipeline": "classic",
  "role": "user"
}
```

Pydantic 数据结构：

```python
AskRequest(
    query: str,
    role: str | None,
    pipeline: Literal["classic", "agent"],
    session_id: str | None,
)
```

API 依赖首先执行：

```text
get_rag()
get_principal()
authenticate()
effective_role()
```

实际角色的产生规则：

- 启用 JWT 时：以 Token 中的 `role` 为准。
- 开发模式且请求提供 `role` 时：允许使用请求体角色。
- 请求体不能在正式鉴权模式下提升权限。

随后生成：

```python
query_id = _new_query_id()
```

条件分支：

```python
if body.pipeline == "agent":
    rag.ask_agent(...)
else:
    rag.ask_question(...)
```

Classic 路径调用：

```python
answer = rag.ask_question(
    body.query,
    stream=False,
    role=role,
    query_id=query_id,
)
```

---

### 2. 进入系统编排层

**文件**

`main.py`

**类 / 函数**

```python
RecipeRAGSystem.ask_question()
```

输入：

```python
question: str
stream: bool = False
role: str
query_id: str
```

首先检查：

```python
self.retrieval_module is not None
self.generation_module is not None
```

如果系统没有完成建库或生成模块未初始化，会抛出异常。

---

### 3. 创建 QueryTrace

**文件**

- `main.py`
- `rag_modules/observability.py`

**类**

```python
QueryTrace
```

创建位置：

```python
trace = QueryTrace(
    question,
    role=role,
    pipeline="classic",
    query_id=query_id,
)
```

Trace 初始数据：

```python
{
    "query_id": str,
    "question": str,
    "role": str,
    "pipeline": "classic",
    "started_at": datetime,
    "events": [],
    "answer": None,
    "error": None,
}
```

下一步：

```python
self._ask_question_inner(
    question,
    stream=stream,
    role=role,
    trace=trace,
)
```

---

### 4. Query Router 判断问题类型

**文件**

`rag_modules/generation_integration.py`

**函数**

```python
GenerationIntegrationModule.query_router()
```

调用位置：

```python
route_type = self.generation_module.query_router(question)
```

输入：

```python
query: str
```

内部调用链：

```text
get_domain().router_prompt
→ ChatPromptTemplate.from_template()
→ Prompt
→ self.llm
→ StrOutputParser
→ chain.invoke(query)
```

LLM 被要求返回三种结果之一：

```python
"list"
"detail"
"general"
```

| `route_type` | 业务含义 | 后续生成方式 |
|---|---|---|
| `list` | 推荐、浏览、只需要条目名称 | 规则生成列表 |
| `detail` | 制作步骤、食材、详细做法 | 分步骤 Generation |
| `general` | 技巧、介绍或其他一般问题 | 基础 Generation |

如果 LLM 返回其他内容：

```python
return "general"
```

路由事件写入：

```python
trace.event("route", route=route_type)
```

> Classic 模式的每个 Query 都会先调用一次 LLM 做路由。

---

### 5. 判断是否执行 Query Rewrite

**文件**

`main.py`

条件分支：

```python
rewritten_query = (
    question
    if route_type == "list"
    else self.generation_module.query_rewrite(question)
)
```

#### `route_type == "list"`

```python
rewritten_query = question
```

不会调用 Query Rewrite LLM。

#### `route_type == "detail"` 或 `"general"`

调用：

```python
GenerationIntegrationModule.query_rewrite(question)
```

**文件**

`rag_modules/generation_integration.py`

内部流程：

```text
rewrite_prompt
→ PromptTemplate
→ self.llm
→ StrOutputParser
→ chain.invoke(query)
```

输出：

```python
rewritten_query: str
```

Prompt 可能：

- 返回改写后的 Query。
- 判断原 Query 足够明确，原样返回。

项目没有生成多个查询，因此这里是单 Query Rewrite，不是 Multi-Query Expansion。

记录事件：

```python
trace.event("rewrite", query=rewritten_query)
```

---

### 6. 产生 metadata filter

**文件**

`main.py`

**函数**

```python
RecipeRAGSystem._extract_filters_from_query()
```

调用位置：

```python
filters = self._extract_filters_from_query(question)
```

注意输入是原始 `question`，而不是 `rewritten_query`。

它从原问题中识别两类条件：

```python
category
difficulty
```

分类和难度词汇来自：

```text
DataPreparationModule.get_supported_categories()
DataPreparationModule.get_supported_difficulties()
→ domain_config.get_domain()
```

示例：

```python
"推荐几道简单的汤品"
```

可能产生：

```python
{
    "category": "汤品",
    "difficulty": "简单",
}
```

如果未识别到：

```python
filters = {}
```

---

### 7. 调用 retrieve 并产生 visibility filter

**文件**

`main.py`

**函数**

```python
RecipeRAGSystem.retrieve()
```

调用：

```python
relevant_chunks = self.retrieve(
    rewritten_query,
    filters=self._extract_filters_from_query(question),
    role=role,
)
```

输入：

```python
query: str                  # rewritten_query
top_k: int | None
filters: dict[str, Any]     # 从原始问题提取
role: str
```

首先确定最终数量：

```python
limit = top_k or self.config.top_k
```

然后产生权限过滤：

```python
visibility_expr = self._visibility_expr_for_role(role)
```

角色映射：

```python
guest → ["public"]
user  → ["public"]
staff → ["public", "internal"]
```

因此：

#### guest/user

```python
visibility_expr = 'visibility in ["public"]'
```

#### staff

```python
visibility_expr = None
```

因为 staff 可以访问全部可见范围，无需过滤表达式。未知角色会抛出 `ValueError`。

---

### 8. retrieve 选择检索方法

`retrieve()` 中的判断：

```python
active_filters = (
    filters
    if filters is not None
    else self._extract_filters_from_query(query)
)
```

Classic 调用总是显式传入一个 `dict`，即使是空字典，因此这里不会再次从 rewritten Query 提取过滤条件。

随后：

```python
if active_filters or visibility_expr:
    return self.retrieval_module.metadata_filtered_search(...)
return self.retrieval_module.hybrid_search(...)
```

#### 进入 `metadata_filtered_search()` 的条件

满足任意一个：

```text
存在 category/difficulty metadata filter
或
存在 visibility filter
```

所以：

- `guest/user` 基本总会进入 `metadata_filtered_search()`。
- `staff` 且没有 category/difficulty 时，进入 `hybrid_search()`。
- `staff` 但 Query 包含分类或难度时，进入 `metadata_filtered_search()`。

这不是按 Query 内容决定是否使用混合检索：

> 两个方法底层都使用混合检索，区别主要是是否附加 metadata/visibility 限制。

---

### 9. Retrieval 进入 Milvus 或 FAISS

**文件**

`rag_modules/retrieval_optimization.py`

**类**

```python
RetrievalOptimizationModule
```

该对象在建库时已经绑定：

```python
self.vectorstore: VectorStoreBackend
```

可能是：

```python
MilvusVectorStore
FaissBackend
```

判断属性：

```python
self.server_side_hybrid = getattr(
    vectorstore,
    "supports_server_side_hybrid",
    False,
)
```

#### 分支 A：Milvus

```python
server_side_hybrid = True
```

有过滤条件：

```text
metadata_filtered_search()
→ build_filter_expr(filters)
→ combine_expr(metadata_expr, visibility_expr)
→ MilvusVectorStore.hybrid_search()
```

无过滤条件：

```text
hybrid_search()
→ MilvusVectorStore.hybrid_search()
```

真正调用 Milvus 的位置是 `rag_modules/vector_store.py` 中的：

```python
MilvusVectorStore.hybrid_search()
```

内部完成：

```text
Query Embedding
→ Dense Search Request
→ Sparse Search Request
→ Hybrid Search
→ Fusion
→ 返回 Top-K
```

过滤表达式会挂到两路检索请求上。

#### 分支 B：FAISS

```python
server_side_hybrid = False
```

调用链：

```text
RetrievalOptimizationModule.hybrid_search()
├── FaissBackend.similarity_search()
└── 本地 bm25_retriever.invoke()
        ↓
客户端融合
        ↓
可选 Reranker
```

如果使用 `metadata_filtered_search()`：

```text
先扩大候选池执行混合检索
→ 再检查 category/difficulty
→ 再检查 visibility expression
→ 收集到 top_k 后停止
```

所以两个后端的过滤位置不同：

```text
Milvus：过滤进入后端检索请求
FAISS：召回候选后在 Python 中过滤
```

---

### 10. Reranker 条件分支

**文件**

`rag_modules/retrieval_optimization.py`

**函数**

```python
_apply_rerank()
```

条件：

```python
self.reranker is not None
```

默认配置：

```python
rerank_enabled = False
reranker = None
```

因此：

```python
return documents[:top_k]
```

开启重排时：

```text
扩大召回池
→ RerankerModule.rerank()
→ 与原召回顺序再次组合
→ 截取 Top-K
```

所以并非所有 Classic Query 都经过 Cross-Encoder。

---

### 11. 检索返回 Child Chunk

Retrieval 的最终输出：

```python
relevant_chunks: list[Document]
```

其中每个 `Document` 是 Child Chunk：

```python
metadata["doc_type"] == "child"
metadata["chunk_id"] == ...
metadata["parent_id"] == ...
```

它可能是文本块：

```python
metadata["modality"] == "text"
```

也可能是图片 caption 块：

```python
metadata["modality"] == "image"
```

检索阶段不会直接返回完整 Parent Document。

记录事件：

```python
trace.event("retrieve", hits=len(relevant_chunks))
```

---

### 12. 无结果条件分支

**文件**

`main.py`

```python
if not relevant_chunks:
    trace.event("no_results")
    return get_domain().classic_no_results
```

无结果时：

- 不执行父文档回溯。
- 不构造 Context。
- 不调用最终答案生成 LLM。
- 不追加 citation。
- 直接返回固定拒答字符串。

但是前面的 Router LLM，以及非 `list` 路径的 Rewrite LLM，已经调用过。

---

### 13. Child Chunk 转换为 Parent Document

**文件**

`rag_modules/data_preparation.py`

**函数**

```python
DataPreparationModule.get_parent_documents()
```

调用：

```python
relevant_documents = self.data_module.get_parent_documents(
    relevant_chunks
)
```

输入：

```python
list[Document]  # Child Chunks
```

处理：

1. 从每个 Child 读取 `parent_id`。
2. 在 `_parent_documents` 中查找完整 Parent。
3. 统计每个 Parent 拥有多少命中 Child。
4. Parent 去重。
5. 按命中 Child 数量和首次出现位置排序。

输出：

```python
relevant_documents: list[Document]
```

其中：

```python
metadata["doc_type"] == "parent"
page_content == 完整 Markdown 原文
```

同时从 Child Chunk 中收集图片路径：

```python
image_paths = self._collect_image_paths(relevant_chunks)
```

只处理：

```python
metadata["modality"] == "image"
```

记录：

```python
trace.event(
    "parents",
    count=len(relevant_documents),
    images=len(image_paths),
)
```

---

### 14. 根据 route_type 选择 Generation 方法

**文件**

`main.py`

**函数**

`_ask_question_inner()`

#### 分支 1：`route_type == "list"`

```python
return self.generation_module.generate_list_answer(
    question,
    relevant_documents,
)
```

特点：

- 从 Parent metadata 提取 `dish_name`。
- 规则拼接列表。
- 不构造完整 Context。
- 不调用最终生成 LLM。
- 不调用 `format_citations()`。

但 Query Router LLM 已调用。

#### 分支 2：`route_type == "detail"`

同步 API：

```python
generate_step_by_step_answer(
    question,
    relevant_documents,
    image_paths=image_paths,
)
```

流式 API：

```python
generate_step_by_step_answer_stream(
    question,
    relevant_documents,
)
```

用于具体制作方法、食材和步骤。

#### 分支 3：`route_type == "general"`

同步 API：

```python
generate_basic_answer(
    question,
    relevant_documents,
    image_paths=image_paths,
)
```

流式 API：

```python
generate_basic_answer_stream(
    question,
    relevant_documents,
)
```

用于一般性介绍、技巧等问题。

> 检索使用 rewritten Query，但最终 Generation 接收的是用户原始 question。

---

### 15. 构建 Context

**文件**

`rag_modules/generation_integration.py`

**函数**

```python
GenerationIntegrationModule._build_context()
```

`detail` 和 `general` 路径进入：

```python
context = self._build_context(
    context_docs,
    self.max_context_chars,
    image_paths,
)
```

输入：

```python
docs: list[Document]       # Parent Documents
max_length: int
image_paths: list[str] | None
```

每个 Parent 被格式化为：

```text
【食谱 1】 菜名 | 分类: ... | 难度: ...
完整父文档内容
```

多个 Parent 之间增加分隔符。

如果存在图片路径，同步生成路径会在 Context 末尾加入：

```text
【相关图片（可在回答中提示用户查看）】
- image/path/...
```

输出：

```python
context: str
```

条件差异：

- `list` 不调用 `_build_context()`。
- 无结果不调用 `_build_context()`。
- 当前 Classic 流式生成方法没有接收 `image_paths`，因此不会把图片路径加入流式 Context。

---

### 16. 构建 Generation Prompt

**文件**

- `rag_modules/generation_integration.py`
- `rag_modules/domain_config.py`

#### `detail`

```python
ChatPromptTemplate.from_template(
    get_domain().step_by_step_prompt
)
```

#### `general`

```python
ChatPromptTemplate.from_template(
    get_domain().basic_answer_prompt
)
```

Prompt 输入：

```python
{
    "question": 原始用户问题,
    "context": context字符串,
}
```

输出是发送给聊天模型的消息结构。

---

### 17. LLM 真正被调用的位置

**文件**

`rag_modules/generation_integration.py`

同步 Generation 链：

```python
chain = (
    {
        "question": RunnablePassthrough(),
        "context": lambda _: context,
    }
    | prompt
    | self.llm
    | StrOutputParser()
)

response = chain.invoke(query)
```

真正触发最终答案 LLM 请求的是：

```python
chain.invoke(query)
```

流式生成则是：

```python
chain.stream(query)
```

需要区分三次可能的 LLM 调用：

| 调用 | 是否每个 Classic Query 都执行 |
|---|---|
| `query_router()` 中的 `chain.invoke()` | 是 |
| `query_rewrite()` 中的 `chain.invoke()` | 非 `list` 执行 |
| 最终答案 Generation | `detail/general` 且检索有结果时执行 |

所以：

- `list`：通常只调用 Router LLM。
- 无结果的 `detail/general`：调用 Router 和 Rewrite，但不调用最终 Generation。
- 有结果的 `detail/general`：通常调用三次 LLM。

---

### 18. 追加 Citation

**文件**

`rag_modules/generation_integration.py`

**函数**

```python
format_citations()
```

同步生成返回：

```python
return response + format_citations(context_docs)
```

Citation 来自实际 Parent Documents：

```python
dish_name
source_path
```

输出格式：

```text
——
📚 以上回答参考自：
- 菜名（语料相对路径）
- 菜名（语料相对路径）
```

系统有两种来源表示：

#### 内联 Citation

Prompt 要求 LLM 使用：

```text
【食谱 N】
```

这是 LLM 生成行为，不保证一定正确执行。

#### 确定性来源附录

由 `format_citations()` 根据 Parent Documents 生成，不依赖 LLM 自己记住来源。

条件分支：

- `detail/general` 同步：追加 citation。
- `detail/general` 流式：生成结束后 `yield citations`。
- `list`：不追加 citation。
- 无结果：不追加 citation。

---

### 19. QueryTrace 结束

**文件**

- `main.py`
- `rag_modules/observability.py`

同步路径得到字符串后：

```python
trace.finish(
    answer=str(answer) if isinstance(answer, str) else None
)
```

`QueryTrace.finish()` 会：

1. 保存 `answer` 或 `error`。
2. 计算 `elapsed_ms`。
3. 转换为 `dict`。
4. 调用 `append_trace()`。
5. 追加写入 `logs/query_trace.jsonl`。

异常分支：

```python
except Exception as exc:
    trace.event("pipeline_error", error=str(exc))
    trace.finish(error=exc)
    return f"处理问题时出错: {exc}"
```

Classic 异常不会重新抛给 API，而是转换为错误字符串。

#### 流式路径的重要区别

`stream=True` 时，`_ask_question_inner()` 返回的是生成器。

因此：

```text
ask_question()
→ trace.finish(answer=None)
→ 返回生成器
→ API 才开始迭代生成器
```

当前 Classic 流式路径的 QueryTrace 会在真正流式 LLM 生成完成前结束，不能完整记录后续生成错误和答案正文。这是当前真实实现的可观测性边界。

---

### 20. 最终 Answer 返回 API

**文件**

`api/app.py`

同步 Classic：

```python
answer = rag.ask_question(...)

return AskResponse(
    query_id=query_id,
    answer=str(answer),
    pipeline="classic",
)
```

响应结构：

```json
{
  "query_id": "xxxxxxxxxxxx",
  "answer": "最终答案及来源附录",
  "pipeline": "classic",
  "route": null,
  "hits": [],
  "citations": [],
  "error": null
}
```

注意：

- Classic 的 citation 被追加在 `answer` 字符串中。
- `AskResponse.citations` 的结构化字段在 Classic 路径保持空列表。
- Classic API 不返回 `route_type`。
- Classic API 不返回命中的菜名列表。

最终由 FastAPI 的 `UTF8JSONResponse` 序列化为 UTF-8 JSON。

---

## B. 面试时应该讲的业务流程链

### 2 分钟口述版本

用户通过 FastAPI 的 `/api/ask` 提交问题后，API 首先完成请求校验和 JWT 角色解析，再把 Query、角色和 query_id 交给常驻内存的 `RecipeRAGSystem`。Classic 管线会先创建 QueryTrace，然后用 LLM 把问题分成列表、详情和一般问答三类。列表问题保持原 Query，其他问题再经过一次 Query Rewrite，但改写只用于检索，最终回答仍使用用户原问题。

检索前，系统从原始问题中提取分类和难度条件，并根据用户角色生成 visibility 权限过滤。只要存在业务 metadata 或权限限制，就进入带过滤的混合检索；否则进入普通混合检索。实际使用 Milvus 还是 FAISS，是服务启动建库时已经确定的。Milvus 在后端执行检索与过滤，FAISS 则组合本地向量检索和稀疏检索，并在客户端完成部分过滤。

检索返回的是 Child Chunk，不是完整菜谱。系统通过每个 Child 的 `parent_id` 回溯 Parent Document，并按照命中子块数量对 Parent 去重排序。详情和一般问答会把这些完整 Parent 构造成有长度限制的 Context，再选择对应 Prompt 调用 LLM；列表请求则直接从 Parent metadata 中生成菜名列表，不再调用最终生成 LLM。

最终答案除 LLM 的内联食谱编号外，还会根据实际 Parent Document 确定性追加来源路径。完成后，QueryTrace 把路由、改写、命中数量、父文档数量、耗时、答案或错误写入 JSONL，API 再把答案封装为 `AskResponse` 返回给调用方。

### 一句话压缩

```text
API 鉴权
→ Query 路由/条件改写
→ 业务过滤与权限过滤
→ 后端混合检索 Child
→ parent_id 回溯 Parent
→ Context + Route Prompt
→ LLM Generation
→ Citation + Trace
→ API Answer
```

---

## C. 节点类型分类

| 节点 | 源码对象 | 类型 | 产生或执行什么 |
|---|---|---|---|
| HTTP 请求 | `AskRequest` | 数据结果 / Pydantic 对象 | 保存 Query、pipeline、role、session_id |
| API 入口 | `ask()` | 函数 | 校验身份、选择 Classic/Agent |
| API 应用 | `FastAPI` | 类实例 | 注册接口和管理 lifespan |
| RAG 总控 | `RecipeRAGSystem` | 类实例 | 持有各模块并编排完整流程 |
| Classic 入口 | `ask_question()` | 函数 | 创建 Trace、运行 Classic 链 |
| Trace | `QueryTrace` | 类实例 | 累积事件、答案、错误和耗时 |
| Query 路由 | `query_router()` | 函数 + LLM 步骤 | 产生 `route_type` |
| `route_type` | `"list"/"detail"/"general"` | 数据结果 | 决定 Rewrite 和 Generation 分支 |
| Query Rewrite | `query_rewrite()` | 函数 + LLM 步骤 | 产生单个 `rewritten_query` |
| `rewritten_query` | `str` | 数据结果 | 作为 Retrieval Query |
| metadata 提取 | `_extract_filters_from_query()` | 函数 | 产生分类、难度过滤字典 |
| metadata filters | `dict[str, str]` | 数据结果 | 限制分类和难度 |
| 权限表达式 | `_visibility_expr_for_role()` | 函数 | 把角色转换成可见范围 |
| visibility expr | `str | None` | 数据结果 | 限制 public/internal |
| 检索入口 | `retrieve()` | 函数 | 选择 filtered 或普通检索 |
| 检索编排 | `RetrievalOptimizationModule` | 类实例 | 统一编排后端、融合、过滤和重排 |
| 普通检索 | `hybrid_search()` | 函数 + 算法步骤 | 执行混合召回 |
| 过滤检索 | `metadata_filtered_search()` | 函数 + 算法步骤 | 附加 metadata/visibility 约束 |
| Milvus 后端 | `MilvusVectorStore` | 类实例 | 执行服务端检索 |
| FAISS 后端 | `FaissBackend` | 类实例 | 执行本地向量检索 |
| Dense Retrieval | 后端搜索调用 | 算法步骤 | 召回语义相关 Child |
| Sparse Retrieval | Milvus / 本地 Retriever | 算法步骤 | 召回字面相关 Child |
| Fusion | Milvus ranker / `_rrf_rerank()` | 算法步骤 | 合并两路排名 |
| Reranking | `RerankerModule.rerank()` | 函数 + 可选算法步骤 | 对候选池重新排序 |
| Top-K Child | `list[Document]` | 数据结果 | 最终检索命中的 Child Chunk |
| Parent 回溯 | `get_parent_documents()` | 函数 | 根据 `parent_id` 去重、排序并取回全文 |
| Parent Documents | `list[Document]` | 数据结果 | 用于最终 Context 的完整菜谱 |
| 图片路径收集 | `_collect_image_paths()` | 函数 | 从图片 Child 中提取 `image_path` |
| Context 构造 | `_build_context()` | 函数 | 把 Parent 格式化为受长度限制的字符串 |
| Context | `str` | 数据结果 | 提供给最终 Prompt 的参考内容 |
| Prompt 模板 | `ChatPromptTemplate` | 类实例 | 组合原始问题与 Context |
| 最终生成 | `chain.invoke()` / `chain.stream()` | LLM 执行步骤 | 真正请求生成模型 |
| 模型原始回答 | `response: str` | 数据结果 | 尚未追加确定性来源附录 |
| Citation | `format_citations()` | 函数 | 根据 Parent 生成来源附录 |
| 最终 Answer | `str` | 数据结果 | 模型回答与 citation 的组合 |
| Trace 持久化 | `finish()`、`append_trace()` | 函数 | 写入 JSONL |
| API 响应 | `AskResponse` | 数据结果 / Pydantic 对象 | 将最终答案返回客户端 |

最需要记住的边界是：

```text
函数：负责执行动作
类：持有状态或封装模块能力
数据结果：在节点之间流动
算法步骤：发生在函数内部，不一定对应独立类
```

Classic 管线并不是所有 Query 都执行同一套步骤：

```text
list
→ Router → Retrieval → Parent → 规则列表

detail/general 且有结果
→ Router → Rewrite → Retrieval → Parent → Context → LLM → Citation

detail/general 但无结果
→ Router → Rewrite → Retrieval → 固定拒答

FAISS/Milvus
→ 共享上层流程，但底层检索与过滤位置不同
```

## 事实边界

- 上述调用链、条件分支、类、函数和数据结构均可从当前源码直接确认。
- 本文没有假设所有 Query 都经过 Rewrite、最终 LLM、Citation 或 Reranker。
- 后端在服务启动建库阶段确定，Query 阶段只通过已经装配好的 `RetrievalOptimizationModule` 进入对应后端。
- Classic 同步和流式路径的 QueryTrace 结束时机不同；流式 Trace 当前会早于生成器实际消费结束。
