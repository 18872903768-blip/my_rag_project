# Recipe RAG（V1）

面向客服问答场景的中文菜谱 RAG 系统：**Milvus 混合检索 + 多模态图片索引 + 元数据级权限 + LangGraph Agentic RAG + 全链路追踪 + 可量化评测**。纯 CPU 运行，不训练模型、不需要显卡。

## V1 架构

```
Markdown 菜谱 ──┬─ 父子块切分（header 级子块，确定性 chunk_id）
               ├─ 图片 caption 化（DeepSeek-Vision，caption 缓存，LFS 指针自动跳过）
               ▼
        Milvus 2.5 collection（HNSW dense + BM25 sparse[jieba] + 标量字段 + category 分区）
               ▼
     ┌─ 固定管线（classic）：路由 → 改写 → 混合检索 → 父文档回溯 → 生成
     └─ Agentic 管线（LangGraph）：
          START → route(LLM 选工具) → act(执行) → reflect(空结果→改写重试一次)
                → generate(生成/拒答)，各节点分级兜底
               ▼
     query_id 贯穿的 JSONL 追踪 + LangSmith（可选）+ 权限过滤（guest/user/staff × public/internal）
```

## V1 能力清单

- **向量库**：Milvus standalone（默认），dense（bge-small-zh + HNSW/COSINE）+ sparse（内置 BM25，jieba 分词）混合检索，RRF 融合在服务端完成；标量字段建索引，`category` 为分区键；FAISS 后端保留为离线降级路径（`RAG_BACKEND=faiss`），客户端 RRF 兜底。
- **多模态**：菜品图片经 DeepSeek 视觉模型生成中文 caption 后作为 image chunk 入库（`modality=image` + `image_path`），支持文搜图（`--retrieve-images`）、图片检索结果回注生成上下文；caption 按（路径+文件哈希+模型+prompt 版本）落盘缓存，重建索引不重复计费。
- **权限**：`visibility` 元数据（`半成品` 等内部类目默认 internal）+ 查询级角色（guest/user/staff）→ Milvus 表达式下推强制过滤，普通用户查内部内容只返回公开菜谱（优雅降级而非报错）。
- **Agentic RAG**：LangGraph StateGraph，LLM function-calling 在 `search_recipes / search_images / list_by_metadata / direct_answer` 四个工具间路由；反思节点对空结果自动改写重查一次，仍无结果则拒答。
- **多轮对话记忆**：agent 链路支持会话历史——路由与生成节点注入最近 3 轮对话（解决"它/第二道"类指代）；CLI 交互模式自动维护，API 侧传 `session_id` 即启用（内存会话存储，最近 3 轮 + FIFO 驱逐）。
- **答案引用溯源**：双层设计——prompt 指示 LLM 内联标注【食谱 N】，同时由检索到的父文档确定性生成"📚 参考来源"附录（菜名+语料路径），生成降级时引用依然可验证；API 的 agent 应答带结构化 `citations` 字段。
- **重排（实验性，默认关闭）**：`rag_modules/reranker.py` 集成 bge-reranker-base cross-encoder 精排（离线优先加载 + hf-mirror 下载回退）。**评测驱动的决策**：73 条 golden set 实测 RRF 基线 MRR 0.931 > CE+RRF 二次融合 0.902 > 纯 CE 0.879（本语料 BM25 精确菜名信号已足够强），故默认 `RAG_RERANK_ENABLED=false`，保留开关供换语料复测。
- **错误兜底（分级）**：路由失败 → 默认检索；工具失败 → 退回混合检索；检索不可用 → 无 LLM 的规则摘要；生成失败 → 返回检索条目摘要。每次降级都记录到追踪事件。
- **追踪**：每次查询生成 `query_id`，节点事件（route/act/rewrite/fallback/…）追加写入 `logs/query_trace.jsonl`，可按 query_id 回溯任意 badcase；配置 LangSmith 环境变量即获得全链路 trace。
- **评测**：73 条 golden set（详情/关键词/浏览/权限正反向五类意图），`eval/run_eval.py` 输出 Hit@k、MRR、过滤精确率、权限泄漏数；`eval/run_judge.py` 用 DeepSeek 做 LLM-judge，对 classic 与 agent 两条链路打 faithfulness / relevance 分。
- **工程化**：确定性 ID（parent/revision/chunk）、语料指纹 manifest（Milvus 与 FAISS 通用）、JSON 替代 pickle、原子写、离线建库（无 key 可建库/检索）、嵌入模型离线优先加载（代理断开/断网不阻塞启动，`RAG_HF_OFFLINE=true` 可锁死）。
- **领域配置层**：所有领域耦合内容（分类/难度词汇、全部 LLM prompt、agent 工具描述、拒答话术、内部类目）集中在 `rag_modules/domain_config.py` 的 `DomainConfig`；检索引擎与领域解耦，切换知识库领域（如英语学习库）只需新写一份 DomainConfig + 对应解析器（`tests/test_domain_config.py` 用英语领域实测了整套替换）。

## 环境与依赖

- Python 3.12；Milvus 2.5.x（Docker standalone，默认 `localhost:19530`）。
- 参见 `.env.example`；最低要求只需 `DEEPSEEK_API_KEY`。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

uv 用户：`uv sync --group dev`（新增依赖后建议 `uv lock`）。

## 常用命令

```powershell
# 建库/校验（含图片 caption，离线可跑但图片需 API key）
.\.venv\Scripts\python.exe main.py --build-only

# 混合检索（带权限）
.\.venv\Scripts\python.exe main.py --retrieve "红烧肉怎么做" --role user --top-k 5

# 文搜图
.\.venv\Scripts\python.exe main.py --retrieve-images "四季豆炒肉末 成品图"

# Agentic RAG 一次问答
.\.venv\Scripts\python.exe main.py --agent "我想看一眼红烧肉的成品长什么样"

# 交互模式（--chat-mode agent 切换 LangGraph 管线）
.\.venv\Scripts\python.exe main.py --chat-mode agent --role staff

# 检索评测 / 答案质量评测
.\.venv\Scripts\python.exe eval\generate_golden_set.py
.\.venv\Scripts\python.exe eval\run_eval.py --top-k 5
.\.venv\Scripts\python.exe eval\run_judge.py --sample 10
```

## 评测基线（2026-08-23，milvus 后端，top_k=5）

| 意图 | n | Hit@5 | MRR | 说明 |
|---|---|---|---|---|
| detail 详情 | 40 | 1.000 | 1.000 | "X怎么做"类 |
| keyword 关键词 | 5 | 1.000 | 1.000 | recall 0.629（期望集大于 top5 容量）|
| browse 浏览 | 18 | 1.000 | 1.000 | 过滤精确率 100% |
| permission_allowed | 5 | 1.000 | 1.000 | staff 可见 internal |
| permission_denied | 5 | 1.000 | – | internal 泄漏 0 条 |
| **总体** | **73** | **1.000** | **0.931** | |

答案质量（DeepSeek LLM-judge，n=8）：classic faithfulness **4.875** / relevance **5.0**；agent faithfulness **4.75** / relevance **5.0**。完整报告见 `.artifacts/eval/`。

## 测试与质量检查

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m mypy rag_modules main.py
```

单测全部离线：fake embeddings / fake MilvusClient / fake LLM，不依赖网络、模型缓存或 API key。

## 目录结构

```
main.py                     # CLI 编排：双后端、角色、agent、trace
config.py                   # RAGConfig（全部环境变量化）
rag_modules/
  domain_config.py          # 领域配置层：词汇表/prompt/工具描述/话术（换领域只改这里）
  data_preparation.py       # 加载/元数据增强/父子块切分/visibility
  image_ingestion.py        # 图片 caption 化（缓存+并发+降级）
  vector_store.py           # VectorStoreBackend 抽象 + Milvus/FAISS 实现
  index_construction.py     # FAISS 后端 + 语料指纹 manifest + 离线优先模型加载
  retrieval_optimization.py # 混合检索编排（服务端/客户端两条路径）
  generation_integration.py # DeepSeek 生成链（prompt 来自领域配置）
  agentic_rag.py            # LangGraph agent 图（工具 schema 由领域配置生成）
  observability.py          # query_id JSONL 追踪 + LangSmith 状态
eval/                       # golden set 生成、检索评测、LLM judge
scripts/                    # 一次性工具（LFS 图片修复等）
logs/query_trace.jsonl      # 查询级追踪（gitignore）
vector_index/               # manifest + caption 缓存（构建产物）
```

## 已知边界

- 语料图片大多为 Git LFS 指针（上游仓库对象在服务端 404，无法找回）；当前仅 2 张真图完成视觉 caption 入库。真实图片补充到菜品目录后，建库时自动摄取（缓存按文件哈希增量，不重复计费）。
- CLI 每次启动有 ~30 秒冷启动成本（torch/transformers import + 嵌入模型装载），属进程级固定开销；服务化后（V2）模型常驻，仅首启支付一次。

## 迭代路线

### V2：服务化（已实现 ✅）

把 RAG 从 CLI 升级为**可被后端系统调用的独立微服务**，业务后端（客服网关等）通过 HTTP/SSE 调用；RAG 服务无状态、可横向扩容，用户体系/会话历史/消息存储仍归业务后端。

```
用户 → 业务后端（用户体系/会话管理）→ HTTP/SSE → RAG 服务（FastAPI 常驻）→ Milvus / DeepSeek
```

- **API 层（FastAPI）**：`POST /api/ask`（同步问答：query + role + pipeline，返回答案/命中条目/query_id）、`POST /api/ask/stream`（SSE 流式逐 token）、`POST /api/search`（纯检索不生成，支持 `images_only` 文搜图）、`GET /api/traces/{query_id}`（按查询回溯，对接客服工单）、`GET /healthz`（探活，含 Milvus 行数）。
- **鉴权**：服务间 JWT（HS256）；角色**从 token 的 role claim 解析**，请求体不可越权（有测试覆盖）。`.env` 设 `RAG_JWT_SECRET` 开启；不设则为开发模式（免鉴权 + 允许 body 指定角色，方便联调）。测试 token：`python -m api.mint_token --role staff`。
- **工程配套**：uvicorn 多 worker、Dockerfile + docker-compose（RAG 服务 + Milvus 全栈一键起）、CORS、SSE 错误事件兜底。
- **冷启动消除**：嵌入模型/agent 图在服务 lifespan 装载一次（约 25 秒），之后所有请求复用——实测问答接口无冷启动开销。
- RAG 内核零重写——`RecipeRAGSystem`/agent/权限/trace 原样复用（仅给 trace 加了 query_id 注入）。

**启动与测试**：

```powershell
# 本地启动（开发模式，免鉴权）
.\.venv\Scripts\python.exe -m uvicorn api.app:app --port 8000

# 接口冒烟
curl http://localhost:8000/healthz
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" -d '{\"query\":\"红烧肉怎么做\",\"pipeline\":\"agent\"}'
curl -N -X POST http://localhost:8000/api/ask/stream -d '{\"query\":\"推荐一道素菜\"}'
curl http://localhost:8000/api/traces/<query_id>

# 开启鉴权：.env 配 RAG_JWT_SECRET 后
.\.venv\Scripts\python.exe -m api.mint_token --role staff   # 生成 token
curl -H "Authorization: Bearer <token>" -X POST http://localhost:8000/api/ask -d '{\"query\":\"...\"}'

# Docker 全栈（Milvus + RAG 服务）
docker compose up -d --build
```

API 离线测试（TestClient + fake RAG，覆盖鉴权/角色透传/agent 字段/SSE/trace 回环）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -v
```

- 同期可选：bge-reranker CPU 重排（评测基线验证增益后再上）、CLIP 双塔真跨模态（图搜图）。

### V3：多知识库 + 换域迁移

- **多知识库**：为每个库建自然语言描述，agent 选库后检索（验证"库描述 + agent 路由"的通用范式）。
- **换域迁移**（菜谱 → 英语学习知识库等）：领域配置层（`domain_config.py`）已就绪，迁移 = 新写一份 `DomainConfig`（词汇表/prompt/工具描述/话术）+ 新文档解析器（PDF/Word）+ 句子窗口切块（长解释文本更适用）+ 中英双语嵌入模型（bge-m3）+ 重标 golden set；检索引擎/agent 图/权限/追踪/服务层零改动。`tests/test_domain_config.py` 已用英语领域实测整套替换。

### V4：生产加固

- 压测与容量规划、熔断限流（Milvus/LLM 双依赖保护）、token 成本监控与预算告警。
- 灰度回归：新策略（换模型/改 prompt）上线前强制跑 golden set，指标不回退才发布。
