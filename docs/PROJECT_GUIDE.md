# 菜谱 RAG 项目学习指南（面试复习用）

> 本文档按"架构 → 模块详解 → 踩坑实录 → 知识点清单 → 面试问答 -> 简历话术"组织，
> 所有数字均来自项目实测（2026-08-23，73 条 golden set）。

---

## 一、项目一页纸

**一句话**：面向客服场景的中文知识库问答系统——Milvus 混合检索 + 多模态 + 权限过滤 + LangGraph Agentic RAG + 全链路追踪 + 可量化评测 + FastAPI 服务化，纯 CPU 运行。

**技术栈**：Python 3.12 / Milvus 2.5 / LangChain + LangGraph / bge-small-zh-v1.5 + bge-reranker-base / DeepSeek（对话 + 视觉）/ FAISS（降级）/ FastAPI + JWT + SSE / pytest。

**量化结果**：

| 指标 | 数值 |
|---|---|
| 检索 Hit@5 / MRR（73 条 golden set） | 1.000 / 0.931 |
| 权限隔离 internal 泄漏 | 0 条 |
| 答案质量 LLM-judge（classic） | faithfulness 4.875/5，relevance 5.0/5 |
| 重排实验（数据驱动决策依据） | RRF 基线 0.931 > CE+RRF 融合 0.902 > 纯 CE 0.879 |
| 离线单测 | 82 个（fake Milvus/LLM/embeddings，无网络依赖） |

**语料**：322 篇菜谱 markdown → 1758 文本块 + 2 图片块（上游 LFS 对象丢失，图片管线已验证可扩展）。

---

## 二、整体架构

```
┌─────────────────────────── 离线建库（可无 API key）──────────────────────────┐
│ markdown 语料                                                                │
│  → DataPreparation: 元数据增强(分类/难度/visibility) + 父子块切分(确定性ID)    │
│  → ImageIngestion: 真图→视觉模型caption→图片块(缓存防重复计费)                 │
│  → MilvusVectorStore: 建collection(HNSW+BM25+标量索引+分区) 指纹校验复用        │
└──────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── 在线问答 ────────────────────────────────────────┐
│ 用户 query + role                                                            │
│  ├─ classic 管线: 路由→改写→混合检索(权限expr下推)→父文档回溯→生成(带引用)       │
│  └─ agent 管线(LangGraph): route(LLM选工具)→act→reflect(空结果改写重试一次)     │
│                            →generate(生成/拒答)，四级错误兜底                 │
│  全程: query_id 贯穿 → logs/query_trace.jsonl (+ LangSmith 可选)             │
└──────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── 服务层（V2）─────────────────────────────────────┐
│ FastAPI: /api/ask /api/ask/stream(SSE) /api/search /api/traces /healthz      │
│ JWT 服务间鉴权(role取自token) + session多轮记忆 + Docker compose 全栈          │
└──────────────────────────────────────────────────────────────────────────┘
```

**分层思想（面试重点）**：
1. **领域与引擎分离**：`domain_config.py` 装所有领域耦合内容（词汇表/prompt/工具描述/话术），
   检索引擎不关心"菜谱"还是"英语知识点"——换领域只写一份新 DomainConfig + 新解析器
   （`tests/test_domain_config.py` 用英语领域实测过整套替换）。
2. **后端抽象**：`VectorStoreBackend` 接口，Milvus 默认 / FAISS 离线降级，检索层不感知差异
   （服务端 expr 下推 vs 客户端后过滤两条路径）。
3. **薄服务层**：FastAPI 只做 HTTP/鉴权/会话，RAG 逻辑全部在 `rag_modules`，CLI 与服务共用内核。

---

## 三、模块详解（每块：针对的问题 → 方案 → 流程 → 知识点）

### 3.1 数据准备（data_preparation.py）

**针对的问题**：语料重建后 ID 漂移导致缓存/索引失效；元数据缺失无法过滤。

**方案与流程**：
1. 递归加载 md → 按**相对路径**生成确定性 `parent_id = sha256("parent"+相对路径)`，
   按内容哈希生成 `revision_id`，按 (parent_id, revision_id, 块位置, 块内容) 生成 `chunk_id`
   ——换目录重建，ID 不变。
2. 元数据增强：目录名→分类（荤菜/素菜/汤品…）；正文 `★` 数量→难度（1-5 档）；
   内部分类（半成品）→`visibility=internal`。
3. **父子块切分**：`MarkdownHeaderTextSplitter` 按 #/##/### 切子块（子块继承父元数据），
   检索用小块（语义集中、向量精确），生成用**整篇父文档**（上下文完整）。
   `parent_child_map` 支持子块命中→父文档回溯（按命中块数排序去重）。

**知识点**：chunking 策略对比（固定长度/递归/父子块/句子窗口——本项目语料按标题天然分段，
父子块最优；句子窗口适合长解释型文本，换英语语料时再换）；确定性 ID 的价值（增量更新、缓存键）。

### 3.2 向量存储（vector_store.py + index_construction.py）

**针对的问题**：FAISS 无标量过滤/分区/运维能力，不适合线上。

**方案**：Milvus collection schema：`chunk_id(PK) + vector(FLOAT_VECTOR,HNSW,COSINE)
+ sparse(BM25,jieba 分词) + 13 个标量字段`，`category` 为分区键（过滤时直接剪枝），
difficulty/visibility/modality 建倒排索引。

**关键流程**：
- 建库：语料指纹（所有 chunk_id+内容哈希排序后整体 sha256）写 manifest → 指纹不变则复用 collection。
- **混合检索**：两个 AnnSearchRequest（dense 向量 + sparse 原文查询串）→ 服务端 `RRFRanker(60)` 融合
  → 表达式过滤（**必须挂在每个 AnnSearchRequest.expr 上**，见踩坑 6.1）。
- `build_filter_expr`：白名单字段 + 引号转义，防表达式注入。

**知识点**：HNSW 原理（分层跳表式近似最近邻，M/efConstruction 参数）；BM25（词频饱和 + 文档长度归一，
TF-IDF 的改进）；jieba 中文分词；RRF 公式 `score = Σ 1/(k+rank_i)`（k=60，无需调权重的多路融合）；
**pre-filter vs post-filter**（Milvus 服务端先过滤再检索 vs FAISS 客户端检索后过滤——召回完整性差异）；
分区键 vs 标量索引的剪枝差异。

### 3.3 混合检索编排（retrieval_optimization.py）

**流程**：`hybrid_search(query, top_k, expr)`：
1. 服务端路径：Milvus hybrid_search 拿 pool（默认 top_k，重排开启时扩到 20）。
2. 客户端路径（FAISS）：dense 召回 + 本地 BM25（**中文感知分词**：单字+二元组+短词，无词典依赖）
   → 客户端 RRF 融合 → expr 后过滤（`document_matches_expr` 解析自建表达式子集）。
3. 可选 cross-encoder 重排（默认关）。

**知识点**：为什么需要混合检索（向量怕专有名词/精确匹配，词法怕同义改写，两者互补）；
中文 BM25 分词策略；rank fusion 系列方法（RRF vs 加权归一 vs CombSUM）。

### 3.4 多模态（image_ingestion.py）

**针对的问题**：无 GPU；语料图片信息完全丢失。

**方案**：**caption 化**——视觉模型把图转中文描述，与文本统一进同一向量空间（而非 CLIP 双塔）。
CPU 零负担、文字↔图片检索都能走，代价是真"图搜图"做不到（V3 可加 CLIP）。

**流程**：扫描菜品目录真图（**自动跳过 <2KB 的 Git LFS 指针文件**）→ DeepSeek-Vision 生成
≤80字 caption（并发 4）→ 构造 image chunk（`modality=image, image_path, parent_id` 挂回菜谱，
父文档排序自动受益）→ **缓存键 = 路径+文件哈希+模型+prompt版本**，重建零重复计费。
失败单图跳过（strict 模式可改为抛错），全失败降级纯文本建库。

**知识点**：多模态检索三条路线对比（caption 化 / CLIP 双塔 / 多模态 LLM 端到端 embedding）；
缓存设计（内容寻址）；**推理型视觉模型陷阱**（reasoning 消耗 completion tokens，max_tokens 要给足）。

### 3.5 权限（visibility + role）

**模型**：guest/user/staff × public/internal；`role→允许的visibility列表→Milvus expr 下推`，
**在数据库层强制执行**，绕不过；user 查内部内容返回相关公开菜谱（优雅降级而非报错/空结果）。

**知识点**：元数据级 ACL 是 RAG 权限的主流做法（比文档级/行级轻，比 prompt 约束硬）；
"权限泄漏"的评测定义（返回 internal 才算泄漏，公开结果是合理降级）。

### 3.6 Agentic RAG（agentic_rag.py）

**针对的问题**：固定管线"无脑检索"，不会按意图选检索方式、不会自救。

**图结构**：
```
START → route → act → reflect → generate → END
                ^       │ (空结果且未改写过: 改写→retry_pending=True)
                └───────┘
```
- **route**：DeepSeek function calling 在 4 个工具里选并抽参数（`search_recipes(query,category,difficulty)
  / search_images / list_by_metadata / direct_answer`），工具 schema 由 DomainConfig 生成（枚举锁参数）。
  **多轮记忆**：最近 3 轮历史注入 route 消息列表（指代消解）和生成问题前缀。
- **act**：执行工具；direct_answer 短路返回。
- **reflect**：空结果 → LLM 改写 query（结合对话背景）重试一次；仍空 → generate 拒答。
- **generate**：父文档 + 图片路径 → 生成（带引用）；LLM 失败 → 规则摘要兜底。

**四级兜底**（错误分级，全部落 trace）：路由失败→默认检索；工具失败→退回混合检索；
检索不可用→无 LLM 规则摘要；生成失败→返回检索条目列表。

**知识点**：Agent = LLM + 工具 + 循环 + 状态；function calling 机制（schema 注入 → 模型输出结构化调用）；
**工具选择的四层约束**（描述/schema 枚举/代码校验兜底/图结构限轨迹）；LangGraph StateGraph、
conditional edges、**状态合并语义**（节点返回缺省 key 保留旧值——必须显式覆盖才能结束循环）；
反思（reflection）/重写（rewrite）模式 vs Self-RAG vs Corrective RAG 的关系。

### 3.7 生成与引用溯源（generation_integration.py）

**双层引用**：① prompt 指示 LLM 内联标注【食谱 N】（上下文本身带【食谱 N】头）；
② **确定性附录**：由实际检索到的父文档生成"参考来源：菜名（路径）"，不依赖 LLM 配合，
生成降级时依然可验证。列表类回答走规则模板（不调 LLM）。

**知识点**：grounding/faithfulness；结构化引用是客服审计刚需；"LLM 标注 + 程序化兜底"的分层思路。

### 3.8 追踪（observability.py）

`QueryTrace`：query_id(uuid) + 事件流(route/act/rewrite/fallback/no_results/…) + 耗时 + 答案 + 错误
→ JSONL 追加写（线程安全锁）→ `load_traces` 回放。LangSmith 走环境变量零代码接入。

**知识点**：可观测三支柱（日志/指标/追踪）中 RAG 场景的"查询级 trace"设计；
badcase 回放思路（query_id → events 复现决策链）。

### 3.9 评测体系（eval/）

**Golden set**（73 条，5 类意图）：detail 40（"X怎么做"）/ keyword 5 / browse 18（"推荐几道{难度}的{分类}"，
测过滤精确率）/ permission_denied 5（user 查 internal，断言零泄漏）/ permission_allowed 5。

**检索指标**：Hit@k、MRR `= 1/首个相关排名`、recall、过滤精确率；**LLM judge**：DeepSeek 按
faithfulness（答案是否忠于检索内容）+ relevance（是否切题）打 1-5 分，对比 classic vs agent 双管线。

**评测驱动迭代的实例**（简历/面试最佳素材）：重排三组对照实验 → 数据说服力 → 默认关闭的决策。

**知识点**：RAG 评测分层（检索指标/生成指标/端到端）；golden set 设计（意图覆盖 + 权限正反向）；
LLM-as-judge 的偏差与缓解（明确量表、结构化输出、双管线对照）。

### 3.10 服务层（api/）

FastAPI lifespan 一次性装载模型/Milvus/agent（**消除 CLI 每次启动 30s 冷启动**）；
`/api/ask`（含 session_id 多轮 + citations 字段）、`/api/ask/stream`（SSE：token 事件流 + done 带 query_id，
中途异常以 error 事件收尾）、`/api/search`（纯检索 + images_only 文搜图）、`/api/traces/{id}`、`/healthz`。

**鉴权**：JWT HS256；**role 只从 token claim 取**（请求体不可越权，有测试锁死）；
未配 `RAG_JWT_SECRET` 自动降级开发模式。`api/mint_token` 造测试 token。
**会话**：内存存储（线程安全 + FIFO 驱逐 + 最近 3 轮），多副本部署应换 Redis（已在代码注释说明）。

**知识点**：SSE vs WebSocket（单向流 vs 双向）；JWT 结构（header.payload.signature）与 HS256；
lifespan 钩子；CORS；线程安全共享态；**charset=utf-8 显式声明的必要性**（PowerShell/.NET 默认 Latin-1）。

### 3.11 工程化（贯穿）

- 确定性 ID + **语料指纹 manifest**：索引与语料强一致（不匹配拒载重建）。
- JSON 替代 pickle（防反序列化注入）；原子写（tmp+replace）。
- **离线优先模型加载**：先翻 `huggingface_hub.constants.HF_HUB_OFFLINE` 再 import（时序！），
  断网/代理挂死不阻塞；下载回退 hf-mirror。
- 82 个离线单测（fake Milvus/LLM/embeddings）+ ruff + mypy 全绿。
- Docker compose 全栈一键起。

---

## 四、踩坑实录（每条都是面试细节弹药）

| # | 坑 | 根因 | 解法 |
|---|---|---|---|
| 1 | hybrid_search 过滤失效（权限形同虚设且无报错） | pymilvus 2.5 的 `filter` kwarg 被**静默忽略** | 对照实验定位；expr 挂在每个 AnnSearchRequest 上 |
| 2 | 断网/代理挂死启动卡几分钟 | `HF_HUB_OFFLINE` 环境变量在 huggingface_hub **import 后**设置无效（常量已固化） | import 前翻转 `constants.HF_HUB_OFFLINE`；离线失败再在线 |
| 3 | 328 张"图片"全失败 | 是 **Git LFS 指针**（131B 文本），上游 LFS 对象 404 拉不回 | <2KB+指针魔数检测自动跳过；真图管线验证通过 |
| 4 | 视觉 caption 返回空 | deepseek-vision 是**推理模型**，300 token 全被 reasoning 吃掉 | max_tokens=2000 + "直接输出描述" |
| 5 | agent 改写重试死循环 | LangGraph 节点返回**缺省 key 保留旧值**，pop 无效 | retry_pending 显式写 False |
| 6 | BM25 单测"空结果"总有返回 | BM25Retriever 无匹配也返回 top-k | 测试夹具改走服务端语义路径 |
| 7 | 重排上线反而掉点 | 精确菜名查询下 BM25 信号已最强，CE 把精确匹配挤后 | 三组对照评测 → 默认关闭，保留开关 |
| 8 | PowerShell 中文乱码 | 响应缺 charset，.NET 默认 Latin-1 解码 | 自定义 JSONResponse 显式 `charset=utf-8` |
| 9 | 类属性在方法内裸引用 NameError | Python 类作用域不进入方法作用域 | 模块级常量 / self. 前缀 |

---

## 五、知识点总清单（对照复习）

**检索**：向量检索/ANN/HNSW、BM25、RRF、混合检索、pre/post-filter、父子块/句子窗口、rerank（cross-encoder）、语义缓存（V2 候选）
**向量库**：Milvus schema/分区键/标量倒排/expr/hybrid_search、FAISS 对比、索引选型
**Agent**：function calling、ReAct、LangGraph 状态机、反思/改写、工具描述工程、轨迹约束
**生成**：grounding、引用溯源、拒答校准、流式（SSE）、幻觉缓解
**RAG 工程**：元数据 ACL、确定性 ID、指纹校验、增量更新、多模态三路线、领域解耦
**评测**：golden set 设计、Hit@k/MRR/recall、LLM-as-judge、A/B 对照、评测驱动决策
**服务化**：FastAPI、JWT、SSE、会话状态、lifespan、Docker compose、CORS、编码
**可观测**：结构化日志、查询级 trace、LangSmith、（V2 候选：Prometheus 指标）

---

## 六、高频面试问答（背熟）

**Q: 为什么 Milvus 不用 FAISS？**
A: 三点——标量过滤与表达式下推（权限/分类过滤必须在检索层做）、分区与索引运维能力、生产生态（备份/监控/扩容）。FAISS 保留为离线降级路径，接口层抽象屏蔽差异。

**Q: 父子块和句子窗口怎么选？**
A: 由语料结构决定。菜谱 md 按标题天然分段、每段语义自洽，父子块（小块检索/整篇生成）最优；长解释型文本（如英语语法讲解）上下文依赖强，句子窗口（检索句+前后窗口拼接）更合适——所以切块策略放在领域层，换语料时换。

**Q: Agent 比固定管线好在哪？代价是什么？**
A: 好处——意图分发从规则变成语义理解（问天气不浪费检索、看图走图检索、浏览走结构化过滤）、能自救（改写重查）、多轮指代。代价——每问多 1-2 次 LLM 调用（延迟/成本）、行为不完全可预测（靠枚举+校验+图结构约束）。我的 judge 评测里 classic faithfulness 4.875 vs agent 4.75，agent 略低主要来自拒答场景，可用性（不硬答）反而更好。

**Q: 重排为什么默认关？**
A: 不是不会用，是评测说话：73 条 golden set 上 RRF 基线 MRR 0.931，纯 CE 0.879，CE+RRF 融合 0.902——本语料查询以精确菜名为主，BM25 信号已最强，CE 反而把精确匹配挤后。保留开关和评测脚本，换语料（精确匹配信号弱）一键复测。

**Q: 权限怎么做的？能被绕过吗？**
A: visibility 写进向量库标量字段，role→可见列表→Milvus expr 下推，在数据库层强制执行；表达式由白名单字段构建+转义（防注入）；服务化后 role 只从 JWT claim 取，请求体不可越权（有测试）。评测里 internal 泄漏 0 条。

**Q: 项目离生产还差什么？**
A: 按优先级：增量更新管道（现在是全量重建）、Prometheus 指标+成本核算、CI/CD、限流熔断、语义缓存、多格式解析（PDF/Word）。HA/压测在 demo 阶段刻意不做（说得出"为什么不做"也是工程判断力）。

---

## 七、简历项目描述（可直接粘贴）

> **基于 LangGraph 的 Agentic RAG 知识库问答系统**（个人项目，Python/Milvus/LangGraph/FastAPI）
>
> - 设计并实现面向客服场景的 RAG 系统：Milvus 混合检索（HNSW 稠密 + BM25 中文稀疏，RRF 服务端融合）+ 父子块映射，自建 73 条 golden set 评测，Hit@5 达 100%、MRR 0.931；
> - 实现 Agentic RAG：LangGraph 状态图（路由→执行→反思→生成），LLM function-calling 在 4 个检索工具间自主路由并抽取结构化过滤参数，空结果自动改写重查，四级错误兜底（路由/工具/检索/生成逐层降级，全程落 trace）；
> - 元数据级权限：visibility 标量字段 + 角色表达式下推至向量库强制过滤，配合 JWT（角色取自 token，请求体不可越权），评测 internal 泄漏 0 条；
> - 多模态与可观测：无 GPU 条件下以视觉模型 caption 化实现文搜图（内容寻址缓存防重复计费）；query_id 贯穿的 JSONL 全链路追踪，任意 badcase 可回溯；
> - 评测驱动迭代：cross-encoder 重排三组对照实验证明本语料无增益，据此默认关闭（0.931 > 融合 0.902 > 纯 CE 0.879）；LLM-as-judge 评 faithfulness 4.875/5；
> - 工程化：FastAPI 服务化（SSE 流式/JWT 鉴权/多轮会话记忆/Docker 全栈）、82 个离线单测 + ruff/mypy 全绿、确定性 ID + 语料指纹校验、领域配置层支持零成本切换知识库领域。

（投递时建议再附 GitHub 链接；数字随迭代更新。）
