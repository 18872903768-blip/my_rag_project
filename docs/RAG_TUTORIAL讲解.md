# 从零搭建一套生产级 RAG：菜谱客服系统完整教程

> **写给谁**：学过 RAG 概念（向量检索、chunking、prompt 这些词都见过），但没有真正从零写过代码、串不起完整链路的你。
> **怎么用**：每一章都是"为什么 → 怎么设计 → 代码走读 → 知识点 → 动手验证"五段式。建议对照仓库源码边读边跑。
> **总-分结构**：第 0-1 章是"总"（全景与演进路线），第 2-12 章是"分"（每个模块拆子模块精讲）。

---

# 第 0 章 总：先想清楚我们在解决什么问题

## 0.1 为什么需要 RAG

LLM 有三个先天缺陷，RAG 是针对前两个的主流解法：

| 缺陷 | 表现 | 解法 |
|---|---|---|
| 知识截止 | 不知道私有数据（你的菜谱库、公司文档） | **RAG：检索增强生成** |
| 幻觉 | 一本正经编造 | RAG（让模型"开卷考试"）+ 引用溯源 |
| 上下文有限 | 塞不下整个知识库 | RAG（只检索相关的几段） |

> **知识点：RAG vs 微调 vs 长上下文**
> - 微调：改模型权重，贵、慢、更新难，适合改"风格/格式"而非"知识"；
> - 长上下文：直接塞全文，token 成本 O(N²) 注意力、召回不稳定，小库可以、大库不可行；
> - RAG：知识放外部库，推理时检索 top-k 注入 prompt——知识可随时更新、可溯源、可控权限。三者不互斥。

## 0.2 一个 query 的完整生命周期（本项目实际链路）

用户问"红烧肉怎么做"，系统内部发生的事：

```
1. query 进入 →（agent 管线）LLM 看到4个工具说明书，决定调 search_recipes("红烧肉 做法")
2. 权限检查 → role=user ⇒ 生成 Milvus 过滤表达式 visibility in ["public"]
3. 混合检索 → dense(bge向量化+HNSW搜索) + sparse(BM25全文) 两路召回各20条 → RRF融合
4. 父文档回溯 → 命中的子块 → 找到所属菜谱原文（整篇）
5. 生成 → 菜谱原文 + 问题 → DeepSeek → 回答 + 【食谱1】内联标注 + 📚来源附录
6. 追踪 → 全程 query_id 串起来写 JSONL，事后可回溯每一步
```

每一步对应一个模块，就是本教程第 2-9 章的目录。

## 0.3 本项目是怎么"一步步"搭起来的（演进史 = 学习路线）

**不要试图一次搭出完整系统。** 本项目分四步走，每一步都有一个"可运行的最小闭环"：

| 阶段 | 目标 | 关键决策 | 得到什么 |
|---|---|---|---|
| V0 | 跑通最小链路 | FAISS + BM25 双路 + 父子块；离线建库不需要 API key | 能问答的 demo |
| V1 | 补生产短板 | FAISS→Milvus；加权限/多模态/agent/追踪/评测 | 可量化、可审计 |
| V1.5 | 可迁移性 | 领域配置层抽取（domain_config） | 换知识库只改配置 |
| V2 | 可上线 | FastAPI 服务化 + JWT + SSE + Docker | 微服务节点 |
| V2.5 | 质量与体验 | 重排实验（数据驱动取舍）、多轮记忆、引用溯源 | 客服级体验 |

> **工程思维**：每一步的验收标准不是"代码写完了"，而是"评测数字/端到端演示证明它工作"。

---

# 第 1 章 总：架构地图与模块清单

## 1.1 三层架构

```
┌─────────────────────── 服务层 api/（第11章）───────────────────────┐
│ FastAPI: ask / ask-stream(SSE) / search / traces / healthz        │
│ JWT 鉴权 · session 多轮记忆 · Docker compose                       │
├─────────────────────── 引擎层 rag_modules/（第2-9章）─────────────┤
│ ① data_preparation   数据准备：加载/元数据/父子块切分               │
│ ② image_ingestion    多模态：图片 caption 化入库                   │
│ ③ vector_store       向量存储：Milvus schema/混合检索/表达式        │
│ ④ retrieval_optimization  检索编排：双路召回/RRF/过滤/可选重排       │
│ ⑤ agentic_rag        Agent：LangGraph 路由/反思/兜底               │
│ ⑥ generation_integration  生成：LCEL 链/prompt/引用                │
│ ⑦ observability      追踪：query_id 事件流 JSONL                   │
│ ⑧ domain_config      领域配置：所有"菜谱知识"集中于此               │
│ ⑨ reranker           重排：cross-encoder（实验性，默认关）          │
│ ⑩ index_construction FAISS 降级后端 + 离线模型加载                 │
├─────────────────────── 评测层 eval/（第10章）─────────────────────┤
│ golden set 生成 · 检索指标 · LLM judge                              │
└──────────────────────────────────────────────────────────────────┘
```

**分层铁律（为什么这么分）**：
1. **领域与引擎分离**——引擎层不允许出现"菜谱"两个字（全部从 domain_config 读），
   换成英语学习库时引擎零改动。这是最重要的架构决策。
2. **依赖倒置**——检索编排只依赖 `VectorStoreBackend` 抽象接口，不依赖具体 Milvus/FAISS。
3. **薄服务层**——API 层只有 HTTP/鉴权/会话编排，没有任何 RAG 逻辑，CLI 和服务共用引擎。

## 1.2 代码目录对照

```
main.py                 # 引擎编排入口（CLI）：建库/检索/问答/agent
rag_modules/            # 引擎层（上面①-⑩）
api/                    # 服务层：app.py 路由 / auth.py JWT / schemas.py 模型
eval/                   # 评测：generate_golden_set / run_eval / run_judge
tests/                  # 82个离线单测（fake 注入，无网络）
docs/                   # 本教程 + PROJECT_GUIDE（面试复习版）
```

---

# 第 2 章 分：数据准备模块（data_preparation.py）

> RAG 的天花板在数据准备，不在模型。"Garbage in, garbage out。"

## 子模块 2.1 文档加载：先把文件变成 Document 对象

**目标**：把磁盘上 322 个 md 变成带元数据的 `Document(page_content, metadata)` 列表。

**设计决策**：
1. **稳定顺序**：按相对路径排序遍历——同样的语料永远产出同样的序列（后面算"语料指纹"依赖这一点）。
2. **模板排除**：`template` 目录是菜谱写作模板，不是知识，路径含 template 即跳过。
3. **失败策略可配置**：`strict=True` 时任何一个文件读取失败就整体报错（宁可不做也不能做半套索引），
   `strict=False` 时跳过并记日志。**工程上"部分成功"往往比"失败"更危险**——用户以为索引是全的。

```python
discovered = sorted(
    data_root.rglob("*.md"),
    key=lambda path: path.relative_to(data_root).as_posix().casefold(),
)
```

> **知识点：LangChain Document**
> RAG 世界的"通用货币"：`page_content`（正文）+ `metadata`（字典，随块继承）。
> 上游切块、下游检索/生成全都围绕它，换框架不用换数据结构。

## 子模块 2.2 元数据增强：给每个文档打标签

**为什么重要**：元数据是 RAG 的"第二检索维度"——向量管语义相似，元数据管精确约束
（"只要素菜""只看简单的"），权限也靠它（visibility）。

```python
def _enhance_metadata(self, document: Document) -> None:
    source_path = Path(document.metadata.get("source_path", ""))
    path_parts = {part.casefold() for part in source_path.parts}

    category = "其他"
    for key, value in get_domain().category_mapping.items():
        if key.casefold() in path_parts:          # 目录名 → 分类（meat_dish→荤菜）
            category = value
            break

    document.metadata["category"] = category
    document.metadata["dish_name"] = source_path.stem   # 文件名 → 菜名
    document.metadata["visibility"] = (
        "internal" if category in self.internal_categories else "public"  # 权限
    )

    star_match = re.search(r"★+", document.page_content)  # 正文★数量 → 难度
    ...
```

三个信号来源：**目录结构**（最可靠）、**文件名**、**正文正则**。能从结构拿的就别用模型——
零成本、零幻觉。只有结构里没有的信息（如以后的"菜系"）才值得用 LLM 标注。

## 子模块 2.3 确定性 ID：整个工程的地基

**问题**：索引缓存、增量更新、父子映射都靠"同一个文档在每次构建时 ID 相同"。随机 UUID 做不到。

```python
@staticmethod
def _stable_id(namespace: str, *parts: str) -> str:
    payload = "\0".join((namespace, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

parent_id   = _stable_id("parent", source_path)              # 相对路径 → 父ID
revision_id = _stable_id("revision", source_path, content_hash)  # 内容变了 → 修订ID变
child_id    = _stable_id("chunk", parent_id, revision_id, str(i), content)
```

设计要点：
- **namespace 防碰撞**：parent/chunk/revision 前缀隔开，同内容不同用途 ID 不同；
- 用**相对路径**而非绝对路径：把语料挪个盘，ID 不变，缓存依然命中；
- `revision_id` 绑定内容哈希：文件改了一个字 → 修订 ID 变 → 后续可做增量更新（V3）。

## 子模块 2.4 父子块切分：本项目最重要的检索设计

**问题（先于方案）**：切块的"两难"——
- 块切**小**：向量语义集中，检索准；但喂给 LLM 的上下文碎片化，回答缺前因后果；
- 块切**大**：上下文完整；但一篇菜谱混着原料/步骤/技巧，向量"什么都有=什么都不像"，检索烂。

**父子块（parent-child chunking）**：把两难拆开——
```
检索用子块（小而准）  →  命中后  →  回溯整篇父文档（大而全）喂给 LLM
```

```python
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "主标题"), ("##", "二级标题"), ("###", "三级标题")],
    strip_headers=False,   # 保留标题：标题本身是强检索信号（"## 必备原料和工具"）
)
for chunk_index, split_chunk in enumerate(split_chunks):
    metadata = {
        **parent.metadata,          # 子块继承父块全部元数据（分类/难度/visibility…）
        "chunk_id": child_id,
        "doc_type": "child",
        "chunk_index": chunk_index,
    }
```

子块继承父元数据是关键细节：过滤"素菜"时命中的子块自己就带 category，不需要回查父文档。

**父文档回溯**（检索之后调用）：
```python
def get_parent_documents(self, child_chunks):
    # 按父文档被命中的子块数排序：一篇菜谱被5个子块命中 > 被1个命中
    relevance[parent_id] = relevance.get(parent_id, 0) + 1
    ranked = sorted(relevance, key=lambda pid: (-relevance[pid], first_seen[pid]))
```

> **知识点：切块策略全景（面试高频）**
> | 策略 | 原理 | 适用 |
> |---|---|---|
> | 固定长度/重叠 | 每500字切一段，重叠50 | 无结构文本，兜底 |
> | 递归分割 | 按段落→句子→字符逐级尝试 | 通用 |
> | **父子块** | 小块检索、大块生成 | 结构化文档（本项目） |
> | 句子窗口 | 检索单句，返回时拼接前后窗口 | 长解释型文本（英语语态讲解） |
> | 语义切块 | 相邻句向量相似度骤降处切 | 无结构但语义分段明显 |
> 选择依据是**语料结构**，不是流行度。

## 子模块 2.5 语料指纹：让索引"可校验"

所有 chunk 的 `(chunk_id, content_hash, parent_id)` 排序后整体 sha256 → 一根指纹。
存在 manifest 里，建库前对比：**指纹一致才允许复用索引**，否则强制重建。
（V0 时代 FAISS 版就有，V1 迁 Milvus 时平移——防止"新语料配旧索引"的静默错误。）

**动手验证**：
```powershell
.\.venv\Scripts\python.exe main.py --build-only          # 第一次：重建
.\.venv\Scripts\python.exe main.py --build-only          # 第二次：日志出现"复用"
# 随便改一个 md 的内容再跑 → 自动触发重建
```

---

# 第 3 章 分：向量存储模块（vector_store.py）

## 子模块 3.1 决策：为什么 FAISS 换 Milvus

| 能力 | FAISS（库） | Milvus（数据库） |
|---|---|---|
| 向量检索 | ✅ 快 | ✅ |
| 标量过滤 | ❌ 只能取回后自己筛 | ✅ **表达式下推，先过滤再检索** |
| 分区/索引管理 | ❌ | ✅ 分区键剪枝、多类型索引 |
| 持久化/运维 | 自己写文件 | ✅ 服务化、备份、监控 |

客服场景的权限过滤必须在数据库层做（性能+安全），所以换。但 FAISS 保留为**降级路径**
（`VectorStoreBackend` 抽象统一接口）—— Milvus 挂了还能跑，这是"依赖倒置"的实际收益。

## 子模块 3.2 Milvus 五个基础概念（5 分钟入门）

1. **Collection** ≈ 数据库表；2. **Field** ≈ 列，类型可以是向量/标量；
3. **索引**：向量字段建 ANN 索引（本项目 HNSW），标量字段建倒排；
4. **分区（partition key）**：按某字段值把数据分桶，带该字段过滤时**整桶剪枝**不进入计算；
5. **表达式（expr）**：类 SQL 布尔表达式 `category == "荤菜" && visibility in ["public"]`，检索时下推。

## 子模块 3.3 Schema 设计逐字段讲解

```python
schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)  # 主键=确定性ID
schema.add_field("vector",   DataType.FLOAT_VECTOR, dim=512)   # bge-small-zh 输出512维
schema.add_field("sparse",   DataType.SPARSE_FLOAT_VECTOR)     # BM25稀疏向量（见3.5）
schema.add_field("content",  DataType.VARCHAR, max_length=16384,
                enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})  # 原文+分词器
schema.add_field("category", DataType.VARCHAR, max_length=32, is_partition_key=True)  # 分区键！
for field in ("difficulty", "visibility", "modality"):
    index_params.add_index(field_name=field, index_type="INVERTED")  # 常过滤字段建倒排
# 其余标量字段：dish_name/parent_id/source_path/image_path/... 原样存，检索时可原样返回
schema.add_function(Function(name="content_bm25", function_type=FunctionType.BM25,
    input_field_names=["content"], output_field_names=["sparse"]))  # 服务端BM25函数
```

设计理由：
- **向量维度 512**：bge-small-zh-v1.5 的输出维度，小模型=CPU 友好，中文效果够用；
- **content 字段带 jieba 分词器**：BM25 在服务端算，不用自己维护倒排；
- **category 做分区键**：11 个分类=11 个桶，"推荐素菜"类查询直接跳过其他 10 桶；
- **标量全量存储**：检索结果直接携带元数据返回，不用二次查表。

## 子模块 3.4 Dense 检索：从文本到向量

```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True},  # 归一化后 COSINE≈内积
)
vector = self.embeddings.embed_query(query)   # 查询时：query→512维向量
results = client.search(collection_name=..., data=[vector], anns_field="vector",
                        search_params={"metric_type": "COSINE"}, limit=k, filter=expr, ...)
```

> **知识点：为什么归一化 + COSINE**
> 文本相似度关心**方向**不关心长度。归一化后所有向量落在单位球面，余弦相似度=内积，
> 计算更快且数值稳定。bge 系列模型官方要求归一化。
> **ANN（HNSW）**：精确检索 O(N) 扫全库，图索引 HNSW 构建分层"高速公路"，
> 查询 O(logN) 近似_top-k，参数 M=16（每节点连接数）/ efConstruction=200（建图质量）。

## 子模块 3.5 Sparse 检索：BM25 在服务端

BM25 = 改进版 TF-IDF：词频饱和（出现10次不如5次的2倍相关）+ 文档长度归一（短文档命中权重更高）。
公式（了解即可）：`score(q,d) = Σ IDF(qi) · TF 饱和项 · 长度归一项`。

Milvus 2.5 的玩法：**写入时**对 content 字段跑 BM25 函数自动生成稀疏向量存进 sparse 字段；
**查询时**直接把查询原文发给 sparse 字段，服务端分词打分——整条词法检索链路零自研代码。

## 子模块 3.6 混合检索代码走读（本模块的高潮）

```python
def hybrid_search(self, query, k=5, *, expr=None, candidate_k=None):
    vector = self.embeddings.embed_query(query)          # ① 查询向量化
    dense_request = AnnSearchRequest(                     # ② dense 路
        data=[vector], anns_field="vector",
        param={"metric_type": "COSINE"}, limit=limit,
        expr=expr,   # ⚠ 表达式必须挂这里！filter kwarg 会被静默忽略（踩坑见附录）
    )
    sparse_request = AnnSearchRequest(                    # ③ sparse 路（发原文）
        data=[query], anns_field="sparse",
        param={"metric_type": "BM25"}, limit=limit, expr=expr,
    )
    results = client.hybrid_search(                       # ④ 服务端 RRF 融合
        collection_name=..., reqs=[dense_request, sparse_request],
        ranker=RRFRanker(60), limit=k, output_fields=...)
```

两路各自返回 top-limit，服务端按 RRF（第 4 章详解）融合成一张榜单。**为什么在服务端融合**：
省一次网络往返、Milvus 内部融合可利用未截断的完整排名信息。

## 子模块 3.7 表达式构建器：安全的第一课

用户输入绝不能直接拼进查询表达式。`build_filter_expr` 三层防线：
1. **字段白名单**：`FILTERABLE_FIELDS = {"category","difficulty","visibility",...}`，不在名单直接抛错；
2. **值转义**：`value.replace("\\","\\\\").replace('"','\\"')` 再包引号；
3. **结构固定**：只生成 `k == v` / `k in [v1,v2]` 两种子句，`&&` 连接。

```python
build_filter_expr({"category": "荤菜"})            # → 'category == "荤菜"'
build_filter_expr({"visibility": ["public"]})      # → 'visibility in ["public"]'
build_filter_expr({"payload; drop": "x"})          # → ValueError（白名单拦截）
```

> **知识点：注入面**
> 向量库的表达式、数据库的 SQL、LLM 的 prompt 是三类注入面。前两类用"白名单+参数化"，
> prompt 注入（检索内容里藏指令）是开放问题，缓解手段是输入输出过滤+权限在检索层强制（越权即使注入也拿不到数据）。

---

# 第 4 章 分：检索编排模块（retrieval_optimization.py）

## 子模块 4.1 为什么必须混合检索（举例直觉）

- 用户问"宫保鸡丁"：**BM25 完胜**——精确词匹配，向量会把"辣子鸡丁""京酱鸡丝"都拉回来；
- 用户问"肉类硬菜怎么做"：**向量完胜**——没有任何文档字面含"硬菜"，语义相似才召得回。
两路互补，融合取长。这是所有生产 RAG 系统的标配。

## 子模块 4.2 RRF：不调参的多路融合（手算一遍）

**公式**：`score(d) = Σ_roads 1/(k + rank_road(d))`，k=60。

例：查询"红烧肉"，两路排名（只看两个文档）：

| 文档 | dense 排名 | bm25 排名 | RRF 得分 |
|---|---|---|---|
| 简易红烧肉 | 1 | 2 | 1/61 + 1/62 = 0.01639 + 0.01613 = **0.0325** |
| 湖南红烧肉 | 2 | 1 | 1/62 + 1/61 = **0.0325**… 等等，并列？ |

细看：1/61+1/62 与 1/62+1/61 相同——**两路排名完全互换时 RRF 无法区分**（这正是它的中性所在）；
差异来自两路**一致**看好谁。真实项目里 top-1 通常两路都排第一，得分 2/61 拉开差距。

**为什么用 RRF 而不是分数加权**：dense 输出余弦相似度（0~1），BM25 输出无界分，**量纲不可比**；
归一化加权需要调权、对分布敏感。RRF 只用排名，天然免调参——这就是它在工业界流行的原因。

## 子模块 4.3 客户端路径：FAISS 兜底 + 中文 BM25 分词器

Milvus 不可用时的降级链路。难点：BM25 默认按空格分词，**中文整篇是一个 token**，词法检索报废。
自写分词器（无词典依赖）：

```python
def tokenize_chinese(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalized)          # 英文数字直接切
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):  # 中文串：
        tokens.extend(sequence)                             # ① 单字（红烧→红,烧）
        tokens.extend(bigrams)                              # ② 二元组（红烧,烧肉）←主力
        if len(sequence) <= 8:
            tokens.append(sequence)                         # ③ 短语整体（精确菜名）
    return tokens
```

单字保召回、二元组保区分度、短语保精确匹配——三层粒度覆盖，比直接上 jieba 轻（jieba 留给 Milvus 服务端用）。

## 子模块 4.4 表达式的两条执行路径（对照记忆）

| | Milvus（服务端 pre-filter） | FAISS（客户端 post-filter） |
|---|---|---|
| 时机 | **先过滤再检索**，向量只在合格子集里搜 | 先检索 top-N 再逐条筛 |
| 召回完整性 | ✅ 不会因过滤丢结果 | ⚠️ top-N 里不合格的扔掉后可能不足 k 条 |
| 代码位置 | expr 挂 AnnSearchRequest | `document_matches_expr` 手写解析器逐条匹配 |

手写解析器只支持自建表达式子集（`==`/`in`/`&&`），因为**我们只生成这几种**——自产自销，安全闭环。

## 子模块 4.5 可选重排：一次完整的"实验-决策"示范

**动机**：RRF 是粗排（只看排名），cross-encoder（如 bge-reranker）把 query 和每个候选**拼在一起**
过编码器，逐对精细打分，通常能显著提升排序。

**接入**：召回池扩到 20 → CE 对 20 对打分 → 排序取 top-k。`RAG_RERANK_ENABLED` 开关。

**实验结果（73 条 golden set）**：

| 方案 | Hit@5 | MRR |
|---|---|---|
| RRF 基线 | 1.000 | **0.931** |
| 纯 CE 定序 | 1.000 | 0.879 |
| CE 序 + RRF 序二次融合 | 1.000 | 0.902 |

**分析**：本语料 40 条 detail 查询都是"X怎么做"——BM25 精确菜名信号已接近完美，
CE 按"语义相关性"反而把同名变体（湖南家常红烧肉）排到精确匹配（简易红烧肉）前面。

**决策**：实现保留、**默认关闭**。这不是失败，是评测体系的价值——
"我实现过、量化过、据此做过取舍"远比"我加过重排"有说服力。

> **知识点：bi-encoder vs cross-encoder**
> bi-encoder（双塔）：query 和 doc 分别编码，向量可**离线预计算**，检索快但精度有限（两者没交互）；
> cross-encoder：query+doc 拼接后联合编码，精度高但**每个候选都要现算**，只能用于小候选池精排。
> 生产标配组合：双塔召回（万级）→ cross-encoder 精排（几十条）。

---

# 第 5 章 分：多模态模块（image_ingestion.py）

## 子模块 5.1 三条技术路线选型（无 GPU 约束下）

| 路线 | 原理 | 优点 | 缺点 |
|---|---|---|---|
| **caption 化（本项目）** | 视觉模型把图转文字描述→和文本统一编码 | CPU 零负担、文搜图/图搜文都能走、描述可读 | 非"真"跨模态（图搜图弱） |
| CLIP 双塔 | 图文各自编码进**同一向量空间** | 真跨模态（图搜图） | CPU 推理慢、中文需 CN-CLIP |
| 多模态 LLM embedding | 端到端多模态向量 | 效果最好 | API 贵、生态不成熟 |

## 子模块 5.2 实现走读：缓存与降级是灵魂

```python
def _caption_image(self, image_path, dish_name, file_hash) -> str:
    cache_key = sha256(路径 + 文件哈希 + 模型名 + prompt版本)   # ① 内容寻址缓存键
    if cached := self._cache.get(cache_key):
        return cached
    caption = self._call_vision_model(...)                    # ② 并发4路调API
    caption = f"《{dish_name}》图片：{caption}"                # ③ 菜名前缀提升检索信号
    self._cache[cache_key] = caption                          # ④ 落盘JSON缓存
```

- **缓存键的四个组成部分各有意义**：文件哈希（图变了重新生成）、模型名（换模型重来）、
  prompt 版本（改提示词重来）、路径（同图不同位置算不同资产）。重建索引永远不重复计费。
- **降级**：单图失败→跳过该图继续（strict 模式可改为报错）；整体失败→纯文本建库照常可用。
- **LFS 指针识别**：`<2KB 且以 "version https://git-lfs" 开头` = 占位文本不是图，直接跳过。

## 子模块 5.3 图片块的特殊设计

图片块的 `parent_id` 指向所属菜谱 → 命中图片自动给父文档**加权**（回溯排序按命中块数）；
`modality=image + image_path` → 文搜图 = `modality=="image"` 过滤 + caption 检索；
生成时把 image_path 附进上下文 → LLM 可以说"配图见 xxx.jpg"。

> **知识点：推理型视觉模型的坑**
> DeepSeek-Vision 是推理模型：max_tokens 先被 reasoning 消耗，给 300 会返回空 content。
> 必须 max_tokens≥2000 并在 prompt 里写"直接输出描述"。这类"沉默失败"只能靠实测发现——单测里务必断言"非空"。

**动手验证**：`.\.venv\Scripts\python.exe main.py --retrieve-images "四季豆炒肉末 成品图"`

---

# 第 6 章 分：权限模块（贯穿 2/3/11 章，此处总述）

## 6.1 威胁模型先行

客服场景：公开菜谱（guest/user 可见）vs 内部工艺文档（staff 专属）。
**红线**：普通用户无论怎么问，都不能检索到 internal 内容。

## 6.2 执行链：role 如何变成数据库过滤

```
JWT(role=staff) → api/auth.py 校验签名 → Principal(role="staff")
  → ROLE_ALLOWED_VISIBILITY["staff"] = ["public","internal"]
  → staff 可见全部 → expr=None（不过滤）
  → user 只见 public → expr='visibility in ["public"]'
  → 下推 Milvus，在数据库层执行（绕不过）
```

**三道防线**：① 表达式白名单构建（第3章）；② 服务端下推（不依赖应用层自觉）；
③ API 层 role 只取 JWT claim（请求体写 role=staff 无效——有单测锁死）。

## 6.3 优雅降级：查不到 ≠ 报错

user 问"速冻水饺（internal）的生产工艺"：不是返回空、不是 403，而是**返回相关的公开菜谱**
（手工水饺等）——权限守住了，体验也守住了。这个语义写进了评测：permission_denied 的
"泄漏"定义是"返回了 internal 条目"，而不是"返回了任何东西"。

**动手验证**：
```powershell
.\.venv\Scripts\python.exe main.py --retrieve "速冻水饺的生产工艺" --role user   # 只有公开菜谱
.\.venv\Scripts\python.exe main.py --retrieve "速冻水饺的生产工艺" --role staff  # internal 排第一
```

---
（第 7-12 章：Agent、生成、追踪、评测、服务化、工程化，见下半部分）

# 第 7 章 分：Agentic RAG 模块（agentic_rag.py）——本项目最核心的进阶设计

## 7.1 动机：固定管线的三个"不会"

固定管线（classic）= 写死的 `路由if-else → 检索 → 生成`。它：
1. **不会选检索方式**——问天气也去查菜谱库、要图片也走文本检索；
2. **不会自救**——检索空了直接拒答，不会想"是不是我搜的词不行"；
3. **不会多轮**——"那第二道呢？"这种指代完全无法处理。

Agentic RAG = 把检索能力**封装成工具**，让 LLM 自己决定"调哪个、传什么参数、结果不好怎么办"。

> **知识点：Agent 的本质公式**：`Agent = LLM(决策) + Tools(能力) + Loop(迭代) + State(记忆)`
> 与 ReAct（Reason+Act 循环）的关系：ReAct 是"模型输出文本思考再调工具"的早期形态；
> function calling 时代，思考压缩为结构化工具选择，循环由框架（LangGraph）承载。

## 7.2 LangGraph 五个核心概念（对着代码学）

```python
class AgentState(TypedDict, total=False):   # ① 状态：一个会变的字典
    question: str
    history: list[dict[str, Any]]           #    多轮记忆
    route: str                              #    当前选的工具
    chunks: list[Document]                  #    检索结果
    retry_pending: bool                     #    重试标志（有坑，见7.6）
    ...

graph = StateGraph(AgentState)
graph.add_node("route",    self._route_node)      # ② 节点=函数：收state、改state、返回state
graph.add_node("act",      self._act_node)
graph.add_node("reflect",  self._reflect_node)
graph.add_node("generate", self._generate_node)
graph.add_edge(START, "route")                    # ③ 固定边
graph.add_conditional_edges("reflect",            # ④ 条件边：函数返回值决定下一跳
    self._decide_after_reflect, {"act": "act", "generate": "generate", END: END})
self.graph = graph.compile()                      # ⑤ 编译成可invoke的图
result = self.graph.invoke({"question": ...}, config={"recursion_limit": 8})  # 递归上限防死循环
```

**图形状**：
```
START → route → act → reflect ─┬─(retry_pending)→ act（改写重试，最多1次）
                    ↑          ├─(有结果)→ generate → END
                    └──────────┴─(无结果且已改写过)→ generate（拒答）→ END
```

## 7.3 工具设计：LLM 怎么"看懂"工具

工具 = 一段 JSON 说明书（name + description + 参数 schema）。**描述写得越好，选择越准**：

```python
{
  "name": "search_recipes",
  "description": "在菜谱知识库中检索菜品的做法、食材、技巧等详细内容",
  "parameters": {
    "query":     {"type": "string", "description": "适合向量检索的中文查询词，如'红烧肉怎么做'"},
    "category":  {"type": "string", "enum": ["荤菜","素菜","汤品",...], "description": "菜品分类（可选）"},
    "difficulty":{"type": "string", "enum": ["非常简单","简单",...], "description": "烹饪难度（可选）"},
  }
}
```

- **enum 是硬约束**：LLM 只能从枚举里选，幻觉出"川菜"会被 schema 拒绝；
- **参数描述里的示例**（"如'红烧肉怎么做'"）教会 LLM 怎么改写 query；
- 四个工具各管一类意图：search_recipes（详情）/ search_images（图）/ list_by_metadata（浏览，只收过滤参数）/ direct_answer（闲聊短路）。

工具列表由 `build_agent_tools(domain)` 从领域配置**动态生成**——换领域时枚举和描述自动换。

## 7.4 路由节点：function calling 的调用与防御

```python
def _route_llm_call(self, question, history=None):
    messages = [
        {"role": "system", "content": self.domain.agent_router_prompt},
        *self._history_messages(history),        # 多轮：历史作为消息注入（指代消解）
        {"role": "user", "content": question},
    ]
    message = self._llm().bind_tools(self.tools).invoke(messages)
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        call = tool_calls[0]
        return {"route": call["name"], "tool_args": dict(call["args"])}   # LLM选的工具+参数
    if content := str(message.content).strip():   # 没调工具但说了话 → 当直接回答
        return {"route": "direct_answer", "tool_args": {"answer": content}}
    raise ValueError("路由 LLM 未返回工具调用")
```

**防御性设计**（_route_node 捕获一切异常）：LLM 返回了不存在的工具名/格式烂掉 →
降级为 `search_recipes(query=原问题)` 默认检索——**Agent 的决策可以错，系统不能崩**。

## 7.5 反思节点：空结果的自我救赎

```python
def _reflect_node(self, state):
    if not state.get("chunks") and state.get("rewrites_used", 0) < MAX_REWRITES:
        rewritten = self._rewrite_query(state["question"], history=state.get("history"))
        state["rewrites_used"] += 1
        state["route"] = "search_recipes"
        state["tool_args"] = {"query": rewritten}   # 用改写后的query再试一次
        state["retry_pending"] = True
        return state
```

改写 prompt 明确要求"结合对话背景解决指代，加入菜品名/食材关键词"。用户问"那种酸酸甜甜的排骨"
检索失败 → 改写成"糖醋排骨 做法" → 重试命中。**为什么限制 MAX_REWRITES=1**：防止
"检索空→改写→还空→再改写"的循环烧钱，递归上限 8 是第二重保险。

## 7.6 状态合并语义：本模块最深的坑

**现象**：改写重试后陷入死循环，直到触发 recursion_limit。
**根因**：LangGraph 节点返回的 dict 是**增量合并**进 state 的——返回里**缺省的 key 保留旧值**。
在节点里 `state.pop("retry_pending")` 无效（旧值仍在 channel 里）。

**解法**：act 节点开头显式写 `state["retry_pending"] = False`（覆盖而非删除）。

> 这个坑值得记住：**声明式状态框架里，"清除"也必须是显式写入**。类似 React 的 setState 心智。

## 7.7 四级错误兜底（分级降级，全部落 trace）

| 故障点 | 兜底动作 | trace 事件 |
|---|---|---|
| 路由 LLM 失败 | 退回默认检索 search_recipes(原问题) | route_fallback |
| 工具执行失败 | 退回混合检索 hybrid_search | tool_fallback |
| 检索彻底不可用 | 无 LLM 规则回答（"检索服务暂不可用，可浏览分类：…"） | retrieval_unavailable |
| 生成 LLM 失败 | 返回检索到的菜名列表（规则摘要） | generate_fallback |

设计原则：**每一层失败都降一级而不是抛异常**；用户体验上"答得差"好过"白屏"；
工程上每个降级都是事件，badcase 可回溯到具体是哪层降的级。

## 7.8 多轮记忆的两个注入点

1. **路由**：历史作为标准 chat messages 插入（`[system, user, assistant, user]`）——
   LLM 在选工具时就能解析"它"指什么，抽出的 query 参数直接带上下文；
2. **生成**：历史压缩成文本前缀（每条截断120字）拼在问题前——回答能接得上话。

历史由调用方维护（CLI 循环 / API session），agent 本身无状态——**状态外置便于水平扩展**。

**动手验证**：`.\.venv\Scripts\python.exe main.py --chat-mode agent`，
先问"推荐一道硬菜"再问"它怎么做"，观察第二条回答末尾 `[agent] route=... hits=...` 与答案正确性。

---

# 第 8 章 分：生成模块（generation_integration.py）

## 8.1 LCEL：用管道组合 LLM 应用

```python
chain = (
    {"question": RunnablePassthrough(), "context": lambda _: context}  # ① 组装输入
    | ChatPromptTemplate.from_template(prompt)                         # ② 模板
    | self.llm                                                         # ③ 模型
    | StrOutputParser()                                                # ④ 提取文本
)
response = chain.invoke(query)      # 同一结构还能 .stream(query) 直接变流式
```

> **知识点：LCEL（LangChain Expression Language）**
> 把"输入处理→prompt→模型→输出解析"声明为管道，`invoke/stream/batch` 一套接口通用。
> 好处：同一份链定义，同步/流式/批量零改动；坏处：调试栈深。本项目 CLI 流式输出就复用了同一 chain 的 `.stream()`。

## 8.2 上下文构造：决定回答质量的无名英雄

```python
# 每篇父文档带元数据头 + 正文，总量截断到 max_context_chars（6000字）
【食谱 1】 简易红烧肉 | 分类: 荤菜 | 难度: 中等
# 简易红烧肉的做法 ……（正文）
==================================================  ← 分隔线防止文档间互相"渗色"
【食谱 2】 ...
```

三个细节：① **元数据头**让 LLM 能区分来源（引用的前提）；② **首篇保底**（截断时至少保留第一篇开头，
防止长文档把上下文挤成空）；③ 分隔线降低跨文档串话。

## 8.3 引用溯源：双层设计

- **内联层（软）**：prompt 指示"引用某食谱内容时用【食谱 N】标注"——上下文头就是【食谱 N】格式，
  LLM 照抄序号成本极低；但 LLM 可能不配合。
- **附录层（硬）**：`format_citations(docs)` 由**实际检索到的父文档**确定性生成：
  ```
  ——
  📚 以上回答参考自：
  - 简易红烧肉（dishes/meat_dish/简易红烧肉/简易红烧肉.md）
  ```
  不依赖 LLM，生成降级时引用依然准确可验证。

> **设计思想**："LLM 负责体验（内联标注好看），程序负责正确性（附录可审计）"——
> 对一切 LLM 参与的功能都值得问一句：如果模型不听话，我的兜底是什么？

## 8.4 流式输出

生成器逐块 `yield`（LCEL `.stream()`），CLI 直接打印；服务层包成 SSE（第 11 章）。

---

# 第 9 章 分：追踪模块（observability.py）

## 9.1 设计：查询级 trace

```python
class QueryTrace:
    query_id = uuid4().hex[:12]          # 每查询一个
    events: [{"ts", "event", ...details}] # 节点事件流
    # finish() 序列化成一行 JSON 追加写 logs/query_trace.jsonl（线程锁保护）
```

一条 trace 长这样：
```json
{"query_id": "f3078e27def1", "pipeline": "agent", "role": "user",
 "events": [
   {"event": "route", "route": "search_images", "args": {"query": "红烧肉成品图"}},
   {"event": "act", "hits": 2, "parents": 2},
   {"event": "summary", "route": "search_images", "rewrites": 0}],
 "answer": "...", "error": null, "elapsed_ms": 4730}
```

**能回答的问题**：这个 badcase 当时路由到了哪？检索命中几条？有没有触发降级？耗时多少？
——客服工单回溯、上线后 badcase 挖掘的数据源。API 的 `/api/traces/{query_id}` 直接暴露。

## 9.2 LangSmith：零代码接入

LangChain 生态的托管 tracing：设 `LANGSMITH_TRACING=true + LANGSMITH_API_KEY`，
所有 LCEL/LangGraph 调用自动上报可视化（每个 prompt 全文、token 数、延迟）。
本地 JSONL 是兜底，LangSmith 是增强——**可观测不能依赖外部服务可用性**。

---

# 第 10 章 分：评测模块（eval/）——整个项目的"裁判"

## 10.1 为什么先建评测再加功能

没有评测的迭代 = 蒙着眼改代码：加了重排，变好了还是变坏了？不知道。
本项目顺序刻意是：**先 golden set + 指标脚本，再做质量优化（重排实验）**——
每个改动都有 before/after 数字。这是本教程最想传递的工程习惯。

## 10.2 golden set 设计：五类意图覆盖

| 意图 | 数量 | 形态 | 考什么 |
|---|---|---|---|
| detail | 40 | "X怎么做"→期望命中菜X | 基础检索 |
| keyword | 5 | "红烧菜品有哪些做法"→期望菜名含"红烧" | 词法召回 |
| browse | 18 | "推荐几道简单的汤"→期望过滤精确 | 元数据过滤 |
| permission_denied | 5 | user 查 internal 内容→期望零泄漏 | 权限红线 |
| permission_allowed | 5 | staff 查同样内容→期望可见 | 权限功能 |

种子由脚本从语料元数据**程序化采样**（保证菜名/分类和真实数据一致），再人工微调。

## 10.3 指标手算示例

**MRR（Mean Reciprocal Rank）**：期望文档在结果里排第 r 位得 1/r，不命中得 0，全体取平均。
> 3 个查询的期望文档分别排第 1、2、5 位 → MRR = (1/1 + 1/2 + 1/5)/3 = 0.567

**Hit@k**：期望文档进前 k 即得分 1。**过滤精确率**（browse 类）：返回结果中符合过滤条件的占比。
**权限泄漏**：返回列表中出现任何 `visibility=internal` 条目即泄漏
（注意：返回公开菜谱不算——那是设计好的降级，不是泄漏）。

## 10.4 LLM-as-judge

DeepSeek 按 5 分制给答案打 faithfulness（是否忠于检索内容）/ relevance（是否切题），
要求只输出 JSON，脚本解析打分。用途：classic vs agent 双管线对比
（faithfulness 4.875 vs 4.75——agent 略低主因是拒答场景被扣分，但"不硬答"正是客服需要的）。
**偏差缓解**：量表定义写死在 prompt、检索上下文一并给 judge、双管线同一 judge 同批跑。

## 10.5 重排实验全流程（评测驱动决策的样板）

提出假设（CE 精排应提升排序）→ 三组实现（基线/纯CE/融合）→ 跑评测 →
数字否定了假设（0.931 > 0.902 > 0.879）→ 归因（精确菜名查询下 BM25 已最强）→
决策（保留实现+开关，默认关）。**这五步就是算法工作的日常**。

**动手验证**：
```powershell
.\.venv\Scripts\python.exe evalun_eval.py --top-k 5          # 检索评测
.\.venv\Scripts\python.exe evalun_judge.py --sample 8        # LLM judge
type .artifacts\evaletrieval_eval_*.json | findstr summary     # 逐条明细
```

---

# 第 11 章 分：服务化模块（api/）

## 11.1 FastAPI 三件套：路由 / 依赖注入 / lifespan

```python
@asynccontextmanager
async def lifespan(app):
    # 启动时一次性装载：嵌入模型+Milvus连接+agent图（约25秒），常驻内存
    # → 所有后续请求零冷启动（CLI 每次启动 30s vs 服务只付一次）
    app.state.rag = system
    yield   # 服务运行期；此处之后可放清理逻辑
```

`Depends` 依赖注入让"取系统实例/校验身份"与业务解耦；
测试时一行 `create_app(rag=fake)` 注入假系统，全部 API 测试离线跑。

## 11.2 SSE 流式

```python
async def event_stream():
    for chunk in rag.ask_question(..., stream=True):
        yield f'data: {json.dumps({"type": "token", "content": chunk})}

'
    yield f'data: {json.dumps({"type": "done", "query_id": qid})}

'
return StreamingResponse(event_stream(), media_type="text/event-stream")
```

> **SSE vs WebSocket**：SSE 单向（服务器→客户端）、HTTP 原生、自动重连，
> LLM 流式输出天然单向所以够用；WebSocket 双向，需要客户端中途干预才必要。
> 生成中途异常 → yield 一个 error 事件优雅收尾，连接不裸断。

## 11.3 JWT 鉴权与越权防护

JWT = `header.payload.signature`，HS256 用同一密钥签发/校验。关键设计：
**role 只从 token 的 payload 里取**——请求体里的 role 仅开发模式生效，
带 token 时填什么都无效（单测锁死：token=staff + body role=guest → 系统收到 staff）。
`python -m api.mint_token --role staff` 生成测试 token；未配密钥自动降级免鉴权开发模式。

## 11.4 会话内存

`app.state.sessions`（dict + 线程锁）存 session_id → 最近 3 轮，FIFO 驱逐超限会话。
注释写明"多副本部署应换 Redis"——内存方案只对单 worker 成立，
**知道边界和知道方案一样重要**。

## 11.5 一个字符引发的 bug：charset

现象：PowerShell 调接口中文全乱（变成 äçé）。根因：响应 Content-Type 缺 charset 时
老版 .NET 按 Latin-1 解码 UTF-8 字节。修复：自定义响应类显式声明
`application/json; charset=utf-8`。教训：**协议层显式优于隐式**。

---

# 第 12 章 分：工程化专题（贯穿全部）

## 12.1 可复现构建三件套

确定性 ID（同语料同 ID）→ 语料指纹 manifest（配置/语料变了拒载旧索引）→
原子写（tmp+replace，写一半断电不留半文件）。
合起来：**任何时刻磁盘上的索引都能证明自己对应哪份语料**。

## 12.2 离线优先加载（HF 模型）

`huggingface_hub` 在 **import 时**把 `HF_HUB_OFFLINE` 固化成模块常量——
之后改环境变量无效。正确姿势：

```python
import huggingface_hub.constants as hf_constants
hf_constants.HF_HUB_OFFLINE = True          # 先翻常量
from langchain_huggingface import HuggingFaceEmbeddings   # 再 import
# 离线失败（本地无缓存）→ 恢复在线 → 仍失败 → 切 hf-mirror.com 再试
```

收益：断网/代理挂死时启动照常（模型有本地缓存），首次下载自动走国内镜像。

## 12.3 测试策略：fake 注入

82 个测试全部离线：FakeEmbeddings（定长向量）、FakeMilvusClient（记录调用不联网）、
FakeChatModel（排队返回脚本化响应）、FakeGeneration、FakeCrossEncoder。
测的是**编排逻辑**（路由/兜底/权限/状态机），不测模型本身。
构造器注入（`RerankerModule(model=fake)`）是可测性的来源。

## 12.4 静态检查

ruff（lint+格式）+ mypy（类型）全绿。类型注解在 LangGraph 的 TypedDict state 上尤其值钱——
字段名拼错直接被编辑器/CI 抓住。

---

# 附录 A 踩坑速查（面试细节弹药）

| 坑 | 一句话根因 | 一句话解法 |
|---|---|---|
| Milvus hybrid 过滤失效 | filter kwarg 被静默忽略 | expr 挂每个 AnnSearchRequest |
| 断网启动卡死 | HF 常量 import 时固化 | import 前翻 constants |
| 328"图"全失败 | Git LFS 指针非图片 | <2KB+魔数检测跳过 |
| 视觉 caption 空 | 推理模型吃光 max_tokens | 给足 2000 + "直接输出" |
| agent 重试死循环 | 缺省 key 保留旧值 | 显式覆盖 retry_pending=False |
| 重排掉点 | BM25 精确名信号已最强 | 评测对照后默认关 |
| PowerShell 乱码 | 缺 charset 按 Latin-1 解 | 显式 charset=utf-8 |
| BM25 单测总有返回 | BM25 无匹配也返回 top-k | 夹具模拟服务端语义 |

# 附录 B 建议学习顺序（由浅入深复习路线）

1. 跑通：build-only → retrieve → ask → agent 四条命令，建立体感
2. 读第 2 章 + data_preparation.py（最简单模块，建立 Document/ID/切块直觉）
3. 读第 3-4 章 + vector_store.py / retrieval_optimization.py（核心难点，RRF 手算一遍）
4. 读第 7 章 + agentic_rag.py（进阶核心，自己画一遍状态图）
5. 读第 10 章 + eval/（工程习惯的载体）
6. 其余按需查阅；面试前刷 PROJECT_GUIDE.md 的问答与简历话术

---

*教程完。配套文件：PROJECT_GUIDE.md（面试速查版）、README.md（使用手册）、源码即教材。*
