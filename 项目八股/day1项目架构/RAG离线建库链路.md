# RAG 项目离线建库链路

本文基于当前仓库的真实源码，按调用顺序梳理从原始 Markdown/图片数据到知识库可检索状态的完整链路。

## 阅读约定

- **源码事实**：可以从当前代码的类、函数和调用关系直接确认。
- **架构解释**：说明这一层为什么存在，不把解释冒充成源码行为。
- 本文只讲架构和数据流，不展开 Chunking、Embedding、HNSW、BM25 或融合算法的数学原理。

## 一、建库链路总入口

### 第 1 步：CLI 或 API 触发建库

**文件**

- [`main.py`](main.py)
- [`api/app.py`](api/app.py)

**类 / 函数**

- `main.main()`
- `RecipeRAGSystem.run_interactive()`
- `api.app.create_app()` 内的 `lifespan()`
- 共同进入 `RecipeRAGSystem.build_knowledge_base()`

**输入**

```python
force_rebuild: bool = False
self.config: RAGConfig
```

`RAGConfig` 提供数据目录、索引目录、后端名称和模型等运行配置。

**源码事实**

CLI 的以下模式都会触发建库或索引校验：

```text
--build-only
--retrieve
--retrieve-images
--agent
交互模式
```

FastAPI 服务启动时，`api.app.create_app()` 的 `lifespan()` 会执行：

```python
system = RecipeRAGSystem()
system.initialize_system(load_generation=True)
system.build_knowledge_base()
```

真正统一的建库入口是：

```python
RecipeRAGSystem.build_knowledge_base()
```

**输出**

```python
statistics: dict[str, Any]
```

同时把以下运行时对象挂到 `RecipeRAGSystem`：

```python
self.data_module
self.index_module
self.backend
self.retrieval_module
```

**这一层存在的目的（架构解释）**

把 CLI、API 与具体数据处理和向量库实现隔离；无论外部如何启动，最终都走同一个建库方法。

---

## 二、模块初始化

### 第 2 步：创建数据模块和索引模块

**文件**

- [`main.py`](main.py)
- [`config.py`](config.py)

**类 / 函数**

- `RecipeRAGSystem.initialize_system()`
- `DataPreparationModule.__init__()`
- `IndexConstructionModule.__init__()`

**输入**

```python
self.config.data_path
self.config.internal_categories
self.config.embedding_model
self.config.embedding_revision
self.config.embedding_device
self.config.normalize_embeddings
self.config.index_save_path
```

**源码事实**

`initialize_system()` 创建：

```python
self.data_module = DataPreparationModule(...)
self.index_module = IndexConstructionModule(...)
```

如果只是离线建库，可以使用：

```python
initialize_system(load_generation=False)
```

此时不会初始化答案生成 LLM。

`IndexConstructionModule` 在构造时只保存 Embedding 配置；真正的 Embedding 模型采用懒加载，尚未在这一步生成向量。

**输出**

```python
DataPreparationModule
IndexConstructionModule
```

**这一层存在的目的（架构解释）**

先完成依赖装配，再开始读取语料；同时让“只建库”不必依赖生成模型 API。

---

## 三、Markdown 加载与 Parent Document

### 第 3 步：扫描并加载 Markdown 文件

**文件**

[`rag_modules/data_preparation.py`](rag_modules/data_preparation.py)

**类 / 函数**

```python
DataPreparationModule.load_documents()
```

**输入**

```python
self.data_path: pathlib.Path
```

数据目录默认由 `config.py` 中的 `RAGConfig.data_path` 指定。

**源码事实**

`load_documents()` 执行以下操作：

1. 校验数据路径存在且为目录。
2. 使用 `data_root.rglob("*.md")` 递归发现 Markdown。
3. 按相对路径稳定排序。
4. 默认跳过路径中名为 `template` 的目录。
5. 使用 UTF-8 读取每个文件。
6. 每个 Markdown 文件创建一个 LangChain `Document`。

**输出**

```python
self.documents: list[Document]
```

这里的每个 `Document` 都是一个完整 Markdown 文件，也就是 Parent Document。

**这一层存在的目的（架构解释）**

把磁盘文件统一转换为后续模块使用的 `Document` 数据结构，并保留完整原文作为最终生成阶段的上下文来源。

---

### 第 4 步：建立 Parent Document 的 ID 和 metadata

**文件**

[`rag_modules/data_preparation.py`](rag_modules/data_preparation.py)

**类 / 函数**

- `DataPreparationModule._stable_id()`
- `DataPreparationModule.load_documents()`
- `DataPreparationModule._enhance_metadata()`

**输入**

```python
source_path: str
content: str
content_hash: str
```

**源码事实**

父文档 ID 在 `load_documents()` 中建立：

```python
parent_id = self._stable_id("parent", source_path)
revision_id = self._stable_id("revision", source_path, content_hash)
```

父文档创建时的主要 metadata：

```python
{
    "source": str,
    "source_path": str,
    "source_hash": str,
    "parent_id": str,
    "revision_id": str,
    "doc_type": "parent",
    "file_type": "markdown",
    "modality": "text",
    "visibility": "public",
    "image_path": "",
}
```

随后 `_enhance_metadata()` 继续加入或修正：

```python
{
    "category": str,
    "dish_name": str,
    "difficulty": str,
    "visibility": "public" | "internal",
}
```

父文档还会登记到：

```python
self._parent_documents[parent_id] = document
```

**输出**

```python
Document(
    page_content=完整 Markdown,
    metadata={... "doc_type": "parent", ...},
)
```

**这一层存在的目的（架构解释）**

为每篇原始文档建立稳定身份、版本身份和业务元数据，使子块能够回溯原文，也让后续索引判断语料是否发生变化。

---

## 四、Child Chunk 产生

### 第 5 步：将 Parent Document 切成文本 Child Chunk

**文件**

[`rag_modules/data_preparation.py`](rag_modules/data_preparation.py)

**类 / 函数**

- `DataPreparationModule.chunk_documents()`
- `DataPreparationModule._markdown_header_split()`

**输入**

```python
self.documents: list[Document]
```

即上一步得到的完整父文档列表。

**源码事实**

`chunk_documents()` 调用 `_markdown_header_split()`，按照 Markdown 标题结构产生多个子 `Document`。

如果某篇文档切分失败或没有产生子块，代码会将整篇父文档作为一个子块继续处理。

每个文本 Child Chunk 的 metadata 由父文档 metadata、标题切分 metadata 和子块字段合并得到：

```python
{
    **parent.metadata,
    **split_chunk.metadata,
    "chunk_id": child_id,
    "parent_id": parent.metadata["parent_id"],
    "doc_type": "child",
    "chunk_index": int,
    "chunking_version": "markdown-headers-v1",
    "batch_index": int,
    "chunk_size": int,
}
```

其中子块 ID 创建方式为：

```python
child_id = self._stable_id(
    "chunk",
    parent_id,
    revision_id,
    str(chunk_index),
    split_chunk.page_content,
)
```

然后保存为 metadata 中的：

```python
metadata["chunk_id"] = child_id
```

并记录父子映射：

```python
self.parent_child_map[chunk_id] = parent_id
```

**重要命名说明**

当前源码不存在：

```python
metadata["child_id"]
```

项目在概念上称其为 Child Chunk，但实际身份字段名是：

```python
metadata["chunk_id"]
```

**输出**

```python
self.chunks: list[Document]
```

此时列表中是文本 Child Chunk。

**这一层存在的目的（架构解释）**

让检索面向主题更集中的局部内容，同时保留 `parent_id`，使查询阶段可以从命中子块回到完整父文档。

---

## 五、图片进入检索体系

### 第 6 步：发现图片并生成 caption

**文件**

- [`main.py`](main.py)
- [`rag_modules/image_ingestion.py`](rag_modules/image_ingestion.py)

**类 / 函数**

- `RecipeRAGSystem._ingest_images()`
- `ImageIngestionModule.ingest_images()`
- `ImageIngestionModule._images_for_parent()`
- `ImageIngestionModule._caption_image()`
- `ImageIngestionModule._build_chunk()`

**输入**

```python
documents: list[Document]  # Parent Documents
```

以及：

```python
DEEPSEEK_API_KEY
vision_model
image_caption_cache.json
```

**源码事实**

图片处理存在两层开关：

1. `enable_image_ingestion=False` 时直接跳过。
2. 没有 `DEEPSEEK_API_KEY` 时直接降级为纯文本建库。

图片发现方式：

1. 根据父文档的 Markdown 文件路径找到同目录图片。
2. 只处理支持的图片后缀。
3. 跳过过小文件和 Git LFS 指针文件。

Caption 处理方式：

1. 根据图片路径、文件哈希、模型和 Prompt 版本计算缓存键。
2. 缓存命中时直接复用 caption。
3. 缓存未命中时调用视觉模型。
4. 每张图片最终生成一个文本 caption。

**输出**

```python
image_chunks: list[Document]
```

每个图片 Child Chunk 的 `page_content` 是 caption，而不是图片二进制数据。

主要 metadata：

```python
{
    "parent_id": str,
    "revision_id": str,
    "chunk_id": str,
    "doc_type": "child",
    "modality": "image",
    "file_type": "image",
    "image_path": str,
    "chunk_index": int,
    "chunking_version": "image-caption-v2",
    "chunk_size": int,
    "source": str,
    "source_hash": str,
    "category": str,
    "dish_name": str,
    "difficulty": str,
    "visibility": str,
}
```

图片 `chunk_id` 根据相对图片路径和文件哈希产生。

成功生成的图片块会追加到：

```python
self.data_module.chunks
```

如果图片处理整体失败，`RecipeRAGSystem._ingest_images()` 捕获异常并返回空列表，建库继续使用文本语料。

**这一层存在的目的（架构解释）**

把不可直接参与当前文本检索的图片转换为可 Embedding、可稀疏检索的文字描述，同时用 `image_path` 保留返回真实图片的能力。

当前实现属于“图片 caption 文本检索”，不是原始图片向量检索。

---

## 六、合并最终待索引 Chunk

### 第 7 步：合并文本块和图片块

**文件**

[`main.py`](main.py)

**类 / 函数**

```python
RecipeRAGSystem.build_knowledge_base()
```

**输入**

```python
chunks: list[Document]        # 文本 Child Chunk
image_chunks: list[Document]  # 图片 caption Child Chunk
```

**源码事实**

合并方式：

```python
all_chunks = chunks + image_chunks
```

后续无论选择 FAISS 还是 Milvus，进入索引的都是：

```python
all_chunks: list[Document]
```

**输出**

统一待索引 Chunk 列表。

**这一层存在的目的（架构解释）**

让文本内容和图片 caption 共享同一套 Embedding、元数据过滤和 Retrieval 接口，上层无需维护两套完全独立的检索系统。

---

## 七、后端选择与 Embedding

### 第 8 步：根据配置选择 Milvus 或 FAISS

**文件**

- [`main.py`](main.py)
- [`config.py`](config.py)

**类 / 函数**

- `RecipeRAGSystem.build_knowledge_base()`
- `RAGConfig.backend`

**输入**

```python
backend_name: str
all_chunks: list[Document]
force_rebuild: bool
```

**源码事实**

当前支持：

```text
milvus
faiss
```

默认配置是：

```python
backend = "milvus"
```

两个后端在 `build_knowledge_base()` 中形成明确分支。

**输出**

进入 Milvus 建库路径或 FAISS 建库/加载路径。

**这一层存在的目的（架构解释）**

保持上层数据准备和查询接口不变，同时允许使用服务化向量库或本地向量索引。

---

## 八、Milvus 建库路径

### 第 9A 步：在 Milvus 中构建或复用 collection

**文件**

- [`main.py`](main.py)
- [`rag_modules/index_construction.py`](rag_modules/index_construction.py)
- [`rag_modules/vector_store.py`](rag_modules/vector_store.py)

**类 / 函数**

- `IndexConstructionModule.setup_embeddings()`
- `MilvusVectorStore.__init__()`
- `MilvusVectorStore.build_index()`
- `MilvusVectorStore._create_collection()`
- `MilvusVectorStore._rows_for_insert()`
- `MilvusVectorStore._write_manifest()`

**输入**

```python
all_chunks: list[Document]
force_rebuild: bool
embeddings: Embeddings
collection_name: str
milvus_uri: str
```

**源码事实：Embedding 在哪里生成**

`main.py` 首先取得 Embedding 对象：

```python
embeddings = self.index_module.setup_embeddings()
```

实际文档向量在 `MilvusVectorStore.build_index()` 的批量写入循环中生成：

```python
vectors = self.embeddings.embed_documents(
    [chunk.page_content for chunk in batch]
)
```

因此，文本 Child Chunk 和图片 caption Child Chunk 都对各自的 `page_content` 生成 Embedding。

**源码事实：索引如何构建**

`build_index()` 依次执行：

1. 校验所有 Chunk 都有唯一 `chunk_id`。
2. 根据全部 Chunk 计算语料指纹。
3. 检查现有 collection、manifest 和行数是否与当前语料一致。
4. 一致时复用现有 collection。
5. 不一致或强制重建时，删除旧 collection 并创建新 collection。
6. 建立向量字段、内容字段和 metadata 标量字段。
7. 分批为 `page_content` 生成 Embedding。
8. 将 `chunk_id`、向量、内容和 metadata 组成 Milvus row。
9. 批量插入并 `flush()`。
10. 写出 `milvus_manifest.json`。

Milvus 中的 `chunk_id` 是主键。

内容字段还会通过 Milvus 内置函数产生供稀疏检索使用的字段；本文不展开该算法。

**输出**

```python
milvus_store: MilvusVectorStore
rebuilt: bool
```

并写入：

```python
self.backend = milvus_store
```

**这一层存在的目的（架构解释）**

将全部 Chunk、向量和可过滤 metadata 持久化到服务化向量库，并通过 manifest 避免错误复用与当前语料不匹配的 collection。

---

## 九、FAISS 建库路径

### 第 9B 步：加载或构建本地 FAISS 索引

**文件**

- [`main.py`](main.py)
- [`rag_modules/index_construction.py`](rag_modules/index_construction.py)
- [`rag_modules/vector_store.py`](rag_modules/vector_store.py)

**类 / 函数**

- `IndexConstructionModule.load_index()`
- `IndexConstructionModule.build_vector_index()`
- `IndexConstructionModule.save_index()`
- `FaissBackend.__init__()`

**输入**

```python
all_chunks: list[Document]
force_rebuild: bool
```

**源码事实：先尝试安全复用**

非强制重建时：

```python
vectorstore = self.index_module.load_index(all_chunks)
```

`load_index()` 会检查：

- 当前语料指纹
- Chunk 数量和身份
- Chunking 配置
- Embedding 配置和维度
- 索引文件摘要
- 文档映射文件
- FAISS 中实际向量数量

只有完全匹配才返回 FAISS 对象，否则返回 `None`，让上层重建。

**源码事实：Embedding 在哪里生成**

需要重建时调用：

```python
FAISS.from_documents(
    documents=all_chunks,
    embedding=self.setup_embeddings(),
    ids=self._storage_ids(all_chunks),
)
```

`FAISS.from_documents()` 内部使用 `Embedding` 对所有 `Document.page_content` 生成向量。

**源码事实：保存哪些产物**

```text
index.faiss
documents.json
manifest.json
```

- `index.faiss` 保存向量索引。
- `documents.json` 保存索引位置、docstore ID 与 `chunk_id` 的映射，不直接保存完整文档正文。
- `manifest.json` 保存语料、Chunking、Embedding、索引格式和文件摘要信息。

随后包装为统一后端：

```python
self.backend = FaissBackend(vectorstore)
```

**输出**

```python
vectorstore: FAISS
self.backend: FaissBackend
```

**这一层存在的目的（架构解释）**

提供无需独立向量数据库服务的本地索引路径，同时通过当前语料重新注入 `Document`，避免索引和语料版本错配。

---

## 十、进入“知识库可检索”状态

### 第 10 步：创建 RetrievalOptimizationModule

**文件**

- [`main.py`](main.py)
- [`rag_modules/retrieval_optimization.py`](rag_modules/retrieval_optimization.py)

**类 / 函数**

- `RetrievalOptimizationModule.__init__()`
- `RetrievalOptimizationModule.setup_retrievers()`

**输入**

```python
vectorstore: VectorStoreBackend
chunks: list[Document]
candidate_k: int
rrf_k: int
reranker: RerankerModule | None
```

**源码事实**

两条后端分支最终都会创建同一个检索编排对象：

```python
self.retrieval_module = RetrievalOptimizationModule(
    self.backend,
    all_chunks,
    candidate_k=self.config.candidate_k,
    rrf_k=self.config.rrf_k,
    reranker=self._build_reranker(),
)
```

Milvus 后端声明支持服务端混合检索，因此 `setup_retrievers()` 不创建本地稀疏索引。

FAISS 后端不支持服务端混合检索，因此 `setup_retrievers()` 会基于 `all_chunks` 创建本地稀疏检索器，供查询阶段与 FAISS 结果组合使用。

建库最后通过 `DataPreparationModule.get_statistics()` 返回文档与 Chunk 统计。

**输出**

```python
self.backend: MilvusVectorStore | FaissBackend
self.retrieval_module: RetrievalOptimizationModule
statistics: dict[str, Any]
```

此时以下调用已经可用：

```python
RecipeRAGSystem.retrieve()
RecipeRAGSystem.retrieve_images()
RecipeRAGSystem.ask_question()
RecipeRAGSystem.ask_agent()
```

其中后两个还需要 Generation 模块可用。

**这一层存在的目的（架构解释）**

“索引文件或 collection 已存在”并不等于上层系统已经可检索；只有后端对象和统一 Retrieval 编排对象都完成装配，`RecipeRAGSystem` 才进入可接收 Query 的运行状态。

---

## 十一、metadata 建立位置总表

| metadata | Parent Document | 文本 Child Chunk | 图片 Child Chunk | 建立位置 |
|---|---:|---:|---:|---|
| `parent_id` | 是 | 继承 / 显式写入 | 继承 | `DataPreparationModule.load_documents()` |
| `revision_id` | 是 | 继承 | 继承 | `DataPreparationModule.load_documents()` |
| `chunk_id` | 否 | 是 | 是 | `_markdown_header_split()` / `ImageIngestionModule._build_chunk()` |
| `child_id` | 不存在 | 不存在 | 不存在 | 当前源码未使用该字段 |
| `doc_type` | `parent` | `child` | `child` | 父文档创建 / 两类 Child Chunk 创建处 |
| `modality` | `text` | 继承为 `text` | `image` | 父文档创建 / 图片块创建 |
| `file_type` | `markdown` | 继承为 `markdown` | `image` | 父文档创建 / 图片块创建 |
| `source` | Markdown 绝对路径 | 继承 | 图片绝对路径 | 父文档创建 / 图片块创建 |
| `source_path` | Markdown 相对路径 | 继承 | 继承 | 父文档创建 |
| `source_hash` | Markdown 内容哈希 | 继承 | 图片文件哈希 | 父文档创建 / 图片块创建 |
| `dish_name` | 是 | 继承 | 继承 | `_enhance_metadata()` |
| `category` | 是 | 继承 | 继承 | `_enhance_metadata()` |
| `difficulty` | 是 | 继承 | 继承 | `_enhance_metadata()` |
| `visibility` | 是 | 继承 | 继承 | `_enhance_metadata()` |
| `chunk_index` | 否 | 是 | 是 | 两类 Child Chunk 创建处 |
| `chunk_size` | 否 | 是 | 是 | `chunk_documents()` / 图片块创建处 |
| `chunking_version` | 否 | 是 | 是 | 两类 Child Chunk 创建处 |
| `image_path` | 空字符串 | 继承为空 | 图片相对路径 | 父文档创建 / 图片块创建 |

## 十二、FAISS 与 Milvus 进入流程的位置

```text
共同部分
Markdown / 图片
→ Parent Document
→ Text Child Chunk / Image Child Chunk
→ all_chunks
→ IndexConstructionModule.setup_embeddings()

后端分叉
├── MilvusVectorStore.build_index(all_chunks)
│   ├── 生成 Embedding
│   ├── 写入 Chunk、向量、内容和 metadata
│   └── 写入 Milvus manifest
│
└── IndexConstructionModule.load_index(all_chunks)
    └── 失败时 build_vector_index(all_chunks)
        ├── 生成 Embedding
        ├── 保存 FAISS
        ├── 保存文档映射
        └── 保存 manifest

重新汇合
→ VectorStoreBackend
→ RetrievalOptimizationModule
→ 知识库可检索
```

## A. 10～15 行极简建库流程图

```text
1. CLI / FastAPI 启动 RecipeRAGSystem
2. initialize_system() 创建数据模块和索引模块
3. load_documents() 递归读取 Markdown
4. 每个 Markdown 生成 Parent Document 和 parent_id
5. _enhance_metadata() 补充菜名、分类、难度、visibility
6. chunk_documents() 将 Parent 切成文本 Child Chunk
7. 每个 Child 建立 chunk_id、parent_id、doc_type=child
8. ingest_images() 将真实图片转换成 caption Child Chunk
9. 文本 Child 与图片 Child 合并为 all_chunks
10. setup_embeddings() 懒加载 Embedding 模型
11. backend=milvus：生成向量并写入 Milvus collection
12. backend=faiss：加载匹配索引，失败则生成向量并重建 FAISS
13. 后端被包装为 VectorStoreBackend
14. 创建 RetrievalOptimizationModule
15. RecipeRAGSystem 进入可检索状态
```

## B. 2 分钟口述版本

这个项目的离线建库入口统一在 `main.py` 的 `RecipeRAGSystem.build_knowledge_base()`。无论是 CLI 的建库、检索、Agent 模式，还是 FastAPI 服务启动，最后都会调用这个方法。

建库开始后，`DataPreparationModule.load_documents()` 会递归读取菜谱目录中的 Markdown。每个 Markdown 文件先成为一个完整的 Parent Document，并在这里建立 `parent_id`、`revision_id`、`doc_type=parent`，同时补充菜名、分类、难度和可见性等 metadata。Parent Document 会保存在内存映射中，供查询阶段从子块回溯完整原文。

接着，`chunk_documents()` 将每个 Parent 切成多个文本 Child Chunk。每个 Child 都建立自己的 `chunk_id`，保留父文档的 `parent_id`，并标记 `doc_type=child`。需要注意，源码没有 `child_id` 字段，实际使用的是 `chunk_id`。

图片走一条可选支路。`ImageIngestionModule` 查找 Markdown 同目录下的真实图片，跳过 LFS 指针，通过视觉模型生成 caption，然后把 caption 包装成 `modality=image` 的 Child Chunk。图片本身不直接生成图像向量，而是让 caption 进入与文本相同的检索体系。

文本 Child 和图片 Child 合并为 `all_chunks` 后，流程根据配置进入 Milvus 或 FAISS。Milvus 路径在 `MilvusVectorStore.build_index()` 中批量生成 Embedding，并把 Chunk、向量、正文和 metadata 写入 collection；如果现有 collection 与语料指纹一致则直接复用。FAISS 路径先调用 `load_index(all_chunks)` 校验本地索引，无法安全复用时才通过 `build_vector_index()` 重新生成向量，并保存 FAISS 文件、文档映射和 manifest。

最后，两种后端都会被交给 `RetrievalOptimizationModule`。到这一步，`RecipeRAGSystem` 同时拥有数据模块、向量后端和检索编排对象，知识库才真正进入可以接受 Query 的状态。

## 十三、事实边界

- 上述调用顺序、类、函数、metadata 字段和后端分支都可以从当前源码直接确认。
- “Parent 用于保留完整上下文”“Child 用于提高检索粒度”等文字属于架构解释，不是额外的源码行为。
- 当前图片路径是 caption 文本检索；源码中没有 CLIP、原始图片 Embedding 或独立图像向量库。
- 当前源码使用 `chunk_id` 表示 Child Chunk 身份，没有 `child_id` metadata。
