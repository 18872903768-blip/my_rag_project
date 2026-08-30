# 菜谱 RAG 系统使用手册

> 全部命令默认在 `E:\Users\all_in_rag\code\C8` 目录、PowerShell 下执行。
> 通用约定：`PY` = `.\.venv\Scripts\python.exe`；需要 LLM 的操作要求 `.env` 里有 `DEEPSEEK_API_KEY`。

---

## 1. 环境准备（一次性）

### 1.1 启动 Milvus（Docker Desktop 里跑起来即可）

```powershell
docker ps    # 确认三个容器 healthy：milvus-standalone / milvus-minio / milvus-etcd
```

服务端点默认 `http://localhost:19530`（`RAG_MILVUS_URI` 可改）。

### 1.2 安装依赖（新机器）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### 1.3 配置密钥

`E:\Users\all_in_rag\.env`（工作区根）或 `code\C8\.env` 至少包含：

```ini
DEEPSEEK_API_KEY=sk-xxx
```

其余全部有默认值，可选项见 `.env.example`（Milvus 地址、视觉模型、重排、JWT、LangSmith 等）。

---

## 2. CLI 用法（main.py）

### 2.1 构建 / 校验知识库

```powershell
# 建库或校验（语料没变→复用索引秒开；变了→自动重建）
.\.venv\Scripts\python.exe main.py --build-only

# 强制全量重建（改了语料/换了embedding模型/清库后）
.\.venv\Scripts\python.exe main.py --build-only --rebuild

# 用 FAISS 降级后端（Milvus 挂掉时）
.\.venv\Scripts\python.exe main.py --build-only --backend faiss
```

### 2.2 检索（不调 LLM，不要 API key）

```powershell
# 混合检索（自动从查询里抽分类/难度过滤）
.\.venv\Scripts\python.exe main.py --retrieve "红烧肉怎么做" --top-k 5

# 指定角色（user 只见公开内容；staff 可见 internal）
.\.venv\Scripts\python.exe main.py --retrieve "速冻水饺的生产工艺" --role staff

# 文搜图（在图片 caption 索引里检索）
.\.venv\Scripts\python.exe main.py --retrieve-images "四季豆炒肉末 成品图" --top-k 3
```

### 2.3 问答

```powershell
# 交互问答（经典固定管线，流式输出）
.\.venv\Scripts\python.exe main.py

# 交互问答（Agentic 模式 + 多轮记忆；每答末尾显示 [agent] route=... hits=...）
.\.venv\Scripts\python.exe main.py --chat-mode agent

# 指定角色进入交互
.\.venv\Scripts\python.exe main.py --chat-mode agent --role staff

# 单次 agent 问答（脚本/演示用）
.\.venv\Scripts\python.exe main.py --agent "推荐一道简单的汤，并告诉我怎么做"
```

交互模式内：输入问题回车作答；输入 `退出` / `quit` / `exit` 结束。

### 2.4 常用参数速查

| 参数 | 作用 |
|---|---|
| `--build-only` | 只建库/校验，不问答 |
| `--retrieve "QUERY"` | 一次混合检索并打印 |
| `--retrieve-images "QUERY"` | 文搜图 |
| `--agent "QUERY"` | 单次 agent 问答（需 API key） |
| `--chat-mode classic\|agent` | 交互模式选链路 |
| `--role guest\|user\|staff` | 权限角色（默认 user） |
| `--backend milvus\|faiss` | 向量后端（默认 milvus） |
| `--rebuild` | 强制重建索引 |
| `--top-k N` | 检索条数 |
| `--log-level DEBUG` | 调试日志 |

### 2.5 添加新图片后入库

把真图（jpg/png/webp，>2KB）放进对应菜品目录 → 跑 `--build-only`（旧图走缓存不重复计费，新图自动 caption 入库）。

---

## 3. 服务模式（V2，FastAPI）

### 3.1 启动 / 停止

```powershell
# 开发模式启动（免鉴权；首次装载模型约25秒，看到 startup complete 即就绪）
.\.venv\Scripts\python.exe -m uvicorn api.app:app --port 8000

# 允许局域网访问
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

停止：服务窗口 `Ctrl+C`。

### 3.2 接口调用（PowerShell）

```powershell
# 健康检查（含 Milvus 行数）
Invoke-RestMethod http://localhost:8000/healthz

# 同步问答（classic）
Invoke-RestMethod -Uri http://localhost:8000/api/ask -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes('{"query":"红烧肉怎么做"}'))

# 同步问答（agent + 多轮会话 + 引用）
Invoke-RestMethod -Uri http://localhost:8000/api/ask -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes('{"query":"推荐一道红烧菜","pipeline":"agent","session_id":"demo"}'))
# 紧接着（同 session_id 会带上文，"它"能被解析）：
Invoke-RestMethod -Uri http://localhost:8000/api/ask -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes('{"query":"它怎么做","pipeline":"agent","session_id":"demo"}'))

# 纯检索
Invoke-RestMethod -Uri http://localhost:8000/api/search -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes('{"query":"红烧肉","top_k":5}'))

# 文搜图
Invoke-RestMethod -Uri http://localhost:8000/api/search -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes('{"query":"四季豆 成品","images_only":true}'))

# 按 query_id 回溯链路
Invoke-RestMethod http://localhost:8000/api/traces/<query_id>
```

### 3.3 SSE 流式

```powershell
curl.exe -N -X POST http://localhost:8000/api/ask/stream `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"推荐一道素菜\"}'
```

事件格式：`{"type":"token","content":"..."}` 逐块 → `{"type":"done","query_id":"..."}` 收尾。

### 3.4 JWT 鉴权（可选）

```powershell
# 1. 在 .env 设置：RAG_JWT_SECRET=<32位以上随机串>，重启服务
# 2. 生成 token
.\.venv\Scripts\python.exe -m api.mint_token --role staff
# 3. 带 token 调用（role 以 token 为准，请求体不可越权）
Invoke-RestMethod -Uri http://localhost:8000/api/ask -Method Post `
  -Headers @{Authorization = "Bearer <token>"} `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes('{"query":"速冻水饺的生产工艺","pipeline":"agent"}'))
```

### 3.5 接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/healthz` | 探活 + Milvus 行数 |
| POST | `/api/ask` | 同步问答（pipeline=classic/agent；session_id 多轮） |
| POST | `/api/ask/stream` | SSE 流式问答 |
| POST | `/api/search` | 纯检索（images_only=true 文搜图） |
| GET | `/api/traces/{query_id}` | 查询级链路回溯 |

---

## 4. 评测

```powershell
# 重新生成 golden set（改语料后跑一次；会覆盖 eval/golden_set.json）
.\.venv\Scripts\python.exe eval\generate_golden_set.py

# 检索评测（Hit@k / MRR / 过滤精确率 / 权限泄漏；报告落 .artifacts/eval/）
.\.venv\Scripts\python.exe eval\run_eval.py --top-k 5

# 对比 FAISS 后端
.\.venv\Scripts\python.exe eval\run_eval.py --backend faiss

# 答案质量评测（DeepSeek 当 judge，classic vs agent 对比；需 API key）
.\.venv\Scripts\python.exe eval\run_judge.py --sample 10
```

---

## 5. 测试与静态检查

```powershell
.\.venv\Scripts\python.exe -m pytest                       # 全量82个离线单测
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -v  # 只跑API层
.\.venv\Scripts\python.exe -m ruff check .                 # lint
.\.venv\Scripts\python.exe -m ruff check . --fix           # lint+自动修复
.\.venv\Scripts\python.exe -m mypy rag_modules main.py api # 类型检查
```

---

## 6. Docker 全栈（可选，一条命令起 Milvus+RAG 服务）

```powershell
# .env 中配好 DEEPSEEK_API_KEY 后：
docker compose up -d --build
docker compose logs -f rag        # 看服务日志（首次建库+下载模型较久）
docker compose down               # 停止
```

注意：compose 里的语料挂载路径默认 `../../data/C8/cook`，按需调整。

---

## 7. 开关速查（.env 可配）

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `RAG_BACKEND` | milvus | milvus / faiss |
| `RAG_MILVUS_URI` | http://localhost:19530 | Milvus 地址 |
| `RAG_TOP_K` / `RAG_CANDIDATE_K` | 3 / 10 | 检索条数 / 每路召回量 |
| `RAG_ENABLE_IMAGE_INGESTION` | true | 建库时是否摄取图片 |
| `RAG_VISION_MODEL` | deepseek-v4-flash-vision-exp | caption 模型 |
| `RAG_INTERNAL_CATEGORIES` | 半成品 | 哪些分类 internal |
| `RAG_RERANK_ENABLED` | false | cross-encoder 精排（实验结论：本语料无增益） |
| `RAG_RERANK_MODEL` / `RAG_RERANK_POOL` | bge-reranker-base / 20 | 精排模型 / 候选池 |
| `RAG_HF_OFFLINE` | 未设 | true=锁死离线（不用网络） |
| `RAG_JWT_SECRET` | 未设 | 设置后 /api/* 需 Bearer JWT |
| `LANGSMITH_TRACING` + `LANGSMITH_API_KEY` | 未设 | 开启 LangSmith 全链路可视化 |

---

## 8. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| 启动卡几分钟在加载模型 | 代理已关但走了在线校验 | 已修复为离线优先；仍卡则检查是否旧进程，或设 `RAG_HF_OFFLINE=true` |
| `Connection refused ...19530` | Milvus 没起 | 启动 Docker Desktop + milvus 容器 |
| 交互/服务正常但报权限 401 | 开了 JWT 未带 token | `python -m api.mint_token` 造 token 带上 |
| PowerShell 返回中文乱码 | 显示编码 | 已修复（charset=utf-8）；仍有则 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` |
| 图片没被入库 | 是 LFS 指针/小于2KB | 换成真实图片再 `--build-only` |
| 用错 Python 环境 | 用了工作区根的 .venv | 必须用 `code\C8\.venv`（本文所有命令已带全路径） |
| 检索结果异常/换语料后 | 索引与语料不匹配 | `--rebuild` 强制重建 |

---

## 9. 常用文件位置

| 路径 | 内容 |
|---|---|
| `logs/query_trace.jsonl` | 每次问答的链路追踪（query_id 可回溯） |
| `vector_index/milvus_manifest.json` | Milvus 索引凭证（语料指纹） |
| `vector_index/image_caption_cache.json` | 图片 caption 缓存 |
| `.artifacts/eval/` | 评测报告（retrieval_eval_*.json / judge_eval_*.json） |
| `docs/` | 全部文档（教程/精讲/面试指南/本手册） |
