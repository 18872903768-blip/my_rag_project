# RAGAS 评测体系与自动化流水线

本文基于当前仓库真实源码，讲解项目如何搭建"检索层确定性指标 + 生成层 RAGAS LLM-as-judge + 记忆层行为断言"的多层评测体系，以及支撑它的自动化流水线。

范围说明：

- 分析当前版本的 `eval/run_eval.py`、`eval/run_ragas_eval.py`、`eval/generate_reference_answers.py`、`eval/run_memory_eval.py`、`eval/run_full_eval.py`、`eval/compare_eval.py` 及 `config.py`、`rag_modules/memory.py`。
- 所有实验数字来自 `.artifacts/eval/` 真实报告（2026-08-23 至 2026-08-30 六轮），非虚构。
- 先讲直觉与整体框架，再逐层拆解，最后给出演算示例与事实边界。

---

## 一、先建立直觉

RAG 系统回答质量差，可能坏在检索（该找的没找到），也可能坏在生成（找到了但答偏、编造）。端到端只看一个分数，永远定位不了瓶颈。

因此本项目的评测体系分三层，每层用对工具：

```text
检索层   Golden Set + 确定性指标（Hit@k / MRR / Recall）
         → 无 LLM 参与、可高频回归、可进 CI
              ↓ 定位"排序/召回"问题

生成层   RAGAS LLM-as-judge（faithfulness / answer_relevancy /
         context_precision / context_recall / answer_correctness）
         → 回答"答案是否忠实、切题、有据"
              ↓ 定位"生成"问题

记忆层   多轮行为断言（规则判分为主 + LLM 判分为辅）
         → 回答"指代消解、记忆写入/更新/生效是否正确"
```

三层的关键分工原则：**确定性指标管能用多久能跑多勤，judge 指标管人管不了的语义判断，行为断言管策略正确性**。检索排序的变化 RAGAS 测不准（它的 context 指标是集合级的），生成编造的变化 Hit@k 测不准——所以两层缺一不可。

---

## 二、总体框架与数据流

```text
                    ┌──────────────────────────────────────┐
                    │            评测数据资产               │
                    │  golden_set.json        73条×5类intent │
                    │  golden_set_fuzzy.json  24条×4类难度   │
                    │  reference_answers.json 45条标准答案   │
                    │  multi_turn_eval.json   7组多轮对话    │
                    └──────────────────────────────────────┘
                        ↓                    ↓                  ↓
             ┌────────────────┐   ┌──────────────────┐  ┌────────────────┐
             │ run_eval.py    │   │ run_ragas_eval.py│  │ run_memory_eval│
             │ 检索层确定性指标│   │ 生成层 RAGAS 打分 │  │ 记忆行为断言    │
             │ Hit/MRR/Recall │   │ 5 指标 × 45 条    │  │ 规则+LLM 判分   │
             └───────┬────────┘   └────────┬─────────┘  └───────┬────────┘
                     │ retrieval_eval_*.json│ ragas_eval_*.json  │ memory_eval_*.json
                     └──────────┬──────────┴────────────────────┘
                                ↓
                      run_full_eval.py（编排）
                                ↓
                pipeline_report_{stamp}.md 总报告
                                ↓
                compare_eval.py（多报告并排 + 逐 query 差异归因）
```

所有产物统一 envelope：`{"summary": {...}, "records": [...]}`，UTC 时间戳命名，写入 `.artifacts/eval/`。统一格式的好处是 `compare_eval.py` 可以对任意两份报告做逐 query 对照。

---

## 三、评测数据资产怎么来的

### 3.1 精确集（golden_set.json，73 条）

覆盖 5 类 intent：detail 40 / browse 18 / keyword 5 / permission_allowed 5 / permission_denied 5。每条字段：`query`、`role`、`intent`、`expected_dishes`（可多个）、`expect_empty`（权限用）、`expected_filters`（browse 用）。

### 3.2 模糊集（golden_set_fuzzy.json，24 条）——覆盖度修正的产物

精确集实测 62% 的 query 字面包含期望菜名（BM25 单字+二元组白捡分），没有一条同义改写、错别字或间接描述——"重排无增益"的早期结论只对这个分布成立。模糊集按四类难度补齐：

| 类别 | 示例 | 考察点 |
|---|---|---|
| synonym 同义/别名 10 条 | 番茄炒蛋→西红柿炒鸡蛋、芒果西柚西米露→杨枝甘露、湘味红烧肉→湖南家常红烧肉 | 语义映射 |
| typo 错字/口语 6 条 | 红烧茄**了**、糖醋**利**脊、水煮**rou**片 | 词面容错 |
| indirect 间接描述 5 条 | "把茄子做出鳗鱼风味"→蒲烧茄子、"五花肉加酱油冰糖慢炖"→红烧肉族 | 无词面重叠 |
| confusable 易混淆 3 条 | "安徽风味的红烧肉"→徽派红烧肉（同族兄弟菜消歧） | 精确区分 |

**每条期望菜名都对着语料 321 个菜名校验过存在性**——这是构造评测集的纪律：期望不存在的东西，指标没有意义。

### 3.3 标准答案（reference_answers.json，45 条）

RAGAS 的 `context_recall` / `answer_correctness` 需要 reference（标准答案），而 golden set 只标注了期望菜名。合成流程：

```text
golden set 条目（detail/keyword，非 expect_empty）
        ↓ 按期望菜名从语料直接取 parent 文档（不经过检索链路）
金标准上下文（不受被测系统排序质量影响）
        ↓ 生成模块（DeepSeek）合成标准答案
剔除引用附录（引用是 provenance，不是供校验的论断）
        ↓
reference_answers.json（query → {reference, intent, expected_dishes}）
        ↓ 人工抽查后使用
```

设计要点：**合成不经过检索**——保证标准答案质量与被测系统无关；默认合并写入（已有 query 跳过），`--force` 全量重生成。

---

## 四、检索层：run_eval.py 的确定性指标

### 4.1 判分规则（先于跑分定义，写在代码里）

对每条 query 取回 top-k 结果，在**菜名级别**判分（`_matches` 双向子串匹配：期望菜名出现在返回菜名里、或返回菜名包含期望，都算命中）：

| 指标 | 计算 |
|---|---|
| Hit@k | 前 k 个返回中至少命中一个期望菜谱，非 0 即 1 |
| MRR | **第一个**命中结果的排名取倒数（rank1=1.0, rank2=0.5…），全批均值 |
| Recall@k | 命中的期望菜谱数 ÷ 期望菜谱总数 |

两个特殊判分规则（解释了报告里的"怪数字"）：

- **browse 类**：按过滤条件判分——返回结果全部符合期望分类/难度且有结果才算 hit，MRR/Recall 有结果即记满；
- **permission_denied 类**：hit = 没有泄漏 internal 菜谱；MRR/Recall 恒记 0。所以总表 MRR 0.9315 是被 5 条权限用例拖低的，看分组表才是干净对比。

### 4.2 指标敏感度分工

| 指标 | 对什么改动敏感 | 实际用例 |
|---|---|---|
| Hit@5 | 灵敏度低，及格线监控 | 三轮精排实验三组全部 1.000，说明"找得到"没变 |
| MRR | 排序变化 | 精排实验的主要指标（0.879/0.902/0.896 的差异全在这里） |
| Recall@5 | 召回池/候选数 | 只在期望菜谱多的 keyword 组有区分度（0.718~0.768） |
| 逐 query diff | 一切 | compare_eval.py 的真正价值：赢在哪条、输在哪条 |

### 4.3 为什么"逐 query 差异"比均分重要

24 条集合上单条 query 就是 ±0.04 的 MRR。项目里所有有效结论都来自逐条对照：

- "麻辣豆腐的正宗做法" MRR 0.33→1.0（qwen3 理解 麻辣豆腐≈麻婆豆腐）
- "水煮rou片" 1.00→0.50（云端精排把 BM25 精确命中挤下去）
- "安徽风味的红烧肉" title 注入后 1.00→0.50（同族菜名消歧的残留 case）

---

## 五、生成层：run_ragas_eval.py 详解

### 5.1 真实调用链

```text
main()
  ↓ 两级 load_dotenv（工作区 .env 优先，与 run_eval.py 同规则）
  ↓ 校验 DEEPSEEK_API_KEY
  ↓ install_ragas_compat_shims()      ← 必须在 import ragas 之前
  ↓ import ragas（此时才触发 shim 路径）
  ↓ RecipeRAGSystem(config) + initialize_system(load_generation=True)
  ↓ build_knowledge_base()
filter_golden_items(items)            ← intent∈(detail,keyword) 且非 expect_empty
sample_items(items, limit/sample, seed)  ← random.Random(seed).sample，可复现
merge_reference(items, reference_map) ← 挂标准答案，缺 reference 记数
        ↓ 逐条采集（classic 或 agent 或 both）
{user_input, response, contexts, reference, route, pipeline}
        ↓ build_judge_llm / build_embeddings / build_metrics
asyncio.run(score_all(records, metrics, max_workers))
        ↓ 逐条逐指标打分（Semaphore 并发 + 3 次重试）
summarize_scores → envelope → ragas_eval_{stamp}.json
```

### 5.2 样本采集：contexts 必须是"生成实际所见"

这是本脚本最关键的设计决策。RAGAS 的 faithfulness 校验"答案论断能否由 context 支撑"，如果喂给它的 context 和生成时实际看到的不一致，分数就是假的。

- **classic 管线**：完整复刻 `_ask_question_inner`——`query_router → query_rewrite → retrieve(带 _extract_filters_from_query + role) → get_parent_documents → 按 route 分发生成`，捕获的 `contexts = [parent.page_content ...]`，即生成 prompt 中真实拼入的父文档正文；
- **agent 管线**：直接用 `ask_agent` 返回的 `parents`（LangGraph 图内 `generate_basic_answer` 收到的同一批对象）；
- **response 统一 `strip_citations`**：答案末尾的确定性引用附录（`——\n📚 以上回答参考自：`）是 provenance 不是论断，不剔除会被 judge 当 claim 逐条校验，稀释分数。

对比早期 `run_judge.py` 的做法（ask_question 与 retrieve 分开调两次）：重写 LLM 温度 0.1 非确定性可能导致两次检索不一致，contexts 与生成上下文错位。本脚本的采集是单次链路内捕获。

### 5.3 RAGAS v0.4.3 的指标矩阵

```python
METRIC_FIELDS = {
    "faithfulness":       ("user_input", "response", "retrieved_contexts),
    "answer_relevancy":   ("user_input", "response),
    "context_precision":  ("user_input", "retrieved_contexts, "reference"),
    "context_recall":     ("user_input", "retrieved_contexts", "reference"),
    "answer_correctness": ("user_input", "response", "reference"),
}
```

- 无参考指标：faithfulness（答案论断可支撑比例）、answer_relevancy（答案反向生成提问与原 query 的嵌入相似度）；
- 需参考指标：context_precision（有用内容是否排前）、context_recall（标准答案所需信息是否被检回）、answer_correctness（与标准答案事实一致度）——**reference 缺失时这三个指标跳过该样本**，并在 summary 里标注 `missing_reference`；
- v0.4 的指标在 `ragas.metrics.collections`，调用 `await metric.ascore(**kwargs)` 返回 `MetricResult`（取 `.value`）；`ground_truths` 关键字会直接 TypeError（v0.4 已改名 `reference`）。

### 5.4 judge LLM 与 embeddings 的定制

- **judge**：`AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)` + `llm_factory(config.llm_model, provider="openai", client=client, max_tokens=8192)`。两个关键点：v0.4 已废弃 `LangchainLLMWrapper`，必须用原生异步客户端走 `llm_factory`；**max_tokens 必须显式给**——默认 1024，中文 claim 拆解 JSON 很占 token，实测 4096 仍会截断，8192 是 deepseek-chat 的输出上限。
- **embeddings**：collections 指标的 `_validate_embeddings` 要求 modern 接口（`BaseRagasEmbedding`），**拒绝 Langchain 包装类**。因此用 ragas 原生 `HuggingFaceEmbeddings(model=config.embedding_model, device=cpu)` 加载与业务同款 bge-small-zh（本地缓存已存在，先 `HF_HUB_OFFLINE=1` 离线加载，失败再允许联网，仍失败返回 None → 跳过依赖嵌入的指标）。

### 5.5 并发与重试

```python
async def score_sample(sample, metrics, semaphore):
    for name, metric in metrics.items():
        if 需要reference且缺失: continue          # 不计入失败
        try: kwargs = _metric_kwargs(...)         # 缺字段 → 记失败不崩溃
        except KeyError: scores[name]=None; errors[...]=...; continue
        for attempt in (1, 2, 3):                 # 截断偶发（约1%），重试吸收
            try:
                async with semaphore:             # 默认 max_workers=4
                    result = await metric.ascore(**kwargs)
                scores[name] = float(result.value); break
            except Exception as error:
                last_error = str(error)[:200]
                if attempt < 3: await asyncio.sleep(1.5 * attempt)
        else: scores[name]=None; errors[name]=last_error
```

两个防御性设计是第一轮全量跑出来的教训：单条采集失败不能炸掉整个 `asyncio.gather`（曾因兜底记录缺 `user_input` 字段在打分阶段 KeyError 全批崩溃，已修复为字段缺失按指标失败计）；judge 输出截断在并发下偶发且重跑即好（standalone 复测同一失败样本成功，finish=stop、最大输出 3641 tokens）。

### 5.6 summary 与产物

```json
{
  "summary": {
    "generated_at": "...", "judge_model": "deepseek-chat", "pipeline": "classic",
    "top_k": 5, "sample_size": 45, "has_reference": true, "missing_reference": 0,
    "metrics": {"faithfulness": 0.6499, "answer_relevancy": 0.5815, ...},
    "failed_counts": {"answer_correctness": 2}
  },
  "records": [{"user_input": "...", "response": "...", "contexts": [...],
               "reference": "...", "scores": {...}, "errors": {...}, "events": [...]}]
}
```

`summarize_scores(records, tuple(metrics))` 只统计本次实际运行的指标——未启用的指标（如无 reference 时的 context_recall）不应被误报为"失败"，这也是一个真实修过的 bug。

---

## 六、记忆层：run_memory_eval.py 的行为断言

### 6.1 用例结构（multi_turn_eval.json，7 组）

```json
{
  "case_id": "memory_supersede",
  "sessions": [{"session_id": "case4-s1", "turns": [
      {"query": "我不吃辣，帮我推荐两道家常菜",
       "expect_memory_written": {"type": "taste_preference", "keyword": "辣"}},
      {"query": "我最近开始能吃微辣了...",
       "expect_supersede": {"type": "taste_preference", "keyword": "辣"}}
  ]}]
}
```

断言类型与检查方式：

| 断言 | 检查方式 |
|---|---|
| expect_recall_dishes | 结果 chunks 的 dish_name 双向子串匹配 |
| expect_contextualized_query_contains | `query_contextualized` 事件的 query 字段包含关键词 |
| expect_memory_written / not_written | 回合前后 `store.list_memories(user_id)` 数量与类型对比 |
| expect_supersede | 旧记录 status=superseded 且新记录 active |
| expect_memory_recalled | `memory_recalled` 事件存在 |
| expect_memory_has_expiry | 写入记录的 expires_at 非空 |
| judge | LLM 判分 1-5，≥4 通过（如"回答是否考虑花生过敏"） |

### 6.2 runner 的隔离设计

- 强制开启被测特性（`RAG_MEMORY_ENABLED` / `RAG_QUERY_CONTEXTUALIZATION` 写入 os.environ，先于 config 加载）；
- 每次运行使用**独立临时 SQLite 库**（`tempfile.mkdtemp()` 下的 RAG_MEMORY_DB），可复现、不污染业务库，`--keep-db` 可保留检查；
- 每个case 独立 user_id（`eval-{uuid8}`），跨会话用例靠同 user_id 不同 session_id 验证"会话隔离、用户共享"的语义。

### 6.3 第一轮跑分暴露的三类问题（评测自身的迭代）

1. **评测器 bug**：`memory_recalled` 事件加在 `trace.events` 上，runner 只读 `result["events"]`——两者未合并导致 2 例误报。修复：`ask_agent` 在 `trace.finish` 后 `result["events"] = list(trace.events)`，调用方读一个字段看全量；
2. **prompt 缺陷**：supersede 靠同 (user,type,key) 匹配，但 LLM 两次给的 key 不同（"辣" vs "微辣"）导致冲突检测失效 → 抽取 prompt 增加 **canonical key 规范**（口味/饮食方式/当前约束为固定键、过敏原用名称）；
3. **用例作者错误**："红烧鲈鱼"根本不在语料里（期望菜名未校验）→ 换成真实存在的"红烧鱼头"。修复后 7/7 通过。

这三条说明：**评测框架自身也要用评测来迭代**——第一轮跑分的失败样本，一半是系统问题，一半是评测器问题，逐条归因分离是唯一办法。

---

## 七、兼容性工程：接入 RAGAS 踩过的四个坑

### 7.1 VertexAI 导入 shim

ragas 0.4.3 在模块导入期无条件执行 `from langchain_community.chat_models.vertexai import ChatVertexAI`，而 langchain-community ≥0.4 已移除该模块（项目锁定 0.4.2）→ `ModuleNotFoundError` 直接炸导入。解决：`install_ragas_compat_shims()` 在 import ragas 前向 `sys.modules` 注入占位模块并给 `langchain_community.llms` 补 `VertexAI` 属性。judge 走 OpenAI 兼容端点，永远不会触碰这些类，因此占位是安全的。该 shim 是脚本内的已知临时方案，ragas 升级或 langchain-community 变更时需复核。

### 7.2 modern embeddings 校验

collections 指标的 `_validate_embeddings` 要求 `isinstance(embeddings, BaseRagasEmbedding)`，`LangchainEmbeddingsWrapper`（已弃用）会被拒绝。实测确认 ragas 原生 `HuggingFaceEmbeddings` 的继承链是 `HuggingFaceEmbeddings → BaseRagasEmbedding → ABC`，且支持 `HF_HUB_OFFLINE=1` 从本地缓存加载——与业务同款模型、零网络依赖。

### 7.3 judge max_tokens 与偶发截断

见 5.4/5.5。要点：默认 1024 不够 → 4096 仍偶发截断 → 8192（deepseek-chat 输出上限）+ 3 次重试 → 失败率约 1.4% → 0，剩余不可恢复的截断（超长 list 答案）在 summary 的 `failed_counts` 透明标注，均值剔除。

### 7.4 结果字段与异常隔离

v0.4 打分返回 `MetricResult` 而非 float（取 `.value`）；`SingleTurnSample.ground_truths` 已改名 `reference`；旧 `evaluate()` 仍可用但带弃用警告——本项目直接用 `metric.ascore()` 稳定路径，自建 envelope，不依赖 ragas 的 Experiment 框架，保持与既有产物格式一致。

---

## 八、自动化编排与归因工具

### 8.1 run_full_eval.py（一条命令出总报告）

```text
subprocess 串行：run_eval.py（检索层）→ run_ragas_eval.py（生成层）
        ↓ 各自产出时间戳命中的报告
读取两份最新报告的 summary
        ↓
build_markdown_report(retrieval_summary, ragas_summary)
        ↓
pipeline_report_{stamp}.md（检索层表 + 生成层表 + 特性开关标注）
```

细节：`_latest()` 取同目录最新报告；`--skip-retrieval/--skip-ragas` 跳过对应阶段时回退读最新已有报告（用于只重跑一层后重新合成总报告）。子进程继承环境变量，可用 `RAG_RERANK_*` 等开关切换被测配置——与单跑 run_eval 完全一致。

### 8.2 compare_eval.py（逐 query 归因）

输入 2~N 份报告，输出：总体指标行 → 按 intent 分组表 → **逐 query 差异清单**（hit 不一致或 MRR 差 >0.3 的记录，按 (query, role) 对齐多份报告）。项目里"麻辣豆腐 0.33→1.0"这类结论全部来自它的输出。

### 8.3 run_reference_answers.py 的角色

标准答案合成是独立脚本而非评测内步骤——一次性资产、需人工审核、可 `--force` 重生成。评测脚本只读取，保证"标准答案的生产"与"评测的执行"解耦。

---

## 九、演算示例：一次 faithfulness 打分的完整数据流

以经典链路 query「西红柿豆腐汤羹怎么做」为例（真实评测记录，faithfulness 0.217，后来定位为生成编造问题）：

```text
1. 采集（classic 路径）
   route=detail → query_rewrite → retrieve(top5) → get_parent_documents（2 篇，1727 字）
   generate_step_by_step_answer(question, parents)
   → response = strip_citations(答案)     # 剔除"📚 以上回答参考自"附录
   → contexts = [parent1.page_content, parent2.page_content]
   → reference = "……"（来自 reference_answers.json）

2. 打分（faithfulness.ascore(user_input, response, retrieved_contexts)）
   ragas 内部：答案拆 claim → 逐 claim 问 judge"能否由 context 推出" → 可支撑比例
   judge = llm_factory("deepseek-chat", client=AsyncOpenAI(...), max_tokens=8192)
   → MetricResult(value=0.217)            # judge 判定约 78% 论断无据

3. 归因（人工核对该样本）
   答案写了"豆腐切 1.5cm 见方""水淀粉约 20ml""约 10 分钟"——语料原文只有
   "豆腐切块备用""加入水淀粉"，没有任何数字 → 编造而非漏检
   （context_precision 0.93 同步佐证：该找的信息都在 context 里）
```

这个样本后来成为 grounding 实验（faithfulness 0.650→0.950）的直接证据：约束加上后同一条 query 的答案是"起锅烧油，放入姜片，5 秒后倒入西红柿翻炒 30 秒【食谱 1】"——每句贴着原文。

---

## 十、关键参数与设计点

| 参数 | 默认 | 作用与调优 |
|---|---|---|
| `--top-k` | 5 | 检索窗口；golden set 历史口径 top_k=5 |
| `--limit / --sample / --seed` | 全量 | 冒烟用 `--limit 5`；抽样经 `random.Random(seed)` 可复现 |
| `--pipeline` | classic | classic / agent / both；agent 臂是记忆与上下文实验的观测对象 |
| `--max-workers` | 4 | 打分并发；DeepSeek 限流时可调低 |
| `max_tokens`（judge） | 8192 | llm_factory 默认 1024 不够；8192 为 deepseek-chat 上限 |
| `RAG_MEMORY_DB` | 临时目录 | 记忆评测强制隔离库，`--keep-db` 保留 |
| reference 阈值 | — | importance/confidence ≥ 0.6 才写入（见记忆篇写入策略） |

---

## 十一、收益 / 成本 / 局限

**收益**

- 检索层回归可进 CI（无 LLM、秒级）；生成层一条命令出报告（约 20-40 分钟 / 几元 API 费）；
- 六轮实验的全部结论有报告背书：精排默认关闭、盲段注入默认开、grounding 默认开、CM/记忆默认开的取舍全部数据化；
- 负结果也有价值：精排"无增益"的误判被模糊集推翻，过程本身是评测覆盖度重要性的实证。

**成本与局限（诚实边界）**

- judge 与生成同源（deepseek-chat 评 deepseek-chat），可能有同源偏好，分数绝对值需人工抽查校准（尚未做）；
- LLM 判分有方差：同配置两轮 faithfulness 0.6529 vs 0.6499（±0.003 噪声带）；
- 无答案 query 的优雅降级评测框架暂不支持（`expect_empty` 只用于权限）；
- ragas 版本语义漂移风险：指标定义随版本变化，跨版本对比需谨慎（项目锁定 0.4.3）。

---

## 十二、面试口述版本（约 90 秒）

> 这个项目的评测体系分三层。检索层用 Golden Set 加确定性指标——Hit@k、MRR、Recall，完全无 LLM 参与，可重复、可进 CI；生成层基于 RAGAS 做 LLM-as-judge，faithfulness、answer_relevancy、context_precision 等五个指标，judge 复用 DeepSeek，embeddings 复用业务的 bge 模型，还合成了 45 条标准答案支撑参考类指标；记忆层是七组多轮行为用例，规则断言加 LLM 判分。
>
> 数据上我先做覆盖度分析——发现原评测集 62% 是精确匹配 query，会掩盖精排的真实收益——补了同义、错字、间接描述、易混淆四类模糊集；标准答案按期望菜谱原文合成，不经过检索链路，保证与被测系统无关。
>
> 这套体系直接驱动了六轮单变量实验：精排的取舍、盲段修复（模糊集 MRR 加 3.5 个百分点）、prompt 忠实性约束（faithfulness 0.65 到 0.95）、上下文与记忆的持平验证。过程中还解决了 ragas 0.4 与 langchain-community 的导入冲突、judge 输出截断、并发偶发失败重试这些工程问题。所有报告都在 .artifacts/eval 下，任何一条简历上的数字都能翻出原始记录。

---

## 十三、源码事实边界

当前源码与产物可以确认：

- 三层评测的脚本、指标定义、判分规则、envelope 格式如上；
- 六轮实验的数字来自 `.artifacts/eval/` 真实 JSON 报告；
- ragas==0.4.3 锁定于 pyproject dev 组；shim、modern embeddings、max_tokens、重试四个兼容点均有对应代码与单测；
- 记忆评测强制特性开启 + 临时隔离库，7/7 通过的报告存在。

需要保留的边界：

- judge 与生成同源的偏差未做人工校准量化；
- `run_eval.py` 文档串声称写 markdown 报告，实际只写 JSON（文档与代码的已知不一致）；
- 无答案 query 场景评测框架不支持，属已知缺口；
- 模糊集由作者单向构造，对 BM25 天然不利、对 CE 有利——相关结论已声明该偏差，且"即便偏袒 CE 仍未赢基线"的读法更稳。
