<div align="center">

# InsightRadar — AI News & Trend Insight Engine

</div>

<p align="center">
  <strong>🛰️ 智能洞察雷达</strong><br/>
  <em>从信息洪水中提炼趋势结构，而不是摘要新闻</em>
</p>

---

## 📖 项目简介

**InsightRadar（智能洞察雷达）** 是一套面向**研究者、产品、投资与创作者**的**趋势洞察系统**。  
> 💡 **产品哲学**：我们不是在回答「发生了什么」，而是在回答「什么正在形成共识，什么正在失控，什么值得提前注意」。它不回答「发生了什么」，而是回答：**「什么正在形成共识，什么正在失控，什么值得提前注意」**。

🔬 系统通过**多源信息摄取 → 语义主题聚合 → 趋势强度建模 → 情感与立场分析 → 时间线生成 → 洞察级输出** 的完整链路，把 RSS、网页、本地文档等异构数据统一成**可回溯、可解释的洞察**，而不是简单的资讯流或热点榜。

🎯 设计目标：

- **跨来源 → 主题级趋势**：不同渠道的信息被归一为文档后，按语义聚类成「主题事件」，每个主题都有人类可读的标题与摘要。
- **趋势形成 / 升温 / 衰减**：通过讨论量变化、加速度、情绪拐点、来源扩散度等指标，量化趋势强度（0–100 的 TrendScore）。
- **情绪 + 因果线索**：多维度情绪（中性 / 担忧 / 乐观 / 愤怒 / 炒作）与立场极化检测，区分「被冷静讨论的趋势」与「被情绪推着走的趋势」。
- **洞察而非资讯**：每个趋势输出一句洞察标题、强度解释、风险/机会提示、关键引用与可能下一步（概率性推断，非预测、非投资建议）。

---

## 🚫 我们不做的事

| 不做 | 说明 |
|------|------|
| ❌ 简单新闻摘要 | 不做「把长文缩成一段」的摘要器 |
| ❌ 热点榜单复刻 | 不做纯关键词/热度排行 |
| ❌ 关键词统计 | 不做词频、TF-IDF 式统计 |
| ❌ 生成假新闻 | 不编造内容，引用可回溯 |
| ❌ 预测具体金融价格 | 不做价格预测 |
| ❌ 提供“投资建议” | 仅作研究/创作参考，不构成投资建议 |

---

## ✨ 核心功能一览

| 模块 | 功能 | 说明 |
|------|------|------|
| 📥 **Ingestion** | 多源信息摄取 | RSS、网页抓取、本地文本，统一为 `NormalizedDocument` |
| 🧠 **Topic Modeling** | 语义主题聚合 | Embedding + 聚类 + LLM 重命名，输出「主题事件」 |
| 📈 **Trend Scoring** | 趋势强度建模 | 讨论量、加速度、情绪变化、来源扩散 → TrendScore 0–100 |
| 😶 **Sentiment** | 情感与立场分析 | neutral/concern/optimism/anger/hype + 极化检测 |
| 📅 **Timeline** | 趋势时间线 | 起点事件、关键转折、情绪拐点、最新进展 |
| 🔮 **Forecasting** | 可能下一步 | 概率性推断，非预测、非投资建议 |
| 📤 **Insight Output** | 洞察级输出 | 一句标题、强度解释、风险/机会、关键引用、Markdown/JSON/API |

---

## 🏗️ 项目结构

```
insight_radar/
├── 📄 README.md
├── 📄 requirements.txt
├── 📁 config/
│   └── sources.yaml          # 多源配置（RSS / 网页 / 本地 / 处理参数）
├── 📁 core/
│   ├── schemas.py             # NormalizedDocument, Topic, TrendScore, InsightOutput 等
│   └── llm_client.py          # 所有 LLM 调用集中封装
├── 📁 ingestion/
│   ├── rss_loader.py          # RSS 摄取
│   ├── web_scraper.py         # 网页 HTML 抓取 + 解析
│   └── local_loader.py        # 本地文本文件（会议纪要 / 内部文档）
├── 📁 processing/
│   ├── embedding.py           # Embedding 抽象接口（可换模型）
│   ├── topic_clustering.py   # 聚类 + LLM 语义重命名
│   ├── sentiment.py          # 多维度情绪 + 极化
│   └── trend_scoring.py      # TrendScore 计算
├── 📁 insight/
│   ├── timeline_builder.py    # 时间线节点
│   ├── insight_generator.py  # 洞察级输出生成
│   └── forecasting.py       # 可能下一步（概率性）
├── 📁 storage/
│   ├── vector_store.py       # FAISS 向量库
│   └── database.py           # SQLite 文档 / Topic / Insight 存储
├── 📁 api/
│   └── app.py                # FastAPI：/ingest, /topics, /insights, /run-pipeline
└── 📁 scheduler/
    └── daily_run.py          # 每日抓取 + 聚类 + 评分 + 生成 Insight
```

---

## 📚 代码解析与设计要点

### 1️⃣ 统一文档模型：`NormalizedDocument`

所有输入源（RSS、网页、本地）都先转成同一种结构，方便后续 embedding、聚类、情绪分析统一处理。

```python
# core/schemas.py
class NormalizedDocument(BaseModel):
    doc_id: str           # 唯一 ID，如 source + timestamp 的 hash
    source: str           # 来源标识，如 rss:tech_crunch, local:meeting.md
    timestamp: datetime  # 文档时间
    author: Optional[str] # 作者或来源账号
    raw_text: str        # 原始正文
    title: Optional[str] # 标题
    url: Optional[str]   # 原文链接
    meta: dict[str, Any] # 扩展元数据（如 category）
```

**设计要点**：`source` 和 `timestamp` 既用于去重与溯源，也用于趋势计算里的「时间序列」和「来源扩散度」；`meta` 可挂载分类、标签等，便于扩展。

---

### 2️⃣ LLM 调用集中封装：`LLMClient`

所有大模型调用都走 `core/llm_client.py`，便于换模型、统一限流与缓存。

```python
# core/llm_client.py
class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, prompt, system=None, max_tokens=1024, temperature=0.3, **kwargs) -> str:
        # 单轮文本补全，返回模型输出字符串

    def complete_json(self, prompt, system=None, max_tokens=1024, **kwargs) -> dict:
        # 要求模型返回合法 JSON，解析为 dict（支持 ```json ... ``` 包裹）
```

**设计要点**：环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` 可指向任意 OpenAI 兼容 API；`complete_json` 用于主题重命名、情绪分析、可能下一步等需要结构化输出的场景。

---

### 3️⃣ RSS 摄取：统一为 `NormalizedDocument`

RSS 条目解析时间、生成稳定 `doc_id`，并组装成统一文档。

```python
# ingestion/rss_loader.py
def load_rss_feed(url: str, source_name: str, category: str = "general") -> list[NormalizedDocument]:
    feed = feedparser.parse(url)
    for entry in feed.entries or []:
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        raw_text = f"{title}\n\n{summary}".strip()
        # ...
        doc_list.append(
            NormalizedDocument(
                doc_id=_doc_id(source_name, entry),  # 基于 link + published 的 hash
                source=f"rss:{source_name}",
                timestamp=_parse_date(entry),        # published_parsed / updated_parsed
                author=getattr(entry, "author", None),
                raw_text=raw_text[:50000],
                title=title[:2000] if title else None,
                url=getattr(entry, "link", None),
                meta={"category": category},
            )
        )
    return doc_list
```

**设计要点**：`doc_id` 用 `source + link + published` 的 hash，保证同一篇文章多次抓取不会重复入库；`raw_text` 截断到 50k 字符，避免单文档过大影响后续 embedding/LLM。

---

### 4️⃣ 语义主题聚合：Embedding + 聚类 + LLM 重命名

不用传统 LDA，而是用 **embedding + K-Means 聚类**，再对每个簇用 **LLM 生成人能读懂的标题与摘要**。

```python
# processing/topic_clustering.py
def cluster_documents(documents, embeddings=None, min_cluster_size=3, n_clusters=None):
    embedder = get_embedder()
    if embeddings is None:
        embeddings = embedder.embed_documents(documents)
    k = n_clusters or max(2, min(20, n // max(1, min_cluster_size)))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    # 按 label 分组，计算每簇中心点，过滤掉过小簇
    # 返回 [(doc_list, centroid_idx), ...]

def rename_topic_with_llm(documents, centroid_idx=0) -> tuple[str, str]:
    # 取簇内文档片段（含中心文档），拼成 prompt，要求 LLM 返回 JSON：
    # {"topic_title": "人能读懂的标题", "summary": "2–4 句摘要"}
```

**设计要点**：聚类在**向量空间**里做，语义相近的文档会聚在一起；`topic_title` 和 `summary` 由 LLM 根据文档内容生成，保证每个 Topic 都是「主题事件」级别的描述，而不是关键词堆砌。

---

### 5️⃣ 趋势强度：讨论量、加速度、情绪变化、来源扩散

`TrendScore` 由四个子指标综合得到 0–100 分，并附带文字解释。

```python
# processing/trend_scoring.py
def _volume_series(docs, bin_days=1) -> list[float]:
    """按时间桶统计文档数 → 讨论量时间序列。"""
    base = min(d.timestamp for d in docs)
    buckets = defaultdict(int)
    for d in docs:
        delta = (d.timestamp - base).total_seconds() / (86400 * bin_days)
        buckets[int(delta)] += 1
    return [float(buckets.get(i, 0)) for i in range(max(buckets.keys()) + 1)]

def _volume_change(series) -> float:
    """后半段均值 / 前半段均值，>1 表示讨论在升温。"""
def _acceleration(series) -> float:
    """后半段斜率 − 前半段斜率，反映增长是否在加速。"""
def _sentiment_shift(docs) -> float:
    """按时间排序，比较前后半段「强情绪」维度之和的变化。"""
def _source_diffusion(docs) -> float:
    """来源种类数 / sqrt(文档数)，归一化到 0–1，表示是否从小圈层扩散。"""

# 最终 score = 0.3*volume_norm + 0.25*accel_norm + 0.25*sentiment_norm + 0.2*diffusion_norm
```

**设计要点**：讨论量看「是否在变多」，加速度看「是否在加速变多」，情绪变化看「是否从中性走向强烈」，来源扩散看「是否从单一渠道扩散到多源」，综合后得到一个可解释的趋势强度。

---

### 6️⃣ 情感分析：多维度 + 极化（非简单正负面）

情绪维度包括 **neutral / concern / optimism / anger / hype**，并估计**立场极化**程度。

```python
# processing/sentiment.py
DIMENSION_NAMES = ["neutral", "concern", "optimism", "anger", "hype"]

def analyze_sentiment(doc: NormalizedDocument) -> SentimentResult:
    prompt = f"""分析下面这段文本的情绪倾向，不要简单判正负面。
    维度：neutral, concern, optimism, anger, hype。每个维度 0–1。
    判断是否存在明显立场分裂（polarization，0–1）。
    若可概括一句（如「被冷静讨论」或「被情绪推着走」）也可写 label。
    请用 JSON 返回：{{"neutral": ..., "polarization": ..., "label": "..."}}"""
    out = llm.complete_json(prompt, max_tokens=256)
    return SentimentResult(dimensions=dims, polarization=pol, label=label)

def sentiment_summary_label(aggregated) -> str:
    # 根据聚合后的维度和 polarization，生成一句总结，例如：
    # 「被相对冷静讨论的趋势」「被情绪推着走的趋势，存在炒作或愤怒驱动」
```

**设计要点**：不做「正/负/中性」三分类，而是多维度强度 + 极化，能更好区分「理性讨论」与「情绪驱动」；`label` 可直接用于洞察文案中的情绪总结。

---

### 7️⃣ 洞察级输出：一句标题 + 强度解释 + 风险/机会 + 引用

每个 Topic 最终生成一个 `InsightOutput`，包含标题、强度解释、风险/机会、关键引用、时间线、情绪总结与可能下一步。

```python
# insight/insight_generator.py
def generate_insight(topic, trend_score=None, timeline=None, sentiment_agg=None, next_steps=None) -> InsightOutput:
    if timeline is None:
        timeline = build_timeline(topic)
    if sentiment_agg is None and topic.documents:
        sentiment_agg = aggregate_sentiment([analyze_sentiment(d) for d in topic.documents])
    if next_steps is None:
        next_steps = suggest_next_steps(topic, trend_score, sentiment_agg)

    strength_explanation = trend_score.explanation if trend_score else f"主题共 {len(topic.documents)} 篇文档，时间跨度 {topic.time_span}。"
    risk_opportunity = _risk_opportunity_text(topic, trend_score, sentiment_agg)  # 根据分数与情绪生成提示

    return InsightOutput(
        insight_id=_insight_id(topic.topic_id, datetime.utcnow()),
        topic_id=topic.topic_id,
        title=topic.topic_title,
        strength_explanation=strength_explanation,
        risk_opportunity=risk_opportunity,
        key_citations=_key_citations(topic),  # 从 topic.documents 抽 5 条，含 doc_id, source, snippet, url
        trend_score=trend_score,
        timeline=timeline,
        sentiment_summary=sentiment_summary_label(sentiment_agg),
        next_steps=next_steps,
        generated_at=datetime.utcnow(),
    )
```

**设计要点**：`key_citations` 保留 `doc_id / source / snippet / url`，保证**可回溯**；`risk_opportunity` 由规则 + 分数/情绪生成，明确不构成投资建议；`next_steps` 来自 `forecasting.suggest_next_steps()`，表述为概率性、情景性描述。

---

## ⚙️ 配置示例：`config/sources.yaml`

多源与处理参数集中在一个 YAML 里，模块化、易扩展。

```yaml
# 多源信息摄取配置
rss:
  - name: tech_crunch
    url: https://techcrunch.com/feed/
    category: tech
  - name: arxiv_cs_ai
    url: http://rss.arxiv.org/rss/cs.AI
    category: academic

web_scrape:
  - name: hn_frontpage
    url: https://news.ycombinator.com/
    selector: "span.titleline > a"
    category: tech
    max_items: 30

local:
  base_path: ./data/local
  extensions: [".txt", ".md"]
  recursive: true

# 处理参数：仅当 TrendScore 变化超过 trend_significant_delta 时才可考虑生成新 Insight
processing:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  cluster_min_size: 3
  trend_significant_delta: 15
```

**说明**：`rss` / `web_scrape` / `local` 决定数据从哪来；`processing` 控制聚类最小簇大小、embedding 模型和「趋势显著变化」的阈值，便于你做增量洞察策略。

---

## 🚀 快速开始

### 环境要求

- **Python 3.10+**
- 建议使用虚拟环境：`python -m venv .venv` 后激活

### 安装依赖

```bash
pip install -r insight_radar/requirements.txt
```

### 配置

1. **多源配置**：编辑 `insight_radar/config/sources.yaml`，填写 RSS URL、网页选择器、本地路径等；若尚无该文件，可复制同目录下示例（如有）或按 README 中的结构自建。
2. **LLM**：设置环境变量  
   - `OPENAI_API_KEY`（必填）  
   - 可选：`OPENAI_BASE_URL`、`OPENAI_MODEL`（默认 `gpt-4o-mini`）
3. **可选**：`.env` 中配置 `INSIGHT_RADAR_CONFIG`、`INSIGHT_RADAR_DB` 等，供 scheduler 使用。

### 单次全流程运行（抓取 + 聚类 + 评分 + 生成 Insight）

在**项目根目录**（即包含 `insight_radar` 包的那一层）执行：

```bash
python -m insight_radar.scheduler.daily_run
```

可选参数：

- `--no-ingest`：跳过抓取，仅用已有 DB 跑 pipeline  
- `--no-pipeline`：仅抓取，不跑聚类与洞察  
- `--config <path>`：指定 `sources.yaml` 路径  
- `--db <path>`：指定 SQLite 路径  

### 启动 API 服务

在项目根目录执行：

```bash
uvicorn insight_radar.api.app:app --reload
```

常用接口：

- `GET /`：服务说明  
- `GET /health`：健康检查  
- `POST /ingest`：执行一次摄取（RSS / 网页 / 本地，由 query 参数控制）  
- `GET /topics`：已存储的 Topic 列表  
- `GET /insights`：已存储的 Insight 列表  
- `GET /run-pipeline`：运行完整 pipeline（从 DB 读文档 → 聚类 → 评分 → 生成 Insight）  
- `GET /insight/{insight_id}`：返回单个 Insight 的 **Markdown** 正文，便于阅读与导出  

### 🌐 API 调用示例

```bash
# 健康检查
curl http://localhost:8000/health

# 触发一次摄取（RSS + 网页，不包含本地）
curl -X POST "http://localhost:8000/ingest?rss=true&web=true&local=false"

# 运行完整 pipeline（聚类 + 评分 + 生成 Insight）
curl "http://localhost:8000/run-pipeline?min_cluster_size=3&generate_insights=true"

# 获取某条 Insight 的 Markdown（替换 {id} 为实际 insight_id）
curl http://localhost:8000/insight/{id}
```

返回的 Markdown 包含：洞察标题、趋势强度解释、风险/机会提示、情绪总结、可能下一步、关键引用列表，可直接用于报告或二次排版。

---

## 📋 示例流程（与设计对齐）

1. **每日定时**：通过 cron 或 APScheduler 调用 `python -m insight_radar.scheduler.daily_run`，执行抓取 + pipeline。  
2. **新文档**：RSS / 网页 / 本地新数据写入 DB，并在 pipeline 中做 embedding（可选写入向量库）。  
3. **更新 Topic**：对当前文档集做聚类 + LLM 重命名，得到最新 Topic 列表并落库。  
4. **重新计算 TrendScore**：对每个 Topic 计算讨论量、加速度、情绪变化、来源扩散，得到 0–100 分及解释。  
5. **生成 Insight**：对每个 Topic 生成时间线、情绪总结、可能下一步，并组装成 `InsightOutput`；可在业务层根据「趋势显著变化」（如 score 变化超过配置的 `trend_significant_delta`）再决定是否推送或高亮。  
6. **输出形式**：JSON（API 返回）、Markdown（如 `/insight/{id}`）、或自行从 DB 导出。  

---

## 🔧 技术约束与扩展点

| 约束 | 说明 |
|------|------|
| **Embedding** | 抽象接口（如 `EmbeddingInterface`），可在配置中更换模型（如 `sentence-transformers/all-MiniLM-L6-v2`）。 |
| **向量库** | 当前为 FAISS（本地）；接口可扩展为 Chroma 等。 |
| **LLM** | 所有调用经 `core/llm_client.py`，可换后端或加限流/缓存。 |
| **缓存与复跑** | 文档 / Topic / Insight 存 SQLite；中间结果可缓存、可重复运行。 |

---
---

## 👤 作者 (Author)

**Haoze Zheng**

*   🎓 **School**: Xinjiang University (XJU)
*   📧 **Email**: zhenghaoze@stu.xju.edu.cn
*   🐱 **GitHub**: [mire403](https://github.com/mire403)

---

<p align="center">
  <strong>🛰️ InsightRadar</strong> — 从信息洪水中提炼趋势结构，面向研究者 / 产品 / 投资 / 创作者的洞察引擎。<br/>
  <em>不是在回答「发生了什么」，而是在回答「什么正在形成共识，什么正在失控，什么值得提前注意」。</em>
</p>
