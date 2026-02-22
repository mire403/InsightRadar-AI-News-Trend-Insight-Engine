"""
InsightRadar — 情感与立场分析：多维度（neutral/concern/optimism/anger/hype）+ 极化。
回答：这是「被冷静讨论的趋势」还是「被情绪推着走的趋势」？
"""

from __future__ import annotations

from typing import Optional

from ..core.schemas import NormalizedDocument, SentimentResult, SentimentDimension
from ..core.llm_client import get_llm_client


DIMENSION_NAMES = [
    "neutral",
    "concern",
    "optimism",
    "anger",
    "hype",
]


def analyze_sentiment(doc: NormalizedDocument) -> SentimentResult:
    """
    单文档情绪分析：返回各维度强度 0–1 及可选的 polarization。
    """
    text = ((doc.title or "") + "\n" + (doc.raw_text or ""))[:4000].strip()
    if not text:
        return SentimentResult(
            dimensions={d: 0.0 for d in DIMENSION_NAMES},
            polarization=0.0,
            label="neutral",
        )

    prompt = f"""分析下面这段文本的情绪倾向，不要简单判正负面。
维度：neutral（中性）, concern（担忧）, optimism（乐观）, anger（愤怒）, hype（炒作/狂热）。
每个维度给 0–1 的强度；且判断是否存在明显立场分裂（polarization，0–1）。
若可概括一句（如「被冷静讨论」或「被情绪推着走」）也可写 label。

文本：
{text}

请用 JSON 返回：
{{"neutral": 0.0–1.0, "concern": 0.0–1.0, "optimism": 0.0–1.0, "anger": 0.0–1.0, "hype": 0.0–1.0, "polarization": 0.0–1.0, "label": "可选一句话"}}"""

    try:
        llm = get_llm_client()
        out = llm.complete_json(prompt, max_tokens=256)
        dims = {d: float(out.get(d, 0) or 0) for d in DIMENSION_NAMES}
        pol = float(out.get("polarization", 0) or 0)
        label = out.get("label") or None
        return SentimentResult(dimensions=dims, polarization=pol, label=label)
    except Exception:
        return SentimentResult(
            dimensions={d: 0.2 for d in DIMENSION_NAMES},
            polarization=0.0,
            label="neutral",
        )


def aggregate_sentiment(
    results: list[SentimentResult],
    weights: Optional[list[float]] = None,
) -> SentimentResult:
    """聚合多文档情绪：加权平均各维度，polarization 取最大或平均。"""
    if not results:
        return SentimentResult(
            dimensions={d: 0.0 for d in DIMENSION_NAMES},
            polarization=0.0,
            label="neutral",
        )
    n = len(results)
    w = weights if weights and len(weights) == n else [1.0] * n
    total_w = sum(w)
    if total_w <= 0:
        total_w = 1.0
    dims = {d: 0.0 for d in DIMENSION_NAMES}
    for r, ww in zip(results, w):
        for d in DIMENSION_NAMES:
            dims[d] += (r.dimensions.get(d, 0) or 0) * ww
    for d in DIMENSION_NAMES:
        dims[d] = round(dims[d] / total_w, 4)
    pol = max((r.polarization or 0) for r in results)
    labels = [r.label for r in results if r.label]
    label = labels[0] if labels else None
    return SentimentResult(dimensions=dims, polarization=pol, label=label)


def sentiment_summary_label(aggregated: SentimentResult) -> str:
    """
    生成一句总结：是「被冷静讨论的趋势」还是「被情绪推着走的趋势」？
    """
    if aggregated.label:
        return aggregated.label
    dims = aggregated.dimensions
    neutral = dims.get("neutral", 0) or 0
    hype = dims.get("hype", 0) or 0
    anger = dims.get("anger", 0) or 0
    pol = aggregated.polarization or 0
    if pol > 0.5:
        return "存在明显立场分裂，讨论极化。"
    if neutral > 0.5 and hype < 0.3 and anger < 0.3:
        return "被相对冷静讨论的趋势。"
    if hype > 0.5 or anger > 0.5:
        return "被情绪推着走的趋势，存在炒作或愤怒驱动。"
    return "讨论情绪混合，无明显单一驱动。"
