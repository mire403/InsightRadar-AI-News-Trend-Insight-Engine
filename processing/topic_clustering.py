"""
InsightRadar — 语义主题聚合：基于 embedding + 聚类，结合 LLM 语义重命名。
不做传统 LDA；输出人能读懂的「主题事件」。
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

import numpy as np

from ..core.schemas import NormalizedDocument, Topic
from ..core.llm_client import get_llm_client
from .embedding import get_embedder


def _topic_id(doc_ids: list[str], seed: str = "") -> str:
    raw = "|".join(sorted(doc_ids)) + seed
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _date_span(docs: list[NormalizedDocument]) -> list[str]:
    if not docs:
        return ["", ""]
    ts = [d.timestamp for d in docs]
    return [
        min(ts).strftime("%Y-%m-%d"),
        max(ts).strftime("%Y-%m-%d"),
    ]


def cluster_documents(
    documents: list[NormalizedDocument],
    embeddings: Optional[np.ndarray] = None,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
) -> list[tuple[list[NormalizedDocument], np.ndarray]]:
    """
    对文档做聚类，返回 [(doc_list, centroid_idx_in_doc_list), ...]。
    若未提供 embeddings 则现场调用 embedder。
    """
    if not documents:
        return []
    embedder = get_embedder()
    if embeddings is None:
        embeddings = embedder.embed_documents(documents)
    n = len(documents)
    if n < min_cluster_size:
        return [(documents, np.array([0]))]

    # 使用 HDBSCAN 或 K-Means；这里用简单 K-Means 便于可控簇数
    from sklearn.cluster import KMeans
    k = n_clusters or max(2, min(20, n // max(1, min_cluster_size)))
    k = min(k, n)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters: list[list[int]] = [[] for _ in range(k)]
    for i, lab in enumerate(labels):
        clusters[lab].append(i)

    out: list[tuple[list[NormalizedDocument], np.ndarray]] = []
    for indices in clusters:
        if len(indices) < min_cluster_size:
            continue
        sub_docs = [documents[i] for i in indices]
        sub_emb = embeddings[indices]
        centroid_idx = np.argmin(
            np.linalg.norm(sub_emb - sub_emb.mean(axis=0), axis=1)
        )
        out.append((sub_docs, np.array([centroid_idx]))
    if not out:
        out.append((documents, np.array([0])))
    return out


def rename_topic_with_llm(
    documents: list[NormalizedDocument],
    centroid_idx: int = 0,
) -> tuple[str, str]:
    """
    用 LLM 根据文档内容生成「人能读懂的」主题标题与摘要。
    返回 (topic_title, summary)。
    """
    if not documents:
        return "未命名主题", ""
    centroid = documents[min(centroid_idx, len(documents) - 1)]
    snippets = []
    for i, d in enumerate(documents[:15]):
        t = (d.title or "")[:200]
        body = (d.raw_text or "")[:600]
        snippets.append(f"[{i+1}] {t}\n{body}")
    text_block = "\n\n".join(snippets)

    prompt = f"""根据下面一组相关文档，提炼一个简洁的「主题事件」标题（中文或英文，不超过 20 字）和一段简短摘要（2–4 句）。

文档片段：
{text_block}

请用 JSON 返回，格式：{{"topic_title": "...", "summary": "..."}}"""

    try:
        llm = get_llm_client()
        out = llm.complete_json(prompt, max_tokens=512)
        title = (out.get("topic_title") or "未命名主题").strip()[:200]
        summary = (out.get("summary") or "").strip()[:2000]
        return title, summary
    except Exception:
        fallback = (centroid.title or centroid.raw_text[:100] or "未命名主题")[:80]
        return fallback, centroid.raw_text[:500] if centroid.raw_text else ""


def build_topics(
    documents: list[NormalizedDocument],
    embeddings: Optional[np.ndarray] = None,
    min_cluster_size: int = 3,
    use_llm_rename: bool = True,
) -> list[Topic]:
    """
    端到端：聚类 + LLM 重命名 → 返回 Topic 列表。
    每个 Topic 包含 topic_id, topic_title, summary, documents, time_span。
    """
    clusters = cluster_documents(
        documents,
        embeddings=embeddings,
        min_cluster_size=min_cluster_size,
    )
    topics: list[Topic] = []
    for sub_docs, centroid_idx_arr in clusters:
        cidx = int(centroid_idx_arr[0])
        if use_llm_rename:
            title, summary = rename_topic_with_llm(sub_docs, cidx)
        else:
            title = (sub_docs[cidx].title or sub_docs[cidx].raw_text[:80] or "未命名")[:80]
            summary = sub_docs[cidx].raw_text[:500] if sub_docs[cidx].raw_text else ""
        doc_ids = [d.doc_id for d in sub_docs]
        topic_id = _topic_id(doc_ids)
        topics.append(
            Topic(
                topic_id=topic_id,
                topic_title=title,
                summary=summary,
                documents=sub_docs,
                time_span=_date_span(sub_docs),
                doc_ids=doc_ids,
                centroid_id=sub_docs[cidx].doc_id,
            )
        )
    return topics
