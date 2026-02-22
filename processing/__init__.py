# InsightRadar — processing: embedding, clustering, sentiment, trend scoring

from .embedding import EmbeddingInterface, get_embedder
from .topic_clustering import cluster_documents, rename_topic_with_llm, build_topics
from .sentiment import analyze_sentiment, aggregate_sentiment
from .trend_scoring import compute_trend_score

__all__ = [
    "EmbeddingInterface",
    "get_embedder",
    "cluster_documents",
    "rename_topic_with_llm",
    "build_topics",
    "analyze_sentiment",
    "aggregate_sentiment",
    "compute_trend_score",
]
