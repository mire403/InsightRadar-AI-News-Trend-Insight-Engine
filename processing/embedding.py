"""
InsightRadar — Embedding 抽象接口，便于换模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..core.schemas import NormalizedDocument


class EmbeddingInterface(ABC):
    """Embedding 抽象接口。"""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """单条文本 → 向量。"""
        ...

    @abstractmethod
    def embed_documents(self, documents: list[NormalizedDocument]) -> np.ndarray:
        """批量文档 → 矩阵 (N, dim)。"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度。"""
        ...


class SentenceTransformerEmbedding(EmbeddingInterface):
    """sentence-transformers 实现，本地可换模型。"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        model = self._get_model()
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_documents(self, documents: list[NormalizedDocument]) -> np.ndarray:
        if not documents:
            return np.zeros((0, self.dimension), dtype=np.float32)
        texts = []
        for d in documents:
            t = (d.title or "") + "\n" + (d.raw_text or "")[:8000]
            texts.append(t.strip() or d.raw_text[:8000])
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True)

    @property
    def dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()


_default_embedder: Optional[EmbeddingInterface] = None


def get_embedder(model_name: Optional[str] = None) -> EmbeddingInterface:
    """获取默认 embedder，可传入 model_name 覆盖配置。"""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SentenceTransformerEmbedding(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    return _default_embedder


def set_embedder(embedder: EmbeddingInterface) -> None:
    global _default_embedder
    _default_embedder = embedder
