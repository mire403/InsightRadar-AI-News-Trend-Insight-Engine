"""
InsightRadar — 向量库抽象：FAISS / Chroma 等本地存储。
支持新文档 embedding 写入与相似检索，便于聚类与复跑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.schemas import NormalizedDocument


class VectorStoreInterface(ABC):
    """向量库抽象接口。"""

    @abstractmethod
    def add(self, doc_ids: list[str], embeddings: np.ndarray, metadata: Optional[list[dict]] = None) -> None:
        """批量写入 (doc_id, embedding)。"""
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """相似检索，返回 [(doc_id, score), ...]。"""
        ...

    @abstractmethod
    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """按 doc_id 取向量，若无则 None。"""
        ...

    @abstractmethod
    def list_ids(self) -> list[str]:
        """所有已存 doc_id。"""
        ...

    @abstractmethod
    def save(self, path: Optional[str | Path] = None) -> None:
        """持久化到路径。"""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """从路径加载。"""
        ...


class FAISSVectorStore(VectorStoreInterface):
    """FAISS 本地向量库。"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._index = None
        self._id_list: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        import faiss
        self._index = faiss.IndexFlatIP(self.dimension)  # 内积，假设向量已归一化
        self._id_list = []

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return x.astype(np.float32) / norms

    def add(self, doc_ids: list[str], embeddings: np.ndarray, metadata: Optional[list[dict]] = None) -> None:
        if not doc_ids or embeddings.size == 0:
            return
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("doc_ids length must match embeddings row count")
        if embeddings.shape[1] != self.dimension:
            self.dimension = embeddings.shape[1]
            self._build_index()
            if self._id_list:
                raise ValueError("dimension change requires new store")
        emb = self._normalize(embeddings.astype(np.float32))
        self._index.add(emb)
        self._id_list.extend(doc_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q = self._normalize(q)
        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
        out = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            out.append((self._id_list[idx], float(scores[0][i])))
        return out

    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        if doc_id not in self._id_list:
            return None
        idx = self._id_list.index(doc_id)
        # FAISS IndexFlatIP 不直接支持按索引取向量，需外部存一份；这里简化：不单独存向量则返回 None，调用方用 embedder 再算
        return None

    def list_ids(self) -> list[str]:
        return list(self._id_list)

    def save(self, path: Optional[str | Path] = None) -> None:
        path = path or "data/faiss_index"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import faiss
        faiss.write_index(self._index, f"{path}.index")
        with open(f"{path}.ids", "w", encoding="utf-8") as f:
            f.write("\n".join(self._id_list))

    def load(self, path: str | Path) -> None:
        path = str(path)
        import faiss
        self._index = faiss.read_index(f"{path}.index")
        with open(f"{path}.ids", encoding="utf-8") as f:
            self._id_list = [line.strip() for line in f if line.strip()]
        self.dimension = self._index.d


_default_store: Optional[VectorStoreInterface] = None


def get_vector_store(dimension: int = 384) -> VectorStoreInterface:
    global _default_store
    if _default_store is None:
        _default_store = FAISSVectorStore(dimension=dimension)
    return _default_store


def set_vector_store(store: VectorStoreInterface) -> None:
    global _default_store
    _default_store = store
