"""
InsightRadar — 本地文本文件摄取：会议纪要、内部文档等。
输出统一为 NormalizedDocument。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.schemas import NormalizedDocument


def _doc_id(source: str, path: str, mtime: float) -> str:
    import hashlib
    raw = f"local:{source}:{path}:{mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_single_file(
    file_path: str | Path,
    source_name: str | None = None,
    category: str = "local",
) -> NormalizedDocument | None:
    """加载单个文本文件为 NormalizedDocument。"""
    path = Path(file_path)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if not raw.strip():
        return None
    stat = path.stat()
    name = source_name or path.name
    return NormalizedDocument(
        doc_id=_doc_id(name, str(path), stat.st_mtime),
        source=f"local:{name}",
        timestamp=datetime.utcfromtimestamp(stat.st_mtime),
        author=None,
        raw_text=raw[:50000],
        title=path.stem,
        url=None,
        meta={"category": category, "file_path": str(path)},
    )


def load_local_documents(
    base_path: str | Path,
    extensions: list[str] | None = None,
    recursive: bool = True,
    category: str = "local",
) -> list[NormalizedDocument]:
    """遍历目录，将匹配扩展名的文件转为 NormalizedDocument。"""
    if extensions is None:
        extensions = [".txt", ".md"]
    base = Path(base_path)
    if not base.is_dir():
        return []
    out: list[NormalizedDocument] = []
    it = base.rglob("*") if recursive else base.iterdir()
    for p in it:
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        doc = load_single_file(p, source_name=p.name, category=category)
        if doc:
            out.append(doc)
    return out
