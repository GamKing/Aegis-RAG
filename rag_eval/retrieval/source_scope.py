"""Resolve explicit request scope before generation; never read evaluation labels.

Production callers can provide requested_sources directly. Text markers support
【来源 id】 / [source: id]; 【模拟政策 id】 is the synthetic-fixture equivalent.
Chunk metadata scope_id identifies a policy/version spanning multiple documents.
doc_id is used as a fallback identifier, not as proof of policy-version ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .base import SearchResult


_MARKER = re.compile(r"【(?:模拟政策|来源)\s+([^】]+)】|\[source:\s*([^\]]+)\]", re.I)


def requested_sources(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((a or b).strip() for a, b in _MARKER.findall(query)))


class SourceScopeError(ValueError):
    """An explicit scope is missing, or several policy versions need clarification."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ContextSelection:
    context: str
    selected_sources: tuple[str, ...]
    discarded_sources: tuple[str, ...]


def select_context(query: str, results: Sequence[SearchResult],
                   sources: Sequence[str] | None = None) -> ContextSelection:
    requested = tuple(dict.fromkeys(sources if sources is not None else requested_sources(query)))
    if any(not source.strip() for source in requested):
        raise SourceScopeError("INVALID_SOURCE_SCOPE", "来源标识不能为空")
    identified = []
    declared = set()
    for result in results:
        meta = result.chunk.metadata or {}
        markers = requested_sources(result.chunk.content)
        if len(markers) > 1:
            raise SourceScopeError("MIXED_SOURCE_CHUNK", "文档块包含多个来源版本，需先分块")
        scope = meta.get("scope_id") or (markers[0] if markers else None)
        if meta.get("scope_id") and markers and meta["scope_id"] != markers[0]:
            raise SourceScopeError("SOURCE_METADATA_CONFLICT", "来源元数据与正文标识不一致")
        if scope:
            declared.add(scope)
        source = scope or meta.get("doc_id") or result.chunk.chunk_id
        identified.append((source, result))

    if not requested and len(declared) > 1:
        raise SourceScopeError("AMBIGUOUS_SOURCE", "检索结果涉及多个政策版本，请指定地区、年份或来源")
    available = {source for source, _ in identified}
    if requested and not set(requested) <= available:
        missing = sorted(set(requested) - available)
        raise SourceScopeError("REQUESTED_SOURCE_NOT_FOUND", f"未召回指定来源: {', '.join(missing)}")
    selected = [(source, hit) for source, hit in identified if not requested or source in requested]
    texts = []
    for source, hit in selected:
        content = hit.chunk.content
        # 元数据来源也要进入可见证据，避免比较多个来源时答案丢失归属。
        if (requested or source in declared) and not requested_sources(content):
            content = f"【来源 {source}】{content}"
        texts.append(content)
    return ContextSelection(
        context="\n\n".join(texts),
        selected_sources=tuple(dict.fromkeys(source for source, _ in selected)),
        discarded_sources=tuple(dict.fromkeys(source for source, _ in identified
                                             if requested and source not in requested)),
    )
