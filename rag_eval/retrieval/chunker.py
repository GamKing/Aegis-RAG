"""工业级分块器：优先按自然段切分，支持重叠窗口与元数据保留。

解决"例外条款被腰斩"的关键在于切分策略：按双换行（段落）优先切分，
仅在单段超长时才回退到标点粒度，从而保证政策条件、例外条款这类
跨行逻辑保持完整。

行业插槽 1 基类：BaseChunker。通用场景直接用 RecursiveChunker；
法律/财报等特定行业继承 BaseChunker 实现行业感知解析器。
"""
from __future__ import annotations

import re
from typing import Sequence

from .base import BaseChunker, DocumentChunk


class RecursiveChunker(BaseChunker):
    """带重叠窗口的递归分块器。"""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 优先按双换行（段落）、单换行、句号分块
        self.separators = ["\n\n", "\n", "。", "；", " "]

    def split_text(self, text: str, doc_id: str = "doc") -> list[DocumentChunk]:
        chunks: list[str] = self._split_into_chunks(text)

        # 构建重叠窗口：若启用重叠，从每个 chunk 结尾截取 overlap 长度的文本
        # 作为下一个 chunk 的前缀，使跨块的政策条件与例外条款能完整出现在某一块中。
        if self.chunk_overlap > 0:
            windowed: list[str] = []
            for i, c in enumerate(chunks):
                if i > 0 and c in windowed:
                    continue
                if i > 0:
                    overlap_text = chunks[i - 1][-self.chunk_overlap :]
                    c = overlap_text + c
                windowed.append(c)
            chunks = windowed

        return [
            DocumentChunk(
                chunk_id=f"{doc_id}_c{idx}",
                content=chunk_text,
                metadata={"index": idx, "doc_id": doc_id},
            )
            for idx, chunk_text in enumerate(chunks)
        ]

    def _split_into_chunks(self, text: str) -> list[str]:
        raw_paragraphs = [p for p in text.split("\n\n") if p.strip()]

        chunks: list[str] = []
        current_chunk = ""
        for para in raw_paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(para) > self.chunk_size:
                    # 单段超长，按标点回退切分
                    chunks.extend(self._split_by_sentences(para))
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(。|；|\n)", text)
        result: list[str] = []
        buf = ""
        for part in sentences:
            buf += part
            if len(buf) >= self.chunk_size:
                result.append(buf)
                buf = ""
        if buf:
            result.append(buf)
        return result


class LegalHierarchyChunker(BaseChunker):
    """法律层级分块器：按"第X条"作为切割边界，强制保留父标题。

    法律文本具有编-章-节-条-款-项的树状层级。切块不能切断父法条与子款，
    因此以"第X条"为天然边界，并把当前所处的章节标题带到每个块的元数据中，
    供后续检索元数据过滤与溯源使用。
    """

    _ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千0-9]+条")
    _TITLE_RE = re.compile(r"^(第[一二三四五六七八九十百0-9]+[章节编]|[一二三四五六七八九十]+、)")

    def split_text(self, text: str, doc_id: str = "doc") -> list[DocumentChunk]:
        # 逐行扫描，识别章节标题与条款
        lines = text.splitlines()
        current_title = ""
        sections: list[tuple[str, str]] = []  # (title, clause_text)
        buf_lines: list[str] = []

        def flush() -> None:
            if buf_lines:
                sections.append((current_title, "\n".join(buf_lines).strip()))
                buf_lines.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            title_match = self._TITLE_RE.match(stripped)
            if title_match and not self._ARTICLE_RE.match(stripped):
                flush()
                current_title = stripped
            else:
                buf_lines.append(stripped)

        flush()

        chunks: list[DocumentChunk] = []
        for idx, (title, clause_text) in enumerate(sections):
            if not clause_text:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_l{idx}",
                    content=(title + "\n" + clause_text).strip() if title else clause_text,
                    metadata={
                        "index": idx,
                        "doc_id": doc_id,
                        "section": title,
                        "kind": "legal_clause",
                    },
                )
            )
        return chunks


class FinancialTableChunker(BaseChunker):
    """财报表格分块器：将财务报表每行转为一段独立的语义描述。

    表格是金融场景的核心载体。逐行转写能保留"2025年研发费用为1.2亿元"
    这类可检索、可审计的语义单元，避免表格被切成无意义的碎片。
    """

    def split_text(self, text: str, doc_id: str = "doc") -> list[DocumentChunk]:
        rows = [r for r in text.splitlines() if r.strip()]
        chunks: list[DocumentChunk] = []

        for idx, row in enumerate(rows):
            # 简单启发式：以逗号/制表符分隔的单元格
            cells = [c.strip() for c in re.split(r"[,，\t]", row) if c.strip()]
            if len(cells) >= 2:
                # 把行转写为语义描述："指标名，数值"
                subject = cells[0]
                rest = "，".join(cells[1:])
                content = f"{subject}为{rest}"
            else:
                content = row

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_f{idx}",
                    content=content,
                    metadata={
                        "index": idx,
                        "doc_id": doc_id,
                        "row": idx,
                        "kind": "financial_table_row",
                    },
                )
            )
        return chunks