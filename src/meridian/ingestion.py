"""Document loading, structure-aware chunking, and the indexing pipeline."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from meridian.clustering import ClusterRouter
from meridian.config import ChunkingSettings
from meridian.embeddings import DenseEmbedder, SparseEmbedder, TokenCounter
from meridian.exceptions import (
    EmbeddingError,
    IngestionError,
    MeridianError,
    VectorStoreError,
)
from meridian.models import Chunk, Document, EmbeddedChunk
from meridian.vector_store import QdrantVectorStore

log = structlog.get_logger(__name__)

_TEXT_SUFFIXES = frozenset({".md", ".txt"})
# Wikipedia plain-text extracts use "== Heading ==" (any depth); Markdown uses "#".
_HEADING = re.compile(r"^(?:(={2,6})\s*(?P<wiki>.+?)\s*\1|#{1,6}\s+(?P<md>.+?))\s*$", re.MULTILINE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PARAGRAPH = re.compile(r"\S(?:.*\S)?")  # one non-blank line, trimmed


# --------------------------------------------------------------------- loading


def load_documents(path: Path) -> Iterator[Document]:
    """Load documents from a JSONL corpus file or a directory of text files.

    JSONL records must carry ``doc_id``, ``title``, ``text`` and ``source_uri``;
    an optional ``metadata`` object of string values is propagated to chunks.
    Directories are walked recursively for ``.md`` and ``.txt`` files.

    Args:
        path: A ``.jsonl`` file or a directory.

    Yields:
        Parsed documents, in file order.

    Raises:
        IngestionError: If the path is missing or a record is malformed.
    """
    if not path.exists():
        raise IngestionError(f"corpus path does not exist: {path}")
    if path.is_file() and path.suffix == ".jsonl":
        yield from _load_jsonl(path)
    elif path.is_dir():
        for file in sorted(p for p in path.rglob("*") if p.suffix in _TEXT_SUFFIXES):
            yield _load_text_file(file, root=path)
    else:
        raise IngestionError(f"unsupported corpus path (expected .jsonl or directory): {path}")


def _load_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                yield Document(
                    doc_id=str(record["doc_id"]),
                    title=str(record["title"]),
                    text=str(record["text"]),
                    source_uri=str(record["source_uri"]),
                    metadata={str(k): str(v) for k, v in record.get("metadata", {}).items()},
                )
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                raise IngestionError(f"{path}:{line_no}: malformed corpus record") from exc


def _load_text_file(file: Path, *, root: Path) -> Document:
    text = file.read_text(encoding="utf-8")
    heading = _HEADING.search(text)
    title = (heading.group("wiki") or heading.group("md")) if heading else file.stem
    return Document(
        doc_id=f"file:{file.relative_to(root).as_posix()}",
        title=title,
        text=text,
        source_uri=file.resolve().as_uri(),
        metadata={"source": "file"},
    )


# -------------------------------------------------------------------- chunking


@dataclass(frozen=True, slots=True)
class _Unit:
    """An indivisible span of text (paragraph, sentence, or word window)."""

    start: int
    end: int
    tokens: int
    section: str | None


class StructuralChunker:
    """Split documents along their structure, then pack to a token budget.

    Text is decomposed into headings → paragraphs → sentences → word windows,
    descending a level only when a span exceeds ``max_tokens``. The resulting
    units are packed greedily into chunks, with trailing units of the previous
    chunk (up to ``overlap_tokens``) repeated at the start of the next. A section
    boundary closes the current chunk once it is at least half full, so chunks
    rarely straddle unrelated sections without producing fragments.
    """

    def __init__(self, count_tokens: TokenCounter, settings: ChunkingSettings) -> None:
        """Create the chunker.

        Args:
            count_tokens: Batch token counter for the embedding model's tokenizer.
            settings: Token budget and overlap.
        """
        self._count = count_tokens
        self._max = settings.max_tokens
        self._overlap = settings.overlap_tokens

    def split(self, document: Document) -> list[Chunk]:
        """Chunk ``document``.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order; empty if the document has no text.
        """
        units = self._units(document.text)
        if not units:
            return []
        doc_hash = document.content_hash
        chunks: list[Chunk] = []
        for index, group in enumerate(self._pack(units)):
            start, end = group[0].start, group[-1].end
            metadata = dict(document.metadata)
            if group[0].section:
                metadata["section"] = group[0].section
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(document.doc_id, doc_hash, index),
                    doc_id=document.doc_id,
                    index=index,
                    text=document.text[start:end],
                    token_count=sum(u.tokens for u in group),
                    char_span=(start, end),
                    title=document.title,
                    source_uri=document.source_uri,
                    content_hash=doc_hash,
                    metadata=metadata,
                )
            )
        return chunks

    def _units(self, text: str) -> list[_Unit]:
        spans: list[tuple[int, int, str | None]] = []
        section: str | None = None
        for match in _PARAGRAPH.finditer(text):
            heading = _HEADING.fullmatch(match.group())
            if heading:
                section = heading.group("wiki") or heading.group("md")
                continue
            spans.append((match.start(), match.end(), section))
        if not spans:
            return []
        counts = self._count([text[s:e] for s, e, _ in spans])
        units: list[_Unit] = []
        for (start, end, sec), tokens in zip(spans, counts, strict=True):
            if tokens <= self._max:
                units.append(_Unit(start, end, tokens, sec))
            else:
                units.extend(self._split_oversized(text, start, end, sec))
        return units

    def _split_oversized(self, text: str, start: int, end: int, section: str | None) -> list[_Unit]:
        """Split a paragraph into sentences, and any oversized sentence into word windows."""
        body = text[start:end]
        cuts = [0, *(m.end() for m in _SENTENCE_BOUNDARY.finditer(body)), len(body)]
        sentences = [
            (start + a, start + a + len(body[a:b].rstrip())) for a, b in zip(cuts, cuts[1:]) if body[a:b].strip()
        ]
        counts = self._count([text[s:e] for s, e in sentences])
        units: list[_Unit] = []
        for (s, e), tokens in zip(sentences, counts, strict=True):
            if tokens <= self._max:
                units.append(_Unit(s, e, tokens, section))
            else:
                units.extend(self._word_windows(text, s, e, section))
        return units

    def _word_windows(self, text: str, start: int, end: int, section: str | None) -> list[_Unit]:
        words = [(m.start() + start, m.end() + start) for m in re.finditer(r"\S+", text[start:end])]
        word_tokens = self._count([text[s:e] for s, e in words])
        units: list[_Unit] = []
        window_start, window_tokens = 0, 0
        for i, tokens in enumerate(word_tokens):
            if window_tokens + tokens > self._max and i > window_start:
                units.append(_Unit(words[window_start][0], words[i - 1][1], window_tokens, section))
                window_start, window_tokens = i, 0
            window_tokens += tokens
        units.append(_Unit(words[window_start][0], words[-1][1], window_tokens, section))
        return units

    def _pack(self, units: list[_Unit]) -> Iterator[list[_Unit]]:
        current: list[_Unit] = []
        current_tokens = 0
        for unit in units:
            section_break = bool(current) and unit.section != current[-1].section
            overflow = current_tokens + unit.tokens > self._max
            if current and (overflow or (section_break and current_tokens >= self._max // 2)):
                yield current
                current = [] if section_break else self._overlap_tail(current, unit.tokens)
                current_tokens = sum(u.tokens for u in current)
            current.append(unit)
            current_tokens += unit.tokens
        if current:
            yield current

    def _overlap_tail(self, previous: list[_Unit], incoming_tokens: int) -> list[_Unit]:
        """Trailing units of ``previous`` that fit both the overlap and the chunk budget."""
        tail: list[_Unit] = []
        tokens = 0
        budget = min(self._overlap, self._max - incoming_tokens)
        for unit in reversed(previous[1:]):  # never repeat the whole previous chunk
            if tokens + unit.tokens > budget:
                break
            tail.insert(0, unit)
            tokens += unit.tokens
        return tail


def embedding_text(chunk: Chunk) -> str:
    """Return the text actually embedded for ``chunk``.

    The document title and section are prepended as a contextual header so a
    chunk like "It was later extended to..." remains retrievable by topic.
    """
    section = chunk.metadata.get("section")
    header = f"{chunk.title} — {section}" if section else chunk.title
    return f"{header}\n\n{chunk.text}"


# -------------------------------------------------------------------- pipeline


@dataclass(slots=True)
class IngestionReport:
    """Outcome of one ingestion run."""

    documents_seen: int = 0
    documents_unchanged: int = 0
    documents_indexed: int = 0
    chunks_upserted: int = 0
    failures: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0


@dataclass(slots=True)
class _Pending:
    document: Document
    chunks: list[Chunk]


class IngestionPipeline:
    """Incrementally index documents into Qdrant.

    Unchanged documents (same content hash) are skipped. Chunks from several
    documents are embedded together so Voyage requests stay full, which matters
    far more for throughput under rate limits than request concurrency does.
    """

    def __init__(
        self,
        chunker: StructuralChunker,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        store: QdrantVectorStore,
        router: ClusterRouter | None = None,
        *,
        flush_chunks: int = 256,
    ) -> None:
        """Create the pipeline.

        Args:
            chunker: Document splitter.
            dense: Dense embedder.
            sparse: Sparse embedder.
            store: Destination vector store.
            router: If given and a cluster model exists, new chunks are assigned
                to their nearest centroid at write time.
            flush_chunks: Number of pending chunks that triggers an embed+write.
        """
        self._chunker = chunker
        self._dense = dense
        self._sparse = sparse
        self._store = store
        self._router = router
        self._flush_chunks = flush_chunks

    async def run(self, documents: Iterable[Document]) -> IngestionReport:
        """Index ``documents``, skipping those already indexed at the same version.

        Failures are isolated per document (or per flush group, for provider
        errors) and recorded in the report; the run continues.

        Args:
            documents: Documents to index.

        Returns:
            A summary of what was indexed, skipped, and failed.
        """
        started = time.perf_counter()
        report = IngestionReport()
        pending: list[_Pending] = []
        pending_chunks = 0

        for document in documents:
            report.documents_seen += 1
            try:
                if await self._store.get_content_hash(document.doc_id) == document.content_hash:
                    report.documents_unchanged += 1
                    continue
                chunks = self._chunker.split(document)
            except VectorStoreError as exc:
                self._record_failure(report, [document], exc)
                continue
            if not chunks:
                self._record_failure(report, [document], IngestionError("document has no text"))
                continue
            pending.append(_Pending(document, chunks))
            pending_chunks += len(chunks)
            if pending_chunks >= self._flush_chunks:
                await self._flush(pending, report)
                pending, pending_chunks = [], 0

        if pending:
            await self._flush(pending, report)
        report.elapsed_s = round(time.perf_counter() - started, 3)
        log.info(
            "ingestion.complete",
            seen=report.documents_seen,
            unchanged=report.documents_unchanged,
            indexed=report.documents_indexed,
            chunks=report.chunks_upserted,
            failures=len(report.failures),
            elapsed_s=report.elapsed_s,
        )
        return report

    async def _flush(self, pending: list[_Pending], report: IngestionReport) -> None:
        chunks = [c for p in pending for c in p.chunks]
        texts = [embedding_text(c) for c in chunks]
        try:
            dense, sparse = await asyncio.gather(
                self._dense.embed_documents(texts), self._sparse.embed_documents(texts)
            )
            clusters: Sequence[int | None] = (
                (await self._router.assign(dense)) if self._router else [None] * len(chunks)
            )
        except (EmbeddingError, VectorStoreError) as exc:
            self._record_failure(report, [p.document for p in pending], exc)
            return

        offset = 0
        for item in pending:
            n = len(item.chunks)
            embedded = [
                EmbeddedChunk(chunk=c, dense=dense[offset + i], sparse=sparse[offset + i], cluster_id=clusters[offset + i])
                for i, c in enumerate(item.chunks)
            ]
            offset += n
            try:
                await self._store.replace_document(item.document.doc_id, embedded)
            except VectorStoreError as exc:
                self._record_failure(report, [item.document], exc)
                continue
            report.documents_indexed += 1
            report.chunks_upserted += n
        log.info("ingestion.flush", documents=len(pending), chunks=len(chunks))

    @staticmethod
    def _record_failure(report: IngestionReport, documents: list[Document], exc: MeridianError) -> None:
        for document in documents:
            report.failures[document.doc_id] = f"{type(exc).__name__}: {exc}"
        log.error(
            "ingestion.failure",
            doc_ids=[d.doc_id for d in documents],
            error=type(exc).__name__,
            detail=str(exc),
        )

