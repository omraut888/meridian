"""Chunker tests on real corpus documents with the real Voyage tokenizer (local, no API key)."""

from functools import cache
from pathlib import Path

import pytest

from meridian.config import ChunkingSettings, VoyageSettings
from meridian.embeddings import TokenCounter, voyage_token_counter
from meridian.ingestion import StructuralChunker, embedding_text, load_documents
from meridian.models import Document

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus" / "documents.jsonl"


@cache
def _counter() -> TokenCounter:
    return voyage_token_counter(VoyageSettings().model)


@cache
def _documents() -> tuple[Document, ...]:
    return tuple(load_documents(CORPUS))


def _wikipedia() -> list[Document]:
    return [d for d in _documents() if d.metadata["source"] == "wikipedia"]


@pytest.fixture(scope="module")
def small_chunker() -> StructuralChunker:
    # A small window forces multi-chunk documents so packing and overlap are exercised.
    return StructuralChunker(_counter(), ChunkingSettings(max_tokens=128, overlap_tokens=32))


def test_chunks_are_exact_spans_within_budget(small_chunker: StructuralChunker) -> None:
    for doc in _wikipedia()[:12]:
        chunks = small_chunker.split(doc)
        assert chunks, doc.doc_id
        for chunk in chunks:
            assert chunk.text == doc.text[chunk.char_span[0] : chunk.char_span[1]]
            assert chunk.token_count <= 128
            # Re-tokenizing the joined span may merge tokens across unit
            # boundaries, but must stay close to the packed estimate.
            assert _counter()([chunk.text])[0] <= 128 * 1.05


def test_every_paragraph_is_covered(small_chunker: StructuralChunker) -> None:
    for doc in _wikipedia()[:12]:
        spans = [c.char_span for c in small_chunker.split(doc)]
        covered = set()
        for start, end in spans:
            covered.update(range(start, end))
        offset = 0
        for line in doc.text.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("=="):
                start = doc.text.index(stripped, offset)
                # Chunks are trimmed spans, so whitespace between sentences may fall between chunks.
                content = {i for i in range(start, start + len(stripped)) if not doc.text[i].isspace()}
                assert content <= covered, (doc.doc_id, stripped[:60])
            offset += len(line) + 1


def test_consecutive_chunks_in_a_section_overlap(small_chunker: StructuralChunker) -> None:
    overlaps = 0
    for doc in _wikipedia():
        chunks = small_chunker.split(doc)
        for a, b in zip(chunks, chunks[1:], strict=False):
            assert b.char_span[0] >= a.char_span[0]
            if b.char_span[0] < a.char_span[1]:
                assert a.metadata.get("section") == b.metadata.get("section")
                overlaps += 1
    assert overlaps > 0


def test_section_metadata_and_contextual_header() -> None:
    chunker = StructuralChunker(_counter(), ChunkingSettings())
    doc = next(d for d in _wikipedia() if "\n== " in d.text)
    chunks = chunker.split(doc)

    sectioned = [c for c in chunks if "section" in c.metadata]
    assert sectioned, "expected at least one chunk under a section heading"
    assert embedding_text(sectioned[0]).startswith(f"{doc.title} — {sectioned[0].metadata['section']}\n\n")
    assert all(c.metadata["topic"] == doc.metadata["topic"] for c in chunks)


def test_abstract_fits_one_default_chunk() -> None:
    chunker = StructuralChunker(_counter(), ChunkingSettings())
    abstracts = [d for d in _documents() if d.metadata["source"] == "arxiv"][:10]
    assert all(len(chunker.split(d)) == 1 for d in abstracts)


def test_chunk_ids_are_deterministic_and_version_sensitive() -> None:
    chunker = StructuralChunker(_counter(), ChunkingSettings())
    doc = _wikipedia()[0]
    first, second = chunker.split(doc), chunker.split(doc)
    edited = Document(doc.doc_id, doc.title, doc.text + "\n\nAn appended paragraph.", doc.source_uri, doc.metadata)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id != chunker.split(edited)[0].chunk_id


def test_overlap_must_be_smaller_than_window() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ChunkingSettings(max_tokens=64, overlap_tokens=64)
