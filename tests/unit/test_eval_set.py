"""Distractor and multi-hop construction (pure, no network)."""

import random

from meridian.corpus import add_mirrors
from meridian.evaluation import _multihop_pairs
from meridian.models import Document


def _record(doc_id: str, topic: str, text: str, source: str = "wikipedia") -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "title": doc_id,
        "text": text,
        "source_uri": f"https://example.org/{doc_id}",
        "metadata": {"source": source, "topic": topic},
    }


ARTICLE = "\n".join(
    [
        "Lead paragraph one.",
        "Lead paragraph two.",
        "== History ==",
        *[f"Body paragraph {i}." for i in range(40)],
        "=== Detail ===",
        "Last paragraph.",
    ]
)


def test_mirrors_keep_lead_and_headings_and_drop_some_body() -> None:
    records = [_record("w1", "t", ARTICLE), _record("w2", "t", ARTICLE), _record("a1", "t", "abs", "arxiv")]
    (mirror,) = add_mirrors(records, per_topic=1)

    lines = mirror["text"].split("\n")
    assert lines[:3] == ["Lead paragraph one.", "Lead paragraph two.", "== History =="]
    assert "=== Detail ===" in lines
    assert 20 < sum(line.startswith("Body") for line in lines) < 40
    assert mirror["metadata"]["source"] == "mirror"
    assert mirror["metadata"]["duplicate_of"] in {"w1", "w2"}
    assert mirror["doc_id"] == f"mirror:{mirror['metadata']['duplicate_of']}"


def test_mirrors_are_deterministic_and_skip_non_wikipedia() -> None:
    records = [_record("w1", "t", ARTICLE), _record("a1", "t", ARTICLE, "arxiv")]
    assert add_mirrors(records, per_topic=5) == add_mirrors(records, per_topic=5)
    assert [m["metadata"]["duplicate_of"] for m in add_mirrors(records, per_topic=5)] == ["w1"]


def test_multihop_pairs_are_distinct_same_topic_wikipedia_only() -> None:
    docs = [
        Document(f"{t}{i}", "", "x", "", {"source": "wikipedia", "topic": t}) for t in "ab" for i in range(4)
    ] + [
        Document("mirror:a0", "", "x", "", {"source": "mirror", "topic": "a"}),
        Document("arxiv:1", "", "x", "", {"source": "arxiv", "topic": "a"}),
    ]
    pairs = _multihop_pairs(docs, 3, random.Random(0))

    assert len(pairs) == 6
    assert len({(x.doc_id, y.doc_id) for x, y in pairs}) == 6
    for x, y in pairs:
        assert x.doc_id != y.doc_id
        assert x.metadata["topic"] == y.metadata["topic"]
        assert {x.metadata["source"], y.metadata["source"]} == {"wikipedia"}
