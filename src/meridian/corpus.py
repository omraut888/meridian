"""Public-corpus builder: Wikipedia articles and arXiv abstracts on six CS subfields.

The topics are deliberately distinct but related, so the corpus has real
cluster structure along with real boundary cases (two-phase commit vs.
two-phase locking, transformers in both NLP and vision, tokenization in both
compilers and NLP).

Licensing: Wikipedia text is CC BY-SA 4.0 (attribution retained per record via
``source_uri`` and ``revision_id``). arXiv metadata, including abstracts, is
released under CC0 1.0.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import httpx
import structlog
from defusedxml import ElementTree
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from meridian.exceptions import CorpusFetchError

log = structlog.get_logger(__name__)

USER_AGENT = "Meridian-RAG/0.1 (portfolio research project; httpx)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_MIN_INTERVAL_S = 3.1  # arXiv API terms: at most one request every 3 seconds
WIKIPEDIA_CONCURRENCY = 4
# arXiv signals throttling with 406 (and its CDN may briefly cache it), so it is
# retried alongside the usual transient statuses. Other 4xx fail immediately.
_RETRYABLE_STATUS = frozenset({406, 429, 500, 502, 503, 504})
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Trailing Wikipedia sections that are link lists or bibliographies, not prose.
_SECTION_HEADING = re.compile(r"^==[^=].*==\s*$", re.MULTILINE)
_BACK_MATTER = re.compile(
    r"^==\s*(See also|References|Notes|Citations|Sources|Bibliography|Further reading|External links)"
    r"\s*==\s*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Topic:
    """A corpus topic: a primary arXiv category plus curated Wikipedia articles."""

    name: str
    arxiv_category: str
    wikipedia_titles: tuple[str, ...]


TOPICS: tuple[Topic, ...] = (
    Topic(
        "distributed_systems",
        "cs.DC",
        (
            "Distributed computing",
            "Consensus (computer science)",
            "Paxos (computer science)",
            "Raft (algorithm)",
            "Byzantine fault",
            "CAP theorem",
            "Vector clock",
            "Two-phase commit protocol",
            "Lamport timestamp",
            "Gossip protocol",
            "Leader election",
            "Eventual consistency",
        ),
    ),
    Topic(
        "cryptography",
        "cs.CR",
        (
            "Public-key cryptography",
            "RSA cryptosystem",
            "Advanced Encryption Standard",
            "Elliptic-curve cryptography",
            "Cryptographic hash function",
            "Diffie–Hellman key exchange",
            "Zero-knowledge proof",
            "Post-quantum cryptography",
            "Digital signature",
            "Merkle tree",
            "Transport Layer Security",
            "Block cipher mode of operation",
        ),
    ),
    Topic(
        "nlp",
        "cs.CL",
        (
            "Natural language processing",
            "Transformer (deep learning)",
            "Word embedding",
            "Large language model",
            "Named-entity recognition",
            "Machine translation",
            "BERT (language model)",
            "Part-of-speech tagging",
            "Attention (machine learning)",
            "Recurrent neural network",
            "Sentiment analysis",
            "Seq2seq",
        ),
    ),
    Topic(
        "computer_vision",
        "cs.CV",
        (
            "Computer vision",
            "Convolutional neural network",
            "Object detection",
            "Image segmentation",
            "Optical flow",
            "Vision transformer",
            "Scale-invariant feature transform",
            "Feature (computer vision)",
            "Residual neural network",
            "U-Net",
            "Edge detection",
            "Image registration",
        ),
    ),
    Topic(
        "database_internals",
        "cs.DB",
        (
            "B-tree",
            "Log-structured merge-tree",
            "Write-ahead logging",
            "Multiversion concurrency control",
            "Query optimization",
            "Database index",
            "ACID",
            "Two-phase locking",
            "Snapshot isolation",
            "Isolation (database systems)",
            "Hash join",
            "Database transaction",
        ),
    ),
    Topic(
        "compilers",
        "cs.PL",
        (
            "Compiler",
            "Static single-assignment form",
            "Register allocation",
            "Lexical analysis",
            "LR parser",
            "Just-in-time compilation",
            "Dead-code elimination",
            "LLVM",
            "Abstract syntax tree",
            "Intermediate representation",
            "Loop optimization",
            "Garbage collection (computer science)",
        ),
    ),
)


class CorpusBuilder:
    """Fetch the corpus from the Wikipedia and arXiv public APIs."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        arxiv_per_topic: int = 15,
        wikipedia_max_chars: int = 5_000,
        arxiv_date_range: tuple[str, str] = ("202401010000", "202412312359"),
    ) -> None:
        """Create the builder.

        Args:
            client: HTTP client (owned by the caller).
            arxiv_per_topic: Abstracts to keep per topic, filtered to papers whose
                *primary* category matches (cross-lists are excluded).
            wikipedia_max_chars: Per-article budget. The lead section is always
                kept; whole top-level sections are then added while they fit, so
                articles are shortened only at section boundaries.
            arxiv_date_range: Submission window ``(from, to)`` as ``YYYYMMDDHHMM``;
                a fixed window keeps rebuilds close to reproducible.
        """
        self._client = client
        self._arxiv_per_topic = arxiv_per_topic
        self._date_range = arxiv_date_range
        self._wikipedia_max_chars = wikipedia_max_chars
        self._arxiv_lock = asyncio.Lock()
        self._wikipedia_slots = asyncio.Semaphore(WIKIPEDIA_CONCURRENCY)

    async def build(self, topics: Sequence[Topic] = TOPICS) -> list[dict[str, Any]]:
        """Fetch every topic and return JSONL-ready records."""
        records: list[dict[str, Any]] = []
        for topic in topics:
            wiki = await asyncio.gather(*(self._wikipedia(topic, t) for t in topic.wikipedia_titles))
            arxiv = await self._arxiv(topic)
            records.extend([*wiki, *arxiv])
            log.info("corpus.topic", topic=topic.name, wikipedia=len(wiki), arxiv=len(arxiv))
        return records

    async def _get(self, url: str, params: dict[str, str | int]) -> httpx.Response:
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_transient),
            wait=wait_random_exponential(multiplier=5.0, min=5.0, max=90.0),
            stop=stop_after_attempt(6),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    response = await self._client.get(url, params=params)
                    response.raise_for_status()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise CorpusFetchError(f"GET {url} failed: {exc}") from exc
        return response

    async def _wikipedia(self, topic: Topic, title: str) -> dict[str, Any]:
        async with self._wikipedia_slots:
            return await self._wikipedia_article(topic, title)

    async def _wikipedia_article(self, topic: Topic, title: str) -> dict[str, Any]:
        response = await self._get(
            WIKIPEDIA_API,
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "extracts|info",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
            },
        )
        pages = response.json().get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or not pages[0].get("extract"):
            raise CorpusFetchError(f"Wikipedia article not found: {title!r}")
        page = pages[0]
        text = _cap_at_sections(_strip_back_matter(page["extract"]), self._wikipedia_max_chars)
        canonical = page["title"]
        return {
            "doc_id": f"wikipedia:{page['pageid']}",
            "title": canonical,
            "text": text,
            "source_uri": f"https://en.wikipedia.org/wiki/{quote(canonical.replace(' ', '_'))}",
            "metadata": {
                "source": "wikipedia",
                "topic": topic.name,
                "license": "CC BY-SA 4.0",
                "revision_id": str(page.get("lastrevid", "")),
            },
        }

    async def _arxiv(self, topic: Topic) -> list[dict[str, Any]]:
        start, end = self._date_range
        query = f"cat:{topic.arxiv_category} AND submittedDate:[{start} TO {end}]"
        params = {
            "search_query": query,
            "start": 0,
            # Over-fetch: some results are cross-lists from other primaries.
            "max_results": self._arxiv_per_topic * 3,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        async with self._arxiv_lock:  # serialize to respect the rate limit
            body = await asyncio.to_thread(_arxiv_get, f"{ARXIV_API}?{urlencode(params)}")
            await asyncio.sleep(ARXIV_MIN_INTERVAL_S)

        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            raise CorpusFetchError(f"arXiv returned malformed XML for {query!r}") from exc

        records: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ATOM):
            primary = entry.find("arxiv:primary_category", ATOM)
            if primary is None or primary.get("term") != topic.arxiv_category:
                continue
            abs_url = _text(entry, "atom:id")
            arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
            records.append(
                {
                    "doc_id": f"arxiv:{arxiv_id}",
                    "title": _squash(_text(entry, "atom:title")),
                    "text": _squash(_text(entry, "atom:summary")),
                    "source_uri": f"https://arxiv.org/abs/{arxiv_id}",
                    "metadata": {
                        "source": "arxiv",
                        "topic": topic.name,
                        "license": "CC0 1.0 (arXiv metadata)",
                        "published": _text(entry, "atom:published"),
                        "primary_category": topic.arxiv_category,
                    },
                }
            )
            if len(records) == self._arxiv_per_topic:
                break
        if len(records) < self._arxiv_per_topic:
            log.warning(
                "corpus.arxiv.short", topic=topic.name, wanted=self._arxiv_per_topic, got=len(records)
            )
        return records


async def fetch_corpus(
    out: Path,
    *,
    arxiv_per_topic: int = 20,
    wikipedia_max_chars: int = 8_000,
    mirrors_per_topic: int = 2,
) -> int:
    """Fetch the full corpus and write it as JSONL.

    Args:
        out: Destination ``.jsonl`` path.
        arxiv_per_topic: arXiv abstracts per topic.
        wikipedia_max_chars: Per-article character budget (see :class:`CorpusBuilder`).
        mirrors_per_topic: Near-duplicate distractors added per topic (see :func:`add_mirrors`).

    Returns:
        The number of records written.
    """
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
    ) as client:
        builder = CorpusBuilder(
            client, arxiv_per_topic=arxiv_per_topic, wikipedia_max_chars=wikipedia_max_chars
        )
        records = await builder.build()
    records.extend(add_mirrors(records, per_topic=mirrors_per_topic))

    seen: set[str] = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            if record["doc_id"] in seen:  # a paper can surface under two topics
                continue
            seen.add(record["doc_id"])
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("corpus.written", path=str(out), documents=len(seen))
    return len(seen)


def add_mirrors(
    records: Sequence[dict[str, Any]], *, per_topic: int, drop_fraction: float = 0.2, seed: int = 7
) -> list[dict[str, Any]]:
    """Build near-duplicate "mirror" copies of Wikipedia articles as redundancy distractors.

    A mirror keeps the original's title, lead, and headings but drops a seeded
    random ``drop_fraction`` of its other paragraphs, like a scraped or lightly
    edited copy. It is tagged ``duplicate_of`` its original, so evaluation
    scores it as that document: it can never earn extra relevance credit, but
    its chunks can crowd genuinely different documents out of the top k.

    Args:
        records: Fetched corpus records.
        per_topic: Wikipedia articles to mirror per topic.
        drop_fraction: Share of non-lead, non-heading paragraphs removed.
        seed: Selection and perturbation seed.

    Returns:
        The mirror records (the originals are not modified).
    """
    rng = random.Random(seed)
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["metadata"]["source"] == "wikipedia":
            by_topic.setdefault(record["metadata"]["topic"], []).append(record)
    mirrors: list[dict[str, Any]] = []
    for topic in sorted(by_topic):
        candidates = sorted(by_topic[topic], key=lambda r: r["doc_id"])
        for original in rng.sample(candidates, min(per_topic, len(candidates))):
            paragraphs = original["text"].split("\n")
            is_heading = [p.startswith("==") for p in paragraphs]  # any depth: "== A ==", "=== B ==="
            first_heading = is_heading.index(True) if any(is_heading) else len(paragraphs)
            kept = [
                p
                for i, p in enumerate(paragraphs)
                if i < first_heading or is_heading[i] or not p.strip() or rng.random() >= drop_fraction
            ]
            mirrors.append(
                {
                    "doc_id": f"mirror:{original['doc_id']}",
                    "title": original["title"],
                    "text": "\n".join(kept),
                    "source_uri": original["source_uri"],
                    "metadata": {
                        **original["metadata"],
                        "source": "mirror",
                        "duplicate_of": original["doc_id"],
                    },
                }
            )
    return mirrors


def _strip_back_matter(text: str) -> str:
    match = _BACK_MATTER.search(text)
    return (text[: match.start()] if match else text).strip()


def _cap_at_sections(text: str, max_chars: int) -> str:
    """Keep the lead, then whole top-level sections while under ``max_chars``."""
    starts = [m.start() for m in _SECTION_HEADING.finditer(text)]
    bounds = [0, *starts, len(text)]
    kept_end = bounds[1]
    for end in bounds[2:]:
        if end > max_chars:
            break
        kept_end = end
    return text[:kept_end].strip()


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RETRYABLE_STATUS


def _arxiv_get(url: str) -> bytes:
    """Blocking GET against the arXiv API using the standard library.

    arXiv's edge consistently answers httpx clients with ``406 Not Acceptable``
    while serving identical requests from urllib and curl, so this one source
    uses urllib (in a worker thread). Requests are serialized by the arXiv rate
    limit regardless, so nothing is lost by not using the async client.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    retrying = Retrying(
        retry=retry_if_exception(
            lambda e: (
                (isinstance(e, urllib.error.HTTPError) and e.code in _RETRYABLE_STATUS)
                or (isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError))
            )
        ),
        wait=wait_random_exponential(multiplier=5.0, min=5.0, max=90.0),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    try:
        for attempt in retrying:
            with attempt, urllib.request.urlopen(request, timeout=60) as response:  # fixed https URL
                body: bytes = response.read()
    except urllib.error.URLError as exc:
        raise CorpusFetchError(f"GET {url} failed: {exc}") from exc
    return body


def _text(entry: Any, path: str) -> str:
    node = entry.find(path, ATOM)
    if node is None or node.text is None:
        raise CorpusFetchError(f"arXiv entry missing {path}")
    return str(node.text)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
