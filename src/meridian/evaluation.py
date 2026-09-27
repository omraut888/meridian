"""Retrieval evaluation: query-set construction and a stage-by-stage ablation.

Queries are generated once from real corpus passages by Claude, then frozen and
committed (``data/eval/queries.jsonl``), so every evaluation run scores the same
queries. A single-hop query's relevant document is the one its passage came
from; a multi-hop query needs, and is judged against, two documents.

The ablation runs every pipeline configuration against the same index and the
same precomputed query embeddings, so differences between rows are attributable
to the retrieval stage being toggled and nothing else.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import numpy as np
import structlog
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from meridian.config import GenerationSettings
from meridian.exceptions import GenerationError, IngestionError
from meridian.ingestion import load_documents
from meridian.models import Document, ScoredChunk
from meridian.retrieval import QueryEmbedding, RetrievalOptions
from meridian.services import Services

log = structlog.get_logger(__name__)

BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 0
README_START = "<!-- eval:start -->"
README_END = "<!-- eval:end -->"


# ----------------------------------------------------------------- query set


class _GeneratedQuestions(BaseModel):
    questions: list[str] = Field(description="One question per passage, in passage order.")


QUESTION_PROMPT = """\
Below are {n} passage(s) from a document titled "{title}".

For each passage, write one question that a practitioner might type into a \
technical search engine, such that the passage answers it. Requirements:
- Answerable from that passage alone.
- Paraphrase: do not copy distinctive multi-word phrases or rare terms from the \
passage unless they are the unavoidable name of the concept being asked about.
- Self-contained: never refer to "the passage", "the paper", "this document", \
or "the authors".
- One sentence, under 25 words.

{passages}"""


class _GeneratedMultiHopQuestion(BaseModel):
    question: str = Field(description="One question that needs both passages to answer.")


MULTIHOP_PROMPT = """\
Below are two passages from two different documents.

Write one question that a practitioner might type into a technical search \
engine, such that answering it fully requires information from BOTH passages: \
for example a comparison, a contrast, or a connection between them. Requirements:
- Neither passage alone is enough to answer it.
- Paraphrase: do not copy distinctive multi-word phrases or rare terms from the \
passages unless they are the unavoidable names of the concepts being asked about.
- Self-contained: never refer to "the passages", "these documents", or "the authors".
- One sentence, under 30 words.

<passage title="{title_a}">
{passage_a}
</passage>

<passage title="{title_b}">
{passage_b}
</passage>"""


def _select_passages(document: Document, per_article: int, rng: random.Random) -> list[str]:
    if document.metadata.get("source") == "arxiv":
        return [document.text]
    paragraphs = [
        p.strip()
        for p in document.text.split("\n")
        if len(p.strip()) >= 300 and not p.strip().startswith("=")
    ]
    return rng.sample(paragraphs, min(per_article, len(paragraphs)))


def _multihop_pairs(
    documents: Sequence[Document], per_topic: int, rng: random.Random
) -> list[tuple[Document, Document]]:
    """Distinct same-topic pairs of Wikipedia articles (mirrors excluded), seeded."""
    by_topic: dict[str, list[Document]] = defaultdict(list)
    for d in documents:
        if d.metadata.get("source") == "wikipedia":
            by_topic[d.metadata.get("topic", "unknown")].append(d)
    pairs: list[tuple[Document, Document]] = []
    for topic in sorted(by_topic):
        docs = sorted(by_topic[topic], key=lambda d: d.doc_id)
        candidates = [(a, b) for i, a in enumerate(docs) for b in docs[i + 1 :]]
        pairs.extend(rng.sample(candidates, min(per_topic, len(candidates))))
    return pairs


async def build_eval_set(
    corpus_path: Path,
    out_path: Path,
    settings: GenerationSettings,
    *,
    passages_per_article: int = 2,
    multihop_per_topic: int = 5,
    concurrency: int = 8,
    seed: int = 13,
) -> int:
    """Generate single-hop and multi-hop queries from corpus passages and write them as JSONL.

    Single-hop: arXiv documents contribute their abstract; Wikipedia articles
    contribute ``passages_per_article`` randomly sampled paragraphs. Multi-hop:
    ``multihop_per_topic`` pairs of same-topic Wikipedia articles each yield one
    question that needs a passage from both; both documents are relevant.
    Near-duplicate mirrors never seed queries. All sampling is seeded.

    Args:
        corpus_path: Corpus JSONL.
        out_path: Destination queries JSONL.
        settings: Claude configuration.
        passages_per_article: Paragraphs sampled per Wikipedia article.
        multihop_per_topic: Two-document queries generated per topic.
        concurrency: Concurrent Claude requests.
        seed: Passage-sampling seed.

    Returns:
        The number of queries written.

    Raises:
        GenerationError: If a request fails, is refused, or returns the wrong count.
    """
    documents = list(load_documents(corpus_path))
    client = anthropic.AsyncAnthropic(
        api_key=settings.api_key.get_secret_value() if settings.api_key else None,
        timeout=settings.timeout_s,
    )
    slots = asyncio.Semaphore(concurrency)

    async def one(document: Document) -> list[dict[str, Any]]:
        if document.metadata.get("source") == "mirror":
            return []
        passages = _select_passages(
            document, passages_per_article, random.Random(f"{seed}:{document.doc_id}")
        )
        if not passages:
            return []
        prompt = QUESTION_PROMPT.format(
            n=len(passages),
            title=document.title,
            passages="\n\n".join(f'<passage index="{i}">\n{p}\n</passage>' for i, p in enumerate(passages)),
        )
        async with slots:
            try:
                response = await client.messages.parse(
                    model=settings.model,
                    max_tokens=16_000,
                    output_config={"effort": "low"},
                    messages=[{"role": "user", "content": prompt}],
                    output_format=_GeneratedQuestions,
                )
            except anthropic.APIError as exc:
                raise GenerationError(f"question generation failed for {document.doc_id}") from exc
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise GenerationError(f"no questions returned for {document.doc_id} ({response.stop_reason})")
        questions = response.parsed_output.questions
        if len(questions) != len(passages):
            raise GenerationError(
                f"{document.doc_id}: expected {len(passages)} questions, got {len(questions)}"
            )
        return [
            {
                "query_id": f"{document.doc_id}#{i}",
                "query": q.strip(),
                "relevant_doc_ids": [document.doc_id],
                "source": document.metadata.get("source", "unknown"),
                "topic": document.metadata.get("topic", "unknown"),
                "passage": p,
            }
            for i, (q, p) in enumerate(zip(questions, passages, strict=True))
        ]

    async def multi(first: Document, second: Document) -> list[dict[str, Any]]:
        rng = random.Random(f"{seed}:{first.doc_id}+{second.doc_id}")
        passages = [_select_passages(d, 1, rng) for d in (first, second)]
        if not all(passages):
            return []
        prompt = MULTIHOP_PROMPT.format(
            title_a=first.title,
            passage_a=passages[0][0],
            title_b=second.title,
            passage_b=passages[1][0],
        )
        async with slots:
            try:
                response = await client.messages.parse(
                    model=settings.model,
                    max_tokens=16_000,
                    output_config={"effort": "low"},
                    messages=[{"role": "user", "content": prompt}],
                    output_format=_GeneratedMultiHopQuestion,
                )
            except anthropic.APIError as exc:
                raise GenerationError(
                    f"multi-hop generation failed for {first.doc_id}+{second.doc_id}"
                ) from exc
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise GenerationError(f"no question for {first.doc_id}+{second.doc_id} ({response.stop_reason})")
        return [
            {
                "query_id": f"multihop:{first.doc_id}+{second.doc_id}",
                "query": response.parsed_output.question.strip(),
                "relevant_doc_ids": [first.doc_id, second.doc_id],
                "source": "multihop",
                "topic": first.metadata.get("topic", "unknown"),
                "passage": [passages[0][0], passages[1][0]],
            }
        ]

    pairs = _multihop_pairs(documents, multihop_per_topic, random.Random(f"{seed}:multihop"))
    try:
        batches = await asyncio.gather(*(one(d) for d in documents), *(multi(a, b) for a, b in pairs))
    finally:
        await client.close()

    records = [r for batch in batches for r in batch]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("eval.queries_written", path=str(out_path), queries=len(records), model=settings.model)
    return len(records)


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One evaluation query with its relevance judgments."""

    query_id: str
    query: str
    relevant_doc_ids: frozenset[str]
    source: str
    topic: str


def load_queries(path: Path) -> list[EvalQuery]:
    """Load a frozen query set.

    Raises:
        IngestionError: If the file is missing or malformed.
    """
    if not path.exists():
        raise IngestionError(f"query set not found: {path} (run `meridian build-eval-set`)")
    queries: list[EvalQuery] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            try:
                r = json.loads(line)
                queries.append(
                    EvalQuery(
                        r["query_id"], r["query"], frozenset(r["relevant_doc_ids"]), r["source"], r["topic"]
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                raise IngestionError(f"{path}:{line_no}: malformed query record") from exc
    return queries


# ------------------------------------------------------------------- metrics


@dataclass(frozen=True, slots=True)
class QueryScores:
    """Per-query metrics for one configuration."""

    first_relevant_rank: int | None
    recall_at_5: float
    recall_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    distinct_docs: int
    intra_list_similarity: float
    latency_ms: float


def _canonical_doc_id(chunk: ScoredChunk) -> str:
    return chunk.chunk.metadata.get("duplicate_of") or chunk.chunk.doc_id


def score_ranking(
    chunks: Sequence[ScoredChunk], relevant: frozenset[str], k: int, latency_ms: float
) -> QueryScores:
    """Score a chunk ranking against document-level relevance.

    A document counts at the rank of its first retrieved chunk; later chunks of
    the same document add no gain (they are redundant, not additional relevant
    items). Near-duplicate mirrors (``duplicate_of`` in chunk metadata) count as
    their original, so they add no credit or distinct-document diversity.

    With one relevant document, Recall@k equals hit rate and nDCG@k reduces to
    ``1 / log2(rank + 1)``. Multi-hop queries have several relevant documents;
    Recall@k is then the fraction of them retrieved.

    Args:
        chunks: Ranked chunks (at least ``k`` are considered if present).
        relevant: Relevant document IDs.
        k: Cutoff.
        latency_ms: Retrieval latency to record alongside.

    Returns:
        The query's scores.
    """
    top = list(chunks[:k])
    docs = [_canonical_doc_id(c) for c in top]
    seen: set[str] = set()
    gains: list[float] = []
    first: int | None = None
    for rank, doc in enumerate(docs, start=1):
        hit = doc in relevant and doc not in seen
        seen.add(doc)
        gains.append(1.0 if hit else 0.0)
        if hit and first is None:
            first = rank
    dcg = sum(g / np.log2(r + 1) for r, g in enumerate(gains, start=1))
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))

    def found_at(cutoff: int) -> float:
        return len(set(docs[:cutoff]) & relevant) / len(relevant)

    if len(top) >= 2:
        vectors = np.vstack([c.dense for c in top])
        sims = vectors @ vectors.T
        ils = float(sims[np.triu_indices(len(top), k=1)].mean())
    else:
        ils = 0.0
    return QueryScores(
        first_relevant_rank=first,
        recall_at_5=found_at(5),
        recall_at_k=found_at(k),
        ndcg_at_k=float(dcg / ideal) if ideal else 0.0,
        mrr_at_k=1.0 / first if first else 0.0,
        distinct_docs=len(seen),
        intra_list_similarity=ils,
        latency_ms=latency_ms,
    )


def bootstrap_ci(values: NDArray[np.float64], *, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% percentile-bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


# ------------------------------------------------------------------- ablation


@dataclass(frozen=True, slots=True)
class Configuration:
    """A named pipeline configuration under evaluation."""

    name: str
    description: str
    options: RetrievalOptions


# Routing depth evaluated as the alternative to exhaustive dense search. It is
# fixed, not read from the defaults, so routing is still measured when it is off.
ROUTING_M = 3


def ablation_configs(defaults: RetrievalOptions, k: int) -> list[Configuration]:
    """The configurations compared, from simplest to fullest.

    The row matching the shipped defaults is tagged "(default)"; if none does,
    a separate default row is appended.
    """
    lam = defaults.mmr_lambda if defaults.mmr_lambda is not None else 0.7
    base = replace(
        defaults,
        top_k=k,
        candidate_pool=max(defaults.candidate_pool, k),
        route_top_m=0,
        mmr_lambda=None,
    )
    configs = [
        Configuration("Dense only", "Voyage embeddings, cosine kNN", replace(base, use_sparse=False)),
        Configuration("BM25 only", "Sparse BM25, server-side IDF", replace(base, use_dense=False)),
        Configuration("Hybrid (RRF)", "Dense + BM25, reciprocal rank fusion", base),
        Configuration(
            "Hybrid + routing (m=1)",
            "Dense branch restricted to nearest cluster",
            replace(base, route_top_m=1),
        ),
        Configuration(
            f"Hybrid + routing (m={ROUTING_M})",
            f"Dense branch restricted to {ROUTING_M} nearest clusters",
            replace(base, route_top_m=ROUTING_M),
        ),
        Configuration(
            f"Hybrid + MMR (λ={lam})", "MMR over the fused candidate pool", replace(base, mmr_lambda=lam)
        ),
        Configuration(
            f"Hybrid + routing + MMR (m={ROUTING_M}, λ={lam})",
            "Routing and MMR combined",
            replace(base, route_top_m=ROUTING_M, mmr_lambda=lam),
        ),
    ]
    shipped = replace(base, route_top_m=defaults.route_top_m, mmr_lambda=defaults.mmr_lambda)
    for i, config in enumerate(configs):
        if config.options == shipped:
            configs[i] = replace(
                config,
                name=f"{config.name} (default)",
                description=f"{config.description}: the shipped configuration",
            )
            return configs
    configs.append(Configuration("Shipped default", "The configured defaults", shipped))
    return configs


@dataclass(frozen=True, slots=True)
class Summary:
    """Aggregate results for one configuration."""

    name: str
    description: str
    ndcg: float
    ndcg_ci: tuple[float, float]
    recall_at_5: float
    recall_at_k: float
    recall_ci: tuple[float, float]
    mrr: float
    delta_ndcg_vs_hybrid: float
    delta_ndcg_ci: tuple[float, float]
    distinct_docs: float
    intra_list_similarity: float
    latency_p50_ms: float
    ndcg_by_source: dict[str, float]
    recall_by_source: dict[str, float]


async def run_evaluation(
    services: Services, *, queries_path: Path, corpus_path: Path, docs_dir: Path, k: int = 10
) -> Path:
    """Run the ablation and write the report, raw JSON, charts, and README section.

    Args:
        services: Wired pipeline components pointed at an ingested, clustered index.
        queries_path: Frozen query set.
        corpus_path: Corpus JSONL (for reporting corpus statistics).
        docs_dir: Output directory for ``eval_results.md``/``.json`` and ``assets/``.
        k: Rank cutoff for all metrics.

    Returns:
        Path of the written markdown report.
    """
    queries = load_queries(queries_path)
    embeddings = await services.retriever.embed_queries([q.query for q in queries])
    configs = ablation_configs(services.retriever.defaults, k)

    per_config: dict[str, list[QueryScores]] = {}
    for config in configs:
        per_config[config.name] = await _run_config(services, queries, embeddings, config.options, k)
        log.info("eval.config_done", config=config.name)

    baseline = np.array([s.ndcg_at_k for s in per_config["Hybrid (RRF)"]])
    summaries = [_summarize(c, per_config[c.name], queries, baseline) for c in configs]
    index_stats = await _index_stats(services, corpus_path)

    # Blocking I/O is fine here: this runs only as the one-shot `meridian eval` command.
    docs_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    assets = docs_dir / "assets"
    assets.mkdir(exist_ok=True)
    meta = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "k": k,
        "queries": len(queries),
        "queries_by_source": dict(Counter(q.source for q in queries)),
        "embedding_model": services.settings.voyage.model,
        **index_stats,
    }
    (docs_dir / "eval_results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "summaries": [asdict(s) for s in summaries],
                "per_query": {name: [asdict(s) for s in scores] for name, scores in per_config.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    render_chart(summaries, k, assets / "eval_results_light.png", dark=False)
    render_chart(summaries, k, assets / "eval_results_dark.png", dark=True)

    report = docs_dir / "eval_results.md"
    report.write_text(_report_markdown(summaries, meta), encoding="utf-8")
    _update_readme(Path("README.md"), _readme_section(summaries, meta))
    return report


async def _run_config(
    services: Services,
    queries: Sequence[EvalQuery],
    embeddings: Sequence[QueryEmbedding],
    options: RetrievalOptions,
    k: int,
) -> list[QueryScores]:
    # Sequential on purpose: latency numbers stay free of self-inflicted contention.
    scores: list[QueryScores] = []
    for query, embedding in zip(queries, embeddings, strict=True):
        started = time.perf_counter()
        result = await services.retriever.retrieve_embedded(query.query, embedding, options)
        latency = (time.perf_counter() - started) * 1000
        scores.append(score_ranking(result.chunks, query.relevant_doc_ids, k, latency))
    return scores


def _summarize(
    config: Configuration,
    scores: Sequence[QueryScores],
    queries: Sequence[EvalQuery],
    baseline_ndcg: NDArray[np.float64],
) -> Summary:
    ndcg = np.array([s.ndcg_at_k for s in scores])
    recall = np.array([s.recall_at_k for s in scores])
    by_source: dict[str, list[float]] = defaultdict(list)
    recall_by_source: dict[str, list[float]] = defaultdict(list)
    for q, s in zip(queries, scores, strict=True):
        by_source[q.source].append(s.ndcg_at_k)
        recall_by_source[q.source].append(s.recall_at_k)
    return Summary(
        name=config.name,
        description=config.description,
        ndcg=float(ndcg.mean()),
        ndcg_ci=bootstrap_ci(ndcg),
        recall_at_5=float(np.mean([s.recall_at_5 for s in scores])),
        recall_at_k=float(recall.mean()),
        recall_ci=bootstrap_ci(recall),
        mrr=float(np.mean([s.mrr_at_k for s in scores])),
        delta_ndcg_vs_hybrid=float((ndcg - baseline_ndcg).mean()),
        delta_ndcg_ci=bootstrap_ci(ndcg - baseline_ndcg),  # paired: same queries, same resamples
        distinct_docs=float(np.mean([s.distinct_docs for s in scores])),
        intra_list_similarity=float(np.mean([s.intra_list_similarity for s in scores])),
        latency_p50_ms=float(statistics.median(s.latency_ms for s in scores)),
        ndcg_by_source={src: float(np.mean(v)) for src, v in sorted(by_source.items())},
        recall_by_source={src: float(np.mean(v)) for src, v in sorted(recall_by_source.items())},
    )


async def _index_stats(services: Services, corpus_path: Path) -> dict[str, Any]:
    documents = list(load_documents(corpus_path))
    chunk_count = await services.store.count_chunks()
    centroids = await services.store.load_centroids()

    pairs: list[tuple[int, str]] = []
    async for payload in services.store.iter_payloads(["cluster_id", "metadata"]):
        cluster = payload.get("cluster_id")
        topic = (payload.get("metadata") or {}).get("topic")
        if cluster is not None and topic is not None:
            pairs.append((int(cluster), str(topic)))
    purity = None
    if pairs:
        by_cluster: dict[int, Counter[str]] = defaultdict(Counter)
        for cluster, topic in pairs:
            by_cluster[cluster][topic] += 1
        purity = sum(c.most_common(1)[0][1] for c in by_cluster.values()) / len(pairs)

    return {
        "documents": len(documents),
        "documents_by_source": dict(Counter(d.metadata.get("source", "unknown") for d in documents)),
        "topics": sorted({d.metadata.get("topic", "unknown") for d in documents}),
        "chunks": chunk_count,
        "clusters": int(centroids.shape[0]) if centroids is not None else 0,
        "cluster_topic_purity": round(purity, 4) if purity is not None else None,
    }


# ------------------------------------------------------------------ rendering


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _fmt_delta(value: float, ci: tuple[float, float]) -> str:
    if abs(value) < 5e-4 and ci == (0.0, 0.0):
        return "—"
    return f"{value:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _main_table(summaries: Sequence[Summary], k: int) -> str:
    header = (
        f"| Configuration | nDCG@{k} [95% CI] | Δ nDCG vs hybrid [95% CI] | Recall@5 | Recall@{k} | "
        f"MRR@{k} | Distinct docs@{k} | p50 latency |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    best_ndcg = max(s.ndcg for s in summaries)
    rows = []
    for s in summaries:
        ndcg = f"{s.ndcg:.3f} {_fmt_ci(s.ndcg_ci)}"
        if s.ndcg == best_ndcg:
            ndcg = f"**{ndcg}**"
        rows.append(
            f"| {s.name} | {ndcg} | {_fmt_delta(s.delta_ndcg_vs_hybrid, s.delta_ndcg_ci)} | "
            f"{s.recall_at_5:.3f} | {s.recall_at_k:.3f} | {s.mrr:.3f} | {s.distinct_docs:.1f} | "
            f"{s.latency_p50_ms:.1f} ms |"
        )
    return header + "\n".join(rows)


def _picture(prefix: str) -> str:
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{prefix}eval_results_dark.png">\n'
        f'  <img alt="nDCG and Recall by pipeline configuration, with 95% confidence intervals" '
        f'src="{prefix}eval_results_light.png">\n'
        "</picture>"
    )


def _report_markdown(summaries: Sequence[Summary], meta: dict[str, Any]) -> str:
    k = meta["k"]
    sources = sorted({src for s in summaries for src in s.ndcg_by_source})
    nan = float("nan")
    by_source = (
        "| Configuration | "
        + " | ".join(f"nDCG@{k} ({src}) | Recall@{k} ({src})" for src in sources)
        + " |\n"
        "|---|"
        + "---|---|" * len(sources)
        + "\n"
        + "\n".join(
            f"| {s.name} | "
            + " | ".join(
                f"{s.ndcg_by_source.get(src, nan):.3f} | {s.recall_by_source.get(src, nan):.3f}"
                for src in sources
            )
            + " |"
            for s in summaries
        )
    )
    diversity = (
        f"| Configuration | Distinct docs@{k} | Intra-list similarity@{k} (lower = more diverse) |\n"
        "|---|---|---|\n"
        + "\n".join(
            f"| {s.name} | {s.distinct_docs:.2f} | {s.intra_list_similarity:.3f} |" for s in summaries
        )
    )
    configs = "\n".join(f"- **{s.name}**: {s.description}" for s in summaries)
    purity = meta.get("cluster_topic_purity")
    purity_line = (
        f"- Cluster/topic purity: **{purity:.1%}** of chunks fall in a cluster "
        "whose majority topic is their own "
        "(topic labels are never shown to the clusterer)."
        if purity is not None
        else "- Clusters: not fitted."
    )
    return f"""# Retrieval evaluation

_Generated by `meridian eval` on {meta["generated_at"]}. Raw per-query scores: \
[`eval_results.json`](eval_results.json)._

{_picture("assets/")}

## Results

{_main_table(summaries, k)}

Δ columns are **paired** differences against the Hybrid (RRF) row on the same queries; an interval that \
excludes zero is a difference the query sample supports. Latency is route + search + re-rank against a \
local Qdrant, excluding query embedding (identical across rows), measured sequentially.

### By query source

{by_source}

### Result-list diversity

{diversity}

## Setup

- Corpus: {meta["documents"]} documents \
({", ".join(f"{v} {k_}" for k_, v in meta["documents_by_source"].items())}) \
across {len(meta["topics"])} topics ({", ".join(meta["topics"])}), indexed as **{meta["chunks"]} chunks**.
- Embeddings: `{meta["embedding_model"]}` (dense) + BM25 with server-side IDF (sparse), in Qdrant.
- Clusters: {meta["clusters"]} (spherical k-means, k chosen by cosine silhouette).
{purity_line}
- Queries: {meta["queries"]} ({", ".join(f"{v} {k_}" for k_, v in meta["queries_by_source"].items())}), \
written by Claude and frozen in `data/eval/queries.jsonl`. Single-hop (`wikipedia`, `arxiv`) queries are \
written from one sampled passage, whose document is the relevant one. Multi-hop (`multihop`) queries are \
written from passages of two same-topic articles and need both; both documents are relevant.
- Distractors: the corpus includes near-duplicate `mirror` copies of some articles (lead and headings \
kept, ~20% of other paragraphs dropped). A mirror is scored as its original, so it earns no extra credit \
and no extra distinct-document count, but its chunks can crowd other documents out of the top k.
- Metrics use document-level relevance at cutoff k={k}: a document counts at the rank of its first chunk. \
For multi-hop queries, Recall@{k} is the fraction of the two relevant documents retrieved.

### Configurations

{configs}

## Caveats

- **Sparse judgments.** Only the source document(s) of each query are judged relevant. Other documents \
that also answer it count as misses, so absolute scores are a lower bound; comparisons between rows are \
the meaningful signal.
- **Generated queries.** Each is written from specific passages. Despite the paraphrasing instruction they \
can share vocabulary with them, which may favor lexical (BM25) matching relative to real user queries.
- **Synthetic distractors.** Mirrors are mechanical copies, more uniform than real-world overlap between \
documents; they isolate the redundancy effect rather than model it faithfully.
- **MMR is a diversity optimization, not a relevance one.** It can still move nDCG either way: a document \
counts at the rank of its first chunk, so demoting redundant chunks of other documents can lift the relevant \
one, while demoting a chunk too similar to one already chosen can push it down. Its intended effect shows \
in the diversity table; the Δ column shows the net effect on ranking quality.
- Sample size: CIs come from {BOOTSTRAP_RESAMPLES:,} bootstrap resamples over queries (seed {BOOTSTRAP_SEED}).
"""


def _readme_section(summaries: Sequence[Summary], meta: dict[str, Any]) -> str:
    return (
        f"{README_START}\n"
        f"_{meta['queries']} queries over {meta['chunks']} chunks from {meta['documents']} documents · "
        f"generated {meta['generated_at']} by `meridian eval` · "
        f"full report: [docs/eval_results.md](docs/eval_results.md)_\n\n"
        f"{_picture('docs/assets/')}\n\n"
        f"{_main_table(summaries, meta['k'])}\n"
        f"{README_END}"
    )


def _update_readme(readme: Path, section: str) -> None:
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        log.warning("eval.readme_markers_missing", path=str(readme))
        return
    readme.write_text(pattern.sub(lambda _: section, text), encoding="utf-8")


# Reference-palette tokens (dataviz skill): one series → one hue, no legend.
_THEMES: dict[bool, dict[str, str]] = {
    False: {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": "#2a78d6",
    },
    True: {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "series": "#3987e5",
    },
}


def render_chart(summaries: Sequence[Summary], k: int, out: Path, *, dark: bool) -> None:
    """Render nDCG@k and Recall@k as two dot-and-interval panels sharing one row axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = _THEMES[dark]
    names = [s.name for s in summaries][::-1]
    panels: list[tuple[str, Callable[[Summary], float], Callable[[Summary], tuple[float, float]]]] = [
        (f"nDCG@{k}", lambda s: s.ndcg, lambda s: s.ndcg_ci),
        (f"Recall@{k}", lambda s: s.recall_at_k, lambda s: s.recall_ci),
    ]
    ordered = list(summaries)[::-1]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(names) + 1.4), sharey=True, dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    lows = [lo for _, _, ci in panels for s in ordered for lo in [ci(s)[0]]]
    x_min = max(0.0, np.floor(min(lows) * 20) / 20 - 0.05)

    for ax, (title, value, ci) in zip(axes, panels, strict=True):
        ax.set_facecolor(theme["surface"])
        ys = np.arange(len(ordered))
        vals = [value(s) for s in ordered]
        cis = [ci(s) for s in ordered]
        ax.hlines(ys, [c[0] for c in cis], [c[1] for c in cis], color=theme["series"], linewidth=2, zorder=2)
        ax.scatter(vals, ys, s=64, color=theme["series"], edgecolor=theme["surface"], linewidth=2, zorder=3)
        best = int(np.argmax(vals))
        ax.annotate(
            f"{vals[best]:.3f}",
            (cis[best][1], ys[best]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=theme["text"],
            fontsize=9,
        )
        ax.set_title(title, loc="left", color=theme["text"], fontsize=11, fontweight="bold")
        ax.set_xlim(x_min, 1.0)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=theme["muted"], length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(np.arange(len(names)), names, color=theme["text"])
    fig.text(
        0.01, 0.01, "Dots: mean over queries · bars: 95% bootstrap CI", color=theme["muted"], fontsize=8.5
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, facecolor=theme["surface"])
    plt.close(fig)
