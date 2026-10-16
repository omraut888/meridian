# Meridian: Design and Evaluation

Most RAG pipelines get evaluated end to end — ask a question, read the answer, decide if it seems right. That measures the net effect only. If one stage helps and another quietly hurts, the aggregate score still looks fine, and the broken stage never gets found.

I wanted evidence for every architectural decision in this system, not just a reasonable-looking output. So Meridian ships with an ablation harness that isolates each stage, runs it against 290 queries, and reports every difference with a 95% confidence interval. This document covers the architecture, the reasoning behind each stage, and what the evaluation actually found — including the one hypothesis it disproved.

## Stack

| Component | Tool | Why |
|---|---|---|
| Embeddings | Voyage AI | Pairs naturally with Claude for generation |
| Vector + sparse storage | Qdrant | Native dense and sparse vector support in one system — hybrid retrieval as a single indexed lookup |
| Clustering | scikit-learn (K-means + DBSCAN) | Mature implementations; the question was whether the architecture using them earns its place, not whether to hand-roll clustering math |
| Generation | Claude | Native citation support for grounded answers |
| API | FastAPI | Async, streaming-native, automatic request validation |

## The corpus

204 documents, 359 chunks, built to stress the architecture rather than flatter it:

- 72 Wikipedia articles and 120 arXiv abstracts across six distinct but related topics — distributed systems, cryptography, NLP, computer vision, database internals, compilers. Distinct topics give clustering real structure; related topics give retrieval something non-trivial to discriminate.
- 12 deliberately near-duplicate "mirror" articles, included so the re-ranking stage would have a real redundancy problem to solve.
- Licensing tracked per record: Wikipedia text is CC BY-SA 4.0, arXiv metadata is CC0. Neither is relicensed by this project — both are excluded explicitly and attributed per document.

The query set: 290 queries, written by Claude as paraphrases of corpus passages, split into single-document and multi-document queries. The multi-document split matters — single-document retrieval is close to solved at this corpus scale, and a benchmark that can't separate architectures isn't a useful benchmark.

## The pipeline

**Chunking.** Documents split along their structure — paragraph and section boundaries — rather than a fixed character count, which avoids cutting a claim away from the context it depends on. Chunks overlap with their neighbors so information near a boundary isn't invisible to retrieval, with overlap logic that stops at section boundaries to avoid duplicating content under the wrong heading.

**Embedding and indexing.** Each chunk gets a dense embedding (Voyage) and a sparse BM25 representation, stored together in Qdrant with its topic label (ground truth, used only for evaluation — the clustering algorithm never sees it) and cluster assignment. Ingestion is incremental: an unchanged document is skipped on re-run rather than re-embedded, which matters given Voyage's rate limits.

**Clustering.** K-means finds well-separated topic groups; DBSCAN catches the density-based outliers K-means would otherwise force into the wrong cluster. Validated against ground-truth labels the clusterer never saw: 81% of chunks land in a cluster matching their actual topic. The clustering works. What I built on top of it is a different story — see below.

**Hybrid retrieval.** Dense and sparse search run in parallel and fuse via Reciprocal Rank Fusion rather than a weighted score combination, since RRF only needs rank position — it doesn't require cosine similarity and BM25 scores to be on comparable scales, which they aren't. An optional cluster-routing stage can restrict the dense branch to the query's nearest cluster(s) before fusion. It ships off by default.

**MMR re-ranking.** The fused candidate pool gets re-ranked by Maximal Marginal Relevance (λ=0.7), which trades off relevance against similarity to already-selected chunks — the mechanism that keeps five near-identical mirror chunks from crowding out genuinely distinct results.

**Cited generation.** The final chunks go to Claude, which writes a grounded answer with citations to the specific supporting chunks, streamed via SSE through FastAPI.

## What the evaluation found

290 queries, 359 chunks, k=10:

| Configuration | nDCG@10 [95% CI] | Δ vs hybrid | Recall@10 | Distinct docs@10 | p50 latency |
|---|---|---|---|---|---|
| Dense only | 0.968 [0.955, 0.980] | −0.004 | 0.990 | 6.7 | 6.9 ms |
| BM25 only | 0.935 [0.915, 0.953] | −0.038 | 0.993 | 7.5 | 5.1 ms |
| **Hybrid (RRF)** | 0.972 [0.961, 0.983] | — | 0.997 | 7.0 | 9.3 ms |
| Hybrid + routing (m=1) | 0.938 [0.921, 0.955] | −0.035 | 0.995 | 6.9 | 15.8 ms |
| Hybrid + routing (m=3) | 0.968 [0.956, 0.979] | −0.005 | 0.997 | 7.0 | 12.7 ms |
| **Hybrid + MMR (default)** | 0.973 [0.961, 0.984] | +0.000 | 0.997 | 8.2 | 35.0 ms |
| Hybrid + routing + MMR | 0.971 [0.960, 0.982] | −0.001 | 0.997 | 8.2 | 52.0 ms |

**Hybrid search — kept.** Dense embeddings blur exact terms — algorithm names, acronyms, identifiers — that BM25 catches cleanly. The clearest evidence is on multi-document queries: hybrid retrieves 96.7% of relevant documents versus 90.0% for dense-only. Against BM25 alone, hybrid wins by 0.038 nDCG@10 (95% CI 0.023–0.053) — outside noise.

**MMR — kept, on by default.** Distinct documents in the top 10 rise from 7.0 to 8.2, with the ranking-quality delta indistinguishable from zero (95% CI −0.005 to +0.006). A real diversity gain at no measured cost, for roughly 6–25ms of added latency.

**Cluster routing — tested, and disproved.** This was the stage I was most confident about going in. Routing to one cluster clearly hurts (−0.035, CI well below zero); routing to three shows a smaller cost that sits at or just past the edge of noise depending on the run.

I didn't stop at the aggregate number. A query-level diagnostic on the queries that got worse showed something more specific than "routing excludes relevant documents":

| | Hybrid | Routing m=1 | Routing m=3 |
|---|---|---|---|
| Queries that lost nDCG | — | 28 | 3 |
| …with relevant chunks outside the routed cluster | — | 27/28 (96%) | 3/3 (100%) |
| Recall@10 | 0.997 | 0.995 | 0.997 |

The first explanation looked confirmed — 96% of hurt queries did have their relevant chunk outside the routed cluster. But Recall@10 barely moved. The documents weren't disappearing from the results, which meant the first explanation was incomplete.

The actual mechanism: routing filters only the dense branch. BM25 still searches everything and still finds the correct chunk — but a chunk found by only one branch earns roughly half the fused RRF score of one found by both. Wrong-topic chunks that happened to sit inside the routed cluster (and were therefore still found by both branches) climbed above the correct, single-branch chunk in the final ranking. The relevant document wasn't lost. It was demoted by an interaction between routing and fusion, not by routing on its own — and filtering both branches instead of one wouldn't fix it, it would just trade demoted documents for missing ones.

Routing ships off by default. It remains implemented and is still measured on every eval run, in case it earns its place at a larger corpus scale where exhaustive search stops being cheap. The bar for a pipeline component here isn't "doesn't hurt." It's "proves it helps."

**Limitations, stated plainly.** The multi-document result rests on 30 queries (60 relevant documents) — real evidence, not enough for a tight interval on the per-source breakdown. Reruns on the same index aren't bit-identical: BM25 scores shifted for 3 of 290 queries between two runs, moving aggregate nDCG by up to 0.005; differences under that threshold are treated as noise. Single-document queries sit near Recall@10 ≈ 1.0 across almost every configuration and barely discriminate between architectures — the multi-document and redundancy-focused parts of the benchmark carry the real signal.

## Testing

37 unit tests (chunking correctness, MMR behavior against hand-built vectors, clustering boundary logic, config validation) and 18 live integration tests against real Qdrant, Voyage, and Claude — no mocks in that layer, covering ingestion, retrieval, routing, MMR, cited generation, and every HTTP endpoint. Lint, formatting, and strict-mode type checking pass clean. CI runs lint/format/type-check/unit tests on every push; integration tests run on manual trigger since they need live keys.

## License

Source: [PolyForm Strict License 1.0.0](../LICENSE) — viewable and runnable for noncommercial purposes, not licensed for redistribution or modification. The corpus in `data/corpus/` is excluded and keeps its original terms — Wikipedia under CC BY-SA 4.0, arXiv metadata under CC0 1.0, each record naming its own source and license.
