# Meridian

A hybrid RAG pipeline with MMR re-ranking, plus the evaluation harness I use to measure what each retrieval
stage actually contributes.

Documents are chunked along their structure, embedded twice (densely with Voyage AI, sparsely with BM25),
and indexed in Qdrant. At query time both branches run, reciprocal rank fusion merges their results, and
MMR re-ranks them. Claude then writes the answer, citing the chunks it drew on. There's also an optional
cluster-routing stage that restricts the dense branch to the nearest topic clusters. I turned it off by
default after the evaluation showed it cost accuracy; [Design rationale](#design-rationale) explains why.

## Setup

You'll need Python 3.11+ and Docker.

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
docker compose up -d --wait          # Qdrant on localhost:6333
cp .env.example .env                 # then fill in both keys
```

`.env` needs two keys:

| Variable | Used for |
|---|---|
| `MERIDIAN_VOYAGE__API_KEY` | Dense embeddings (Voyage AI) |
| `MERIDIAN_GENERATION__API_KEY` | Answer and eval-query generation (Claude). Use a workspace-scoped key. |

### Running without a paid Voyage account

Without a payment method on file, Voyage caps you at 3 requests and 10K tokens per minute. The default
batch size blows past that, so unpaced embedding fails once its retries run out. Client-side pacing fixes
this by capping each batch at the token budget and spacing requests out to stay inside both limits:

```bash
export MERIDIAN_VOYAGE__REQUESTS_PER_MINUTE=3
export MERIDIAN_VOYAGE__TOKENS_PER_MINUTE=6000
```

I use 6000 rather than the nominal 10K because Voyage rejected every ~9K-token batch I sent, even though it
billed those batches at the same token count I measure locally. At 6K, batches went through with zero
retries.

Everything still works this way, only slower: ingesting the full corpus takes several minutes instead of
seconds. Adding a payment method lifts the limits, and Voyage's free token allowance still applies.

## Usage

```bash
meridian ingest              # chunk, embed, and index data/corpus/documents.jsonl (incremental)
meridian recluster           # fit topic clusters (needed only for cluster routing and `meridian eval`)
meridian serve               # HTTP API: POST /v1/retrieve, POST /v1/answer (streamed, cited)
```

## Evaluation

```bash
meridian build-eval-set      # Claude writes single- and multi-document queries -> data/eval/queries.jsonl
meridian eval                # ablation -> docs/eval_results.md, updates the section below
```

<!-- eval:start -->
_290 queries over 359 chunks from 204 documents · generated 2026-09-27 00:01 UTC by `meridian eval` · full report: [docs/eval_results.md](docs/eval_results.md)_

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/eval_results_dark.png">
  <img alt="nDCG and Recall by pipeline configuration, with 95% confidence intervals" src="docs/assets/eval_results_light.png">
</picture>

| Configuration | nDCG@10 [95% CI] | Δ nDCG vs hybrid [95% CI] | Recall@5 | Recall@10 | MRR@10 | Distinct docs@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| Dense only | 0.968 [0.955, 0.980] | -0.004 [-0.016, +0.007] | 0.969 | 0.990 | 0.978 | 6.7 | 6.9 ms |
| BM25 only | 0.935 [0.915, 0.953] | -0.038 [-0.053, -0.023] | 0.967 | 0.993 | 0.926 | 7.5 | 5.1 ms |
| Hybrid (RRF) | 0.972 [0.961, 0.983] | — | 0.983 | 0.997 | 0.977 | 7.0 | 9.3 ms |
| Hybrid + routing (m=1) | 0.938 [0.921, 0.955] | -0.035 [-0.049, -0.022] | 0.978 | 0.995 | 0.931 | 6.9 | 15.8 ms |
| Hybrid + routing (m=3) | 0.968 [0.956, 0.979] | -0.005 [-0.012, +0.002] | 0.983 | 0.997 | 0.971 | 7.0 | 12.7 ms |
| Hybrid + MMR (λ=0.7) (default) | **0.973 [0.961, 0.984]** | +0.000 [-0.005, +0.006] | 0.984 | 0.997 | 0.975 | 8.2 | 35.0 ms |
| Hybrid + routing + MMR (m=3, λ=0.7) | 0.971 [0.960, 0.982] | -0.001 [-0.008, +0.004] | 0.984 | 0.997 | 0.973 | 8.2 | 52.0 ms |
<!-- eval:end -->

## Design rationale

Every retrieval stage started out as a hypothesis, and the eval harness is how I test them. It switches
each stage on and off against the same index and the same query embeddings, so any difference between two
rows comes from that one stage. The numbers here come from the table above (290 queries, 359 chunks, k=10)
and the per-source breakdown in [docs/eval_results.md](docs/eval_results.md), and every claim held up
across two separate runs of the full eval.

### Hybrid search

Dense embeddings capture meaning, and BM25 catches exact terms like algorithm names and acronyms, so I
merge the two with reciprocal rank fusion. Of the three retrieval methods, hybrid is the only one that
stays near the top on every query type. It finds 96.7% of the relevant documents on multi-document
queries, compared with 90.0% for dense search alone, and it beats BM25 alone by 0.038 nDCG@10 overall (95%
CI 0.023 to 0.053). It was an easy one to keep.

### MMR re-ranking

I seeded the corpus with near-duplicate "mirror" articles on purpose, because that kind of redundancy is
what fills a top 10 with the same content. With MMR at λ = 0.7, the number of distinct documents in the
top 10 goes from 7.0 to 8.2, and ranking quality doesn't measurably move: the change in nDCG@10 is
indistinguishable from zero (95% CI −0.005 to +0.006). It does add some latency, though my laptop timings
are too noisy to say how much. MMR stays, and it's on by default.

### Cluster routing

This is the one the eval talked me out of. My hypothesis was that searching only the query's nearest topic
clusters would cut noise from unrelated topics. The clusters themselves came out well: 81% of chunks land
in a cluster whose majority topic is their own, and the clusterer never sees topic labels. The problem is
that routing drops relevant documents sitting in a neighbouring cluster.

Routing to 1 cluster clearly hurts. It cost 0.032 and 0.035 nDCG@10 across two runs, with confidence
intervals well below zero both times. Routing to 3 clusters never helped either. It cost about 0.005,
which is within run-to-run noise (95% CI −0.012 to +0.002 in the latest run), and both settings add
latency.

I built the harness partly to check this design choice, and it did its job by rejecting it. At about 360
chunks, searching everything is already fast, so routing has nothing to save. It might pay off on an index
big enough that exhaustive search gets expensive, but I haven't tested that. So routing ships **off by
default**. The code is still there, and every eval run still measures it: routing to 1 and to 3 clusters,
plus routing combined with MMR. To turn it on, set `MERIDIAN_RETRIEVAL__ROUTE_TOP_M=3` (the number of
clusters to search) and run `meridian recluster` first.

### How far I'd trust these numbers

The multi-document results rest on just 30 queries with 60 relevant documents between them, so 96.7%
versus 90.0% really means 58 versus 54 documents found. The per-source breakdown has no confidence
intervals either, so I read the multi-document differences as indicative rather than established.

Reruns on the same index also aren't bit-identical. Between two runs, BM25 scores changed for 3 of the 290
queries (one was an exact tie between a mirror and its original), which moved nDCG means by up to 0.005. I
haven't pinned down the exact cause, so I treat any difference smaller than about 0.005 as noise.

The single-document queries sit close to ceiling, with Recall@10 at 1.0 for almost every configuration, so
they barely separate the configurations. And because Claude generated the queries from corpus passages,
only each query's source documents count as relevant. The rest of the caveats are in
[docs/eval_results.md](docs/eval_results.md#caveats).

## Tests

```bash
pytest                       # unit tests (no network)
pytest -m integration        # live Qdrant + Voyage + Claude on throwaway collections
```

The integration suite covers ingestion, retrieval, routing and MMR, cited generation, and the HTTP API
(`/healthz`, `/readyz`, `/v1/retrieve`, and the streamed `/v1/answer`). It skips itself when API keys are
missing.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint, format, type checks, and unit tests
on every push and pull request. Integration tests need live keys, so I run them on demand instead: add the
`MERIDIAN_VOYAGE__API_KEY` and `MERIDIAN_GENERATION__API_KEY` repository secrets, then trigger the workflow
from the Actions tab with "Also run live integration tests" checked.

## License

The source code is available under the [PolyForm Strict License 1.0.0](LICENSE). You're free to read it
and run it for noncommercial purposes, but not to distribute it, modify it, or build on it, and commercial
use isn't permitted.

The corpus in `data/corpus/` isn't covered by that license. Its Wikipedia text is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and the arXiv metadata is dedicated to the
public domain under CC0 1.0. Every record names its source URL and license.
