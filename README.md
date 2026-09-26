# Meridian

Hybrid RAG pipeline with cluster-aware retrieval and MMR re-ranking.

Documents are chunked along their structure, embedded with Voyage AI (dense) and BM25 (sparse), and
indexed in Qdrant. Queries run both branches, fuse them with reciprocal rank fusion, optionally restrict
the dense branch to the nearest topic clusters, and re-rank with MMR. Claude writes the answer, citing
the retrieved chunks.

## Setup

Requires Python 3.11+ and Docker.

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

Voyage accounts without a payment method are limited to 3 requests and 10K tokens per minute. Without
pacing, the default batch size exceeds that and embedding fails after its retries. Turn on client-side
pacing, which caps batch size at the token budget and spaces requests to fit both limits:

```bash
export MERIDIAN_VOYAGE__REQUESTS_PER_MINUTE=3
export MERIDIAN_VOYAGE__TOKENS_PER_MINUTE=6000
```

Use 6000, not the nominal 10K: Voyage rejected every ~9K-token batch even though it billed them at the
same token count we measure locally, while 6K batches went through with zero retries.

Everything still works, just slower: ingesting the full corpus takes several minutes instead of seconds.
Adding a payment method lifts the limits (Voyage's free token allowance still applies).

## Usage

```bash
meridian ingest              # chunk, embed, and index data/corpus/documents.jsonl (incremental)
meridian recluster           # fit topic clusters over the indexed chunks
meridian serve               # HTTP API: POST /v1/retrieve, POST /v1/answer (streamed, cited)
```

## Evaluation

```bash
meridian build-eval-set      # Claude writes one query per sampled passage -> data/eval/queries.jsonl
meridian eval                # ablation -> docs/eval_results.md, updates the section below
```

<!-- eval:start -->
_180 queries over 179 chunks from 138 documents · generated 2026-09-25 18:35 UTC by `meridian eval` · full report: [docs/eval_results.md](docs/eval_results.md)_

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/eval_results_dark.png">
  <img alt="nDCG and Recall by pipeline configuration, with 95% confidence intervals" src="docs/assets/eval_results_light.png">
</picture>

| Configuration | nDCG@10 [95% CI] | Δ nDCG vs hybrid [95% CI] | Recall@5 | Recall@10 | MRR@10 | Distinct docs@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| Dense only | **0.995 [0.988, 1.000]** | +0.005 [-0.001, +0.013] | 1.000 | 1.000 | 0.994 | 8.3 | 6.5 ms |
| BM25 only | 0.968 [0.947, 0.986] | -0.022 [-0.041, -0.005] | 0.989 | 0.994 | 0.959 | 9.0 | 8.2 ms |
| Hybrid (RRF) | 0.990 [0.979, 0.998] | — | 1.000 | 1.000 | 0.986 | 8.7 | 14.7 ms |
| Hybrid + routing (m=1) | 0.974 [0.959, 0.988] | -0.016 [-0.031, -0.002] | 1.000 | 1.000 | 0.965 | 8.5 | 19.3 ms |
| Hybrid + routing (m=3) | 0.990 [0.979, 0.998] | +0.000 [-0.006, +0.006] | 1.000 | 1.000 | 0.986 | 8.6 | 23.6 ms |
| Hybrid + MMR (λ=0.7) | 0.992 [0.984, 0.998] | +0.002 [-0.004, +0.008] | 1.000 | 1.000 | 0.989 | 9.2 | 25.8 ms |
| Hybrid + routing + MMR (default) | 0.992 [0.984, 0.998] | +0.002 [+0.000, +0.006] | 1.000 | 1.000 | 0.989 | 9.2 | 34.1 ms |
<!-- eval:end -->

## Tests

```bash
pytest                       # unit tests (no network)
pytest -m integration        # live Qdrant + Voyage + Claude on throwaway collections
```
