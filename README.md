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
_290 queries over 359 chunks from 204 documents · generated 2026-09-25 21:58 UTC by `meridian eval` · full report: [docs/eval_results.md](docs/eval_results.md)_

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/eval_results_dark.png">
  <img alt="nDCG and Recall by pipeline configuration, with 95% confidence intervals" src="docs/assets/eval_results_light.png">
</picture>

| Configuration | nDCG@10 [95% CI] | Δ nDCG vs hybrid [95% CI] | Recall@5 | Recall@10 | MRR@10 | Distinct docs@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| Dense only | 0.968 [0.955, 0.980] | -0.005 [-0.017, +0.005] | 0.969 | 0.990 | 0.978 | 6.7 | 6.8 ms |
| BM25 only | 0.936 [0.916, 0.954] | -0.038 [-0.053, -0.023] | 0.967 | 0.993 | 0.928 | 7.5 | 4.5 ms |
| Hybrid (RRF) | 0.974 [0.963, 0.984] | — | 0.983 | 0.997 | 0.979 | 7.0 | 9.9 ms |
| Hybrid + routing (m=1) | 0.942 [0.926, 0.959] | -0.032 [-0.046, -0.019] | 0.978 | 0.995 | 0.938 | 6.9 | 10.7 ms |
| Hybrid + routing (m=3) | 0.968 [0.956, 0.979] | -0.006 [-0.013, -0.000] | 0.983 | 0.997 | 0.971 | 7.0 | 14.1 ms |
| Hybrid + MMR (λ=0.7) | **0.975 [0.964, 0.985]** | +0.001 [-0.003, +0.004] | 0.986 | 0.997 | 0.977 | 8.2 | 16.0 ms |
| Hybrid + routing + MMR (default) | 0.971 [0.960, 0.982] | -0.003 [-0.009, +0.003] | 0.986 | 0.997 | 0.973 | 8.2 | 20.3 ms |
<!-- eval:end -->

## Tests

```bash
pytest                       # unit tests (no network)
pytest -m integration        # live Qdrant + Voyage + Claude on throwaway collections
```

## License

The source code is available under the [PolyForm Strict License 1.0.0](LICENSE). You may read it and
run it for noncommercial purposes. You may not distribute it, modify it, or build on it, and commercial use
is not permitted.

The corpus in `data/corpus/` is not covered by that license. Wikipedia text there is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and arXiv metadata is dedicated to the
public domain under CC0 1.0. Each record names its source URL and license.
