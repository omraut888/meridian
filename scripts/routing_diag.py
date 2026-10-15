"""Rerun hybrid, routing m=1, and routing m=3 per query and record where the relevant chunks live.

Writes one JSON row per eval query: the 3 nearest clusters, the cluster of every chunk of each relevant
document, and each configuration's nDCG@10, first relevant rank, and top-10 (doc, cluster, fused score).
Needs a running Qdrant with a fitted cluster model and a Voyage key (one embedding pass over the queries).

Usage: python scripts/routing_diag.py [OUT_JSON]   (default: routing_diag.json)
Summarize with: python scripts/show_routing.py [OUT_JSON]
"""

import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from meridian.config import get_settings
from meridian.evaluation import _canonical_doc_id, load_queries, score_ranking
from meridian.services import Services

CONFIGS = {"h": 0, "m1": 1, "m3": 3}


async def main(out: Path) -> None:
    services = await Services.create(get_settings())
    try:
        queries = load_queries(Path("data/eval/queries.jsonl"))
        chunk_clusters: dict[str, list[int | None]] = defaultdict(list)
        async for p in services.store.iter_payloads(["cluster_id", "doc_id", "metadata"]):
            doc = (p.get("metadata") or {}).get("duplicate_of") or p.get("doc_id")
            chunk_clusters[doc].append(p.get("cluster_id"))

        embeddings = await services.retriever.embed_queries([q.query for q in queries])
        base = replace(services.retriever.defaults, top_k=10, candidate_pool=40, mmr_lambda=None)
        rows = []
        for q, e in zip(queries, embeddings, strict=True):
            routed = await services.router.route(e.dense, 3)
            row = {
                "id": q.query_id,
                "source": q.source,
                "relevant": sorted(q.relevant_doc_ids),
                "routed": [m.cluster_id for m in routed],
                "rel_chunk_clusters": {d: chunk_clusters[d] for d in q.relevant_doc_ids},
            }
            for name, m in CONFIGS.items():
                r = await services.retriever.retrieve_embedded(q.query, e, replace(base, route_top_m=m))
                s = score_ranking(r.chunks, q.relevant_doc_ids, 10, 0.0)
                row[name] = {
                    "ndcg": s.ndcg_at_k,
                    "recall": s.recall_at_k,
                    "first": s.first_relevant_rank,
                    "top": [[_canonical_doc_id(c), c.cluster_id, c.score] for c in r.chunks],
                }
            rows.append(row)
        out.write_text(json.dumps(rows))
        print(f"wrote {len(rows)} rows to {out}")
    finally:
        await services.aclose()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1] if len(sys.argv) > 1 else "routing_diag.json")))
