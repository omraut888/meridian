"""Print the stage-by-stage ablation from docs/eval_results.json as a terminal table."""

import json
import sys

from rich import box
from rich.console import Console
from rich.table import Table

data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "docs/eval_results.json"))
meta, rows = data["meta"], data["summaries"]
best = max(r["ndcg"] for r in rows)

table = Table(
    title=f"Meridian retrieval ablation · {meta['queries']} queries · {meta['chunks']} chunks · k={meta['k']}",
    caption="Δ is the paired difference vs Hybrid (RRF) with a 95% bootstrap CI",
    box=box.ROUNDED,
    header_style="bold",
)
table.add_column("Configuration")
for col in ("nDCG@10", "Δ vs hybrid [95% CI]", "Recall@10", "MRR@10", "Distinct docs", "p50 ms"):
    table.add_column(col, justify="right")

for r in rows:
    lo, hi = r["delta_ndcg_ci"]
    significant = lo > 0 or hi < 0
    delta = "—" if r["name"] == "Hybrid (RRF)" else f"{r['delta_ndcg_vs_hybrid']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    delta_style = ("red" if hi < 0 else "green") if significant else "dim"
    ndcg = f"{r['ndcg']:.3f}"
    table.add_row(
        r["name"],
        f"[bold green]{ndcg}[/]" if r["ndcg"] == best else ndcg,
        f"[{delta_style}]{delta}[/]",
        f"{r['recall_at_k']:.3f}",
        f"{r['mrr']:.3f}",
        f"{r['distinct_docs']:.1f}",
        f"{r['latency_p50_ms']:.1f}",
    )

Console().print(table)
