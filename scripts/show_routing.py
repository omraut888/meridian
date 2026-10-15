"""Summarize routing_diag.json: did routing lose queries because relevant chunks sat in unrouted clusters?

Usage: python scripts/show_routing.py [ROUTING_DIAG_JSON]   (default: routing_diag.json)
"""

import json
import sys

from rich import box
from rich.console import Console
from rich.table import Table

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "routing_diag.json"))
n = len(rows)


def outside(row: dict, m: int) -> bool:
    """True if any chunk of any relevant document is outside the m routed clusters."""
    routed = set(row["routed"][:m])
    return any(c not in routed for clusters in row["rel_chunk_clusters"].values() for c in clusters)


def lost(row: dict, key: str) -> bool:
    return row[key]["ndcg"] < row["h"]["ndcg"] - 1e-9


def mean(key: str, field: str) -> float:
    return sum(r[key][field] for r in rows) / n


def frac(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


table = Table(
    title=f"Cluster routing vs Hybrid (RRF) · {n} queries · k=10",
    caption="'outside' = some chunk of a relevant doc is in a cluster the dense branch didn't search",
    box=box.ROUNDED,
    header_style="bold",
)
table.add_column("")
for col in ("Hybrid", "Routing m=1", "Routing m=3"):
    table.add_column(col, justify="right")

stats = {}
for key, m in (("m1", 1), ("m3", 3)):
    lost_rows = [r for r in rows if lost(r, key)]
    out_rows = [r for r in rows if outside(r, m)]
    in_rows = [r for r in rows if not outside(r, m)]
    stats[key] = {
        "lost": str(len(lost_rows)),
        "lost_out": frac(sum(outside(r, m) for r in lost_rows), len(lost_rows)),
        "rate_out": frac(sum(lost(r, key) for r in out_rows), len(out_rows)),
        "rate_in": frac(sum(lost(r, key) for r in in_rows), len(in_rows)),
        "kept": frac(sum(r[key]["first"] is not None for r in lost_rows), len(lost_rows)),
        "ndcg": f"{mean(key, 'ndcg'):.3f}",
        "recall": f"{mean(key, 'recall'):.3f}",
    }

labels = {
    "lost": "Queries that lost nDCG",
    "lost_out": "  …with relevant chunks outside",
    "rate_out": "Loss rate, relevant outside",
    "rate_in": "Loss rate, relevant inside",
    "kept": "Lost queries still in top 10",
    "ndcg": "nDCG@10",
    "recall": "Recall@10",
}
hybrid = {"ndcg": f"{mean('h', 'ndcg'):.3f}", "recall": f"{mean('h', 'recall'):.3f}"}
for field, label in labels.items():
    if field == "ndcg":
        table.add_section()
    table.add_row(label, hybrid.get(field, "—"), stats["m1"][field], stats["m3"][field])

Console().print(table)
