"""Render a /v1/answer SSE stream from stdin: streamed text with [n] markers, then a sources list."""

import json
import sys

BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"

sources: list[dict] = []
cited: dict[str, int] = {}  # doc_id -> footnote number, in first-cited order
event = None
for line in sys.stdin:
    line = line.rstrip("\n")
    if line.startswith("event: "):
        event = line[7:]
    elif line.startswith("data: "):
        data = json.loads(line[6:])
        if event == "sources":
            sources = data["results"]
        elif event == "text":
            print(data["text"], end="", flush=True)
        elif event == "citation":
            doc = sources[data["source_index"]]
            n = cited.setdefault(doc["doc_id"], len(cited) + 1)
            print(f"{CYAN}[{n}]{RESET}", end="", flush=True)
        elif event == "done":
            print(f"\n\n{BOLD}Sources{RESET}")
            by_id = {s["doc_id"]: s for s in sources}
            for doc_id, n in cited.items():
                s = by_id[doc_id]
                print(f"  {CYAN}[{n}]{RESET} {s['title']}  {DIM}{s['source_uri']}{RESET}")
            print(f"{DIM}\n{data['model']} · {data['output_tokens']} output tokens · {data['generation_ms'] / 1000:.1f}s{RESET}")
        elif event == "error":
            print(f"\nerror: {data['type']}: {data['detail']}", file=sys.stderr)
