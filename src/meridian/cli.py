"""Command-line entry point: ``meridian <command>``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, TypeVar

import structlog
import typer

from meridian.clustering import recluster_corpus
from meridian.config import get_settings
from meridian.exceptions import MeridianError
from meridian.ingestion import load_documents
from meridian.observability import configure_logging
from meridian.services import Services

T = TypeVar("T")

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Meridian hybrid RAG pipeline.")
log = structlog.get_logger(__name__)

DEFAULT_CORPUS = Path("data/corpus/documents.jsonl")
DEFAULT_QUERIES = Path("data/eval/queries.jsonl")


@app.callback()
def _setup() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)


def _run(main: Callable[[], Awaitable[T]]) -> T:
    """Run an async command, converting domain errors into a clean non-zero exit."""
    try:
        return asyncio.run(main())
    except MeridianError as exc:
        log.error("cli.failed", error=type(exc).__name__, detail=str(exc))
        raise typer.Exit(code=1) from exc


async def _with_services(fn: Callable[[Services], Awaitable[T]]) -> T:
    services = await Services.create(get_settings())
    try:
        return await fn(services)
    finally:
        await services.aclose()


@app.command("fetch-corpus")
def fetch_corpus_cmd(
    out: Annotated[Path, typer.Option(help="Destination JSONL file.")] = DEFAULT_CORPUS,
    arxiv_per_topic: Annotated[int, typer.Option(min=1, max=100)] = 15,
    wikipedia_max_chars: Annotated[int, typer.Option(min=500)] = 5_000,
) -> None:
    """Download the Wikipedia + arXiv corpus from their public APIs."""
    from meridian.corpus import fetch_corpus

    count = _run(
        lambda: fetch_corpus(out, arxiv_per_topic=arxiv_per_topic, wikipedia_max_chars=wikipedia_max_chars)
    )
    typer.echo(f"wrote {count} documents to {out}")


@app.command()
def ingest(
    path: Annotated[
        Path, typer.Argument(help="Corpus .jsonl file or directory of .md/.txt.")
    ] = DEFAULT_CORPUS,
) -> None:
    """Chunk, embed, and index documents (incremental; unchanged docs are skipped)."""

    async def main(services: Services) -> int:
        report = await services.ingestion_pipeline().run(load_documents(path))
        typer.echo(
            f"seen={report.documents_seen} unchanged={report.documents_unchanged} "
            f"indexed={report.documents_indexed} chunks={report.chunks_upserted} "
            f"failures={len(report.failures)} elapsed={report.elapsed_s}s"
        )
        return len(report.failures)

    if _run(lambda: _with_services(main)):
        raise typer.Exit(code=1)


@app.command()
def recluster() -> None:
    """Refit spherical k-means over all indexed chunks and persist the assignments."""

    async def main(services: Services) -> None:
        model = await recluster_corpus(services.store, services.clusterer())
        typer.echo(f"k={model.k} silhouette={model.silhouette:.4f} version={model.version}")
        typer.echo(f"silhouette by k: {model.silhouette_by_k}")

    _run(lambda: _with_services(main))


@app.command("build-eval-set")
def build_eval_set_cmd(
    corpus: Annotated[Path, typer.Option(help="Corpus JSONL.")] = DEFAULT_CORPUS,
    out: Annotated[Path, typer.Option(help="Destination queries JSONL.")] = DEFAULT_QUERIES,
    passages_per_article: Annotated[int, typer.Option(min=1, max=10)] = 2,
) -> None:
    """Generate evaluation queries from corpus passages with Claude (run once; commit the output)."""
    from meridian.evaluation import build_eval_set

    settings = get_settings()
    count = _run(
        lambda: build_eval_set(corpus, out, settings.generation, passages_per_article=passages_per_article)
    )
    typer.echo(f"wrote {count} queries to {out}")


@app.command("eval")
def eval_cmd(
    queries: Annotated[Path, typer.Option(help="Queries JSONL.")] = DEFAULT_QUERIES,
    corpus: Annotated[Path, typer.Option(help="Corpus JSONL (for reporting stats).")] = DEFAULT_CORPUS,
    docs_dir: Annotated[Path, typer.Option(help="Where to write the report.")] = Path("docs"),
    k: Annotated[int, typer.Option(min=1, max=50)] = 10,
) -> None:
    """Run the retrieval ablation and write docs/eval_results.{md,json} plus a chart."""
    from meridian.evaluation import run_evaluation

    async def main(services: Services) -> Path:
        return await run_evaluation(
            services, queries_path=queries, corpus_path=corpus, docs_dir=docs_dir, k=k
        )

    report = _run(lambda: _with_services(main))
    typer.echo(f"wrote {report}")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("meridian.api:create_app", factory=True, host=host, port=port, log_config=None)


if __name__ == "__main__":
    app()
