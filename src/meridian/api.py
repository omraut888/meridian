"""HTTP API: retrieval and streaming, cited answer generation."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

import structlog
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from meridian.config import Settings, get_settings
from meridian.exceptions import (
    ConfigurationError,
    EmbeddingError,
    GenerationError,
    MeridianError,
    VectorStoreError,
)
from meridian.generation import AnswerCitation, AnswerComplete, AnswerText
from meridian.models import RetrievalResult, ScoredChunk
from meridian.observability import configure_logging
from meridian.retrieval import RetrievalOptions
from meridian.services import Services

log = structlog.get_logger(__name__)

_STATUS_BY_ERROR: dict[type[MeridianError], int] = {
    EmbeddingError: 502,
    GenerationError: 502,
    VectorStoreError: 503,
    ConfigurationError: 500,
}


# ---------------------------------------------------------------------- schemas


class RetrieveRequest(BaseModel):
    """Retrieval request; unset fields use the configured pipeline defaults."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=50)
    mmr_lambda: float | None = Field(None, ge=0.0, le=1.0)
    route_top_m: int | None = Field(None, ge=0, le=64)


class RetrievedChunk(BaseModel):
    """A retrieved chunk as exposed over the wire."""

    chunk_id: UUID
    doc_id: str
    title: str
    source_uri: str
    section: str | None
    text: str
    score: float
    cluster_id: int | None


class RetrieveResponse(BaseModel):
    """Retrieval response."""

    request_id: str
    results: list[RetrievedChunk]
    routed_clusters: list[int]
    timings_ms: dict[str, float]


def _to_wire(chunk: ScoredChunk) -> RetrievedChunk:
    c = chunk.chunk
    return RetrievedChunk(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        title=c.title,
        source_uri=c.source_uri,
        section=c.metadata.get("section"),
        text=c.text,
        score=chunk.score,
        cluster_id=chunk.cluster_id,
    )


def _retrieve_response(request_id: str, result: RetrievalResult) -> RetrieveResponse:
    return RetrieveResponse(
        request_id=request_id,
        results=[_to_wire(c) for c in result.chunks],
        routed_clusters=[m.cluster_id for m in result.routed_clusters],
        timings_ms=result.timings_ms,
    )


# ----------------------------------------------------------------- app factory


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Configuration; defaults to environment-loaded settings.

    Returns:
        A configured application whose lifespan owns all network clients.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = await Services.create(settings)
        log.info("api.startup", collection=settings.qdrant.chunks_collection)
        try:
            yield
        finally:
            await app.state.services.aclose()
            log.info("api.shutdown")

    app = FastAPI(title="Meridian", version="0.1.0", lifespan=lifespan)
    app.middleware("http")(_request_context)
    app.add_exception_handler(MeridianError, _meridian_error_handler)  # type: ignore[arg-type]

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(services: ServicesDep) -> dict[str, str]:
        await services.store.ping()
        return {"status": "ready"}

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    async def retrieve(body: RetrieveRequest, request: Request, services: ServicesDep) -> RetrieveResponse:
        result = await services.retriever.retrieve(body.query, _options(services, body))
        return _retrieve_response(request.state.request_id, result)

    @app.post("/v1/answer", response_class=StreamingResponse)
    async def answer(body: RetrieveRequest, request: Request, services: ServicesDep) -> StreamingResponse:
        """Stream a cited answer as Server-Sent Events.

        Events, in order: one ``sources`` (the retrieval response), then any
        number of ``text`` and ``citation``, then exactly one of ``done`` or
        ``error``. On ``done`` with ``refused: true``, discard streamed text.
        """
        # Retrieve before opening the stream so retrieval failures are real HTTP errors.
        result = await services.retriever.retrieve(body.query, _options(services, body))
        sources = _retrieve_response(request.state.request_id, result)
        return StreamingResponse(
            _answer_events(services, body.query, result, sources),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(_services)]


def _options(services: Services, body: RetrieveRequest) -> RetrievalOptions:
    defaults = services.retriever.defaults
    top_k = body.top_k or defaults.top_k
    return defaults.with_overrides(
        top_k=top_k,
        candidate_pool=max(defaults.candidate_pool, top_k),
        mmr_lambda=body.mmr_lambda,
        route_top_m=body.route_top_m,
    )


async def _answer_events(
    services: Services, question: str, result: RetrievalResult, sources: RetrieveResponse
) -> AsyncIterator[str]:
    yield _sse("sources", sources.model_dump(mode="json"))
    started = time.perf_counter()
    try:
        async for event in services.generator.stream_answer(question, result.chunks):
            if isinstance(event, AnswerText):
                yield _sse("text", {"text": event.text})
            elif isinstance(event, AnswerCitation):
                yield _sse(
                    "citation",
                    {
                        "source_index": event.source_index,
                        "chunk_id": event.chunk_id,
                        "cited_text": event.cited_text,
                    },
                )
            elif isinstance(event, AnswerComplete):
                yield _sse(
                    "done",
                    {
                        "stop_reason": event.stop_reason,
                        "model": event.model,
                        "refused": event.refused,
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "generation_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
    except MeridianError as exc:
        # Headers are already sent; report in-band and end the stream cleanly.
        log.error("api.answer.stream_error", error=type(exc).__name__, detail=str(exc))
        yield _sse("error", {"type": type(exc).__name__, "detail": str(exc)})


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return response


async def _meridian_error_handler(request: Request, exc: MeridianError) -> JSONResponse:
    status = next((code for cls, code in _STATUS_BY_ERROR.items() if isinstance(exc, cls)), 500)
    log.error("api.error", error=type(exc).__name__, detail=str(exc), status=status)
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": f"urn:meridian:error:{type(exc).__name__}",
            "title": type(exc).__name__,
            "status": status,
            "detail": str(exc),
            "instance": getattr(request.state, "request_id", None),
        },
    )
