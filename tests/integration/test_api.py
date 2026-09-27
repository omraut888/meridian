"""HTTP API tests: the real FastAPI app over live Qdrant, Voyage AI, and Claude."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from meridian.api import create_app

from .conftest import IndexedCorpus

pytestmark = pytest.mark.integration

RAFT_QUERY = (
    "consensus algorithm built on leader election and a replicated log, designed to be understandable"
)
RAFT_DOC = "wikipedia:40226710"


@pytest.fixture(scope="session")
async def client(indexed: IndexedCorpus) -> AsyncIterator[httpx.AsyncClient]:
    # Same throwaway collections as the pipeline tests; the app builds its own clients in its lifespan.
    app = create_app(indexed.services.settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://meridian.test") as http:
            yield http


def _sse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


async def test_health_and_readiness(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


async def test_retrieve_returns_ranked_real_chunks(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/retrieve", json={"query": RAFT_QUERY, "top_k": 5}, headers={"x-request-id": "it-retrieve-1"}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert response.headers["x-request-id"] == "it-retrieve-1"
    assert body["request_id"] == "it-retrieve-1"
    assert len(body["results"]) == 5
    assert RAFT_DOC in [r["doc_id"] for r in body["results"]]
    assert all(r["text"] and r["source_uri"].startswith("https://") for r in body["results"])
    assert body["routed_clusters"] == []  # routing is off by default
    assert {"embed", "total"} <= body["timings_ms"].keys()


async def test_retrieve_honours_routing_override(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/retrieve", json={"query": RAFT_QUERY, "route_top_m": 2})
    assert response.status_code == 200, response.text
    assert len(response.json()["routed_clusters"]) == 2


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "mmr_lambda": 1.5}, {}],
)
async def test_retrieve_rejects_invalid_requests(
    client: httpx.AsyncClient, payload: dict[str, object]
) -> None:
    # Validation runs before any embedding call, so these cost no API quota.
    assert (await client.post("/v1/retrieve", json=payload)).status_code == 422


async def test_answer_streams_sources_text_citations_then_done(client: httpx.AsyncClient) -> None:
    question = "How does Raft elect a leader, and what happens when the leader fails?"
    async with client.stream("POST", "/v1/answer", json={"query": question}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse_events((await response.aread()).decode())

    kinds = [kind for kind, _ in events]
    assert kinds[0] == "sources"
    assert kinds[-1] == "done", events[-1]
    assert kinds.count("done") == 1
    assert "error" not in kinds
    assert set(kinds[1:-1]) <= {"text", "citation"}

    done = events[-1][1]
    assert done["refused"] is False
    assert done["stop_reason"] == "end_turn"

    sources = events[0][1]["results"]
    assert isinstance(sources, list)
    text = "".join(str(data["text"]) for kind, data in events if kind == "text")
    citations = [data for kind, data in events if kind == "citation"]
    assert text.strip()
    assert citations, "answer carried no citations"
    for citation in citations:
        source = sources[int(str(citation["source_index"]))]
        assert citation["chunk_id"] == source["chunk_id"]
        assert str(citation["cited_text"]).strip() in source["text"]
