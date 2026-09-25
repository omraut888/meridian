"""RateLimiter tests on a fake clock (no real sleeping, no network)."""

import pytest

from meridian.config import VoyageSettings
from meridian.embeddings import RateLimiter, VoyageEmbedder


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _limiter(clock: FakeClock, *, requests: int | None = None, tokens: int | None = None) -> RateLimiter:
    return RateLimiter(requests=requests, tokens=tokens, clock=clock, sleep=clock.sleep)


async def test_requests_beyond_the_limit_wait_for_the_window() -> None:
    clock = FakeClock()
    limiter = _limiter(clock, requests=3)
    for _ in range(3):
        await limiter.acquire(1)
    assert clock.sleeps == []

    clock.now = 10.0
    await limiter.acquire(1)
    assert clock.now == pytest.approx(60.0)  # first request (t=0) left the window


async def test_token_budget_waits_until_enough_tokens_expire() -> None:
    clock = FakeClock()
    limiter = _limiter(clock, tokens=10_000)
    await limiter.acquire(6_000)  # t=0
    clock.now = 20.0
    await limiter.acquire(3_000)  # t=20, 9_000 used
    clock.now = 30.0
    await limiter.acquire(5_000)  # needs the t=0 entry gone, not the t=20 one
    assert clock.now == pytest.approx(60.0)


async def test_request_larger_than_token_budget_is_rejected() -> None:
    limiter = _limiter(FakeClock(), tokens=10_000)
    with pytest.raises(ValueError, match="exceeds"):
        await limiter.acquire(10_001)


async def test_no_limits_never_waits() -> None:
    clock = FakeClock()
    limiter = _limiter(clock)
    for _ in range(100):
        await limiter.acquire(50_000)
    assert clock.sleeps == []


def test_batches_are_capped_at_the_token_budget() -> None:
    settings = VoyageSettings(api_key="unused", tokens_per_minute=2_000, requests_per_minute=3)
    embedder = VoyageEmbedder(settings)
    texts = ["word " * 300] * 20  # ~300 tokens each
    batches = list(embedder._pack_batches(texts))

    assert all(tokens <= 2_000 for _, tokens in batches)
    assert sum(len(b) for b, _ in batches) == len(texts)
    assert [tokens for _, tokens in batches] == [sum(embedder.count_tokens(b)) for b, _ in batches]
