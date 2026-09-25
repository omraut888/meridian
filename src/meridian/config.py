"""Typed runtime configuration.

All tunables live here and are loaded from environment variables (prefix
``MERIDIAN_``, nested with ``__``) or a local ``.env`` file. Components receive
only the settings slice they need, never the root :class:`Settings` object.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoyageSettings(BaseModel):
    """Voyage AI dense-embedding configuration."""

    api_key: SecretStr | None = None
    model: str = "voyage-4"
    output_dimension: int = 1024
    max_batch_texts: int = Field(128, ge=1, le=1000)
    max_batch_tokens: int = Field(100_000, ge=1_000)
    max_concurrency: int = Field(4, ge=1)
    max_retries: int = Field(6, ge=0)
    timeout_s: float = Field(60.0, gt=0)
    # Client-side pacing for accounts with low rate limits (e.g. Voyage's 3 RPM /
    # 10K TPM without a payment method). Unset means no pacing.
    requests_per_minute: int | None = Field(None, ge=1)
    tokens_per_minute: int | None = Field(None, ge=1)


class SparseSettings(BaseModel):
    """Sparse (lexical) embedding configuration."""

    model: str = "Qdrant/bm25"


class QdrantSettings(BaseModel):
    """Qdrant connection and collection configuration."""

    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None
    chunks_collection: str = "meridian_chunks"
    centroids_collection: str = "meridian_centroids"
    timeout_s: int = Field(30, ge=1)
    upsert_batch_size: int = Field(256, ge=1)


class ChunkingSettings(BaseModel):
    """Structure-aware chunker configuration."""

    max_tokens: int = Field(512, ge=32)
    overlap_tokens: int = Field(64, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_window(self) -> ChunkingSettings:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class ClusteringSettings(BaseModel):
    """Corpus clustering configuration."""

    n_clusters: int | None = Field(None, ge=2)
    k_grid: tuple[int, ...] = (4, 6, 8, 10, 12, 16)
    silhouette_sample_size: int = Field(5_000, ge=100)
    random_state: int = 42


class RetrievalSettings(BaseModel):
    """Hybrid retrieval defaults (all overridable per request)."""

    prefetch_limit: int = Field(100, ge=1)
    candidate_pool: int = Field(40, ge=1)
    top_k: int = Field(8, ge=1)
    mmr_lambda: float | None = Field(0.7, ge=0.0, le=1.0)
    route_top_m: int = Field(3, ge=0)

    @model_validator(mode="after")
    def _pool_covers_top_k(self) -> RetrievalSettings:
        if self.candidate_pool < self.top_k:
            raise ValueError("candidate_pool must be >= top_k")
        return self


class GenerationSettings(BaseModel):
    """Answer-generation configuration (currently backed by Claude)."""

    api_key: SecretStr | None = None
    model: str = "claude-opus-5"
    max_tokens: int = Field(16_000, ge=256)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    server_side_fallback: bool = True
    timeout_s: float = Field(120.0, gt=0)


class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="MERIDIAN_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    voyage: VoyageSettings = VoyageSettings()
    sparse: SparseSettings = SparseSettings()
    qdrant: QdrantSettings = QdrantSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    clustering: ClusteringSettings = ClusteringSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()
