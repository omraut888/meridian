"""Typed error hierarchy.

Provider SDK exceptions never cross a module boundary raw; they are wrapped in
one of these types (with ``raise ... from exc``) so callers, and the API layer's
HTTP mapping, depend only on Meridian's own contract.
"""

from __future__ import annotations


class MeridianError(Exception):
    """Base class for all Meridian errors."""


class ConfigurationError(MeridianError):
    """Required configuration is missing or inconsistent."""


class EmbeddingError(MeridianError):
    """An embedding provider call failed after retries."""


class VectorStoreError(MeridianError):
    """A Qdrant operation failed."""


class SchemaMismatchError(VectorStoreError):
    """An existing collection's schema disagrees with the configured one."""


class IngestionError(MeridianError):
    """A document could not be loaded, chunked, or indexed."""


class ClusterModelMissingError(MeridianError):
    """Cluster routing was requested but no cluster model has been fitted."""


class GenerationError(MeridianError):
    """The generation provider failed or declined to answer."""


class CorpusFetchError(MeridianError):
    """A public-corpus source returned an unusable response."""
