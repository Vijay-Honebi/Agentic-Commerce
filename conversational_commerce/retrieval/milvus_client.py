# conversational_commerce/retrieval/milvus_client.py

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings
from observability.logger import LogEvent, get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

def connect_milvus() -> None:
    """
    Establishes Milvus connection.
    Called once at application startup via main.py lifespan.
    Milvus SDK uses a named alias system — we use 'default'.
    """
    cfg = settings.milvus

    logger.info(
        LogEvent.APP_STARTUP,
        "Connecting to Milvus",
        host=cfg.host,
        port=cfg.port,
        collection=cfg.collection_name,
    )

    connections.connect(
        alias="default",
        host=cfg.host,
        port=cfg.port,
    )

    # Validate collection exists — fail fast at startup, not at first query
    if not utility.has_collection(cfg.collection_name):
        raise RuntimeError(
            f"Milvus collection '{cfg.collection_name}' does not exist. "
            f"Run your embedding_strategy pipeline to create and populate it."
        )

    logger.info(
        LogEvent.APP_STARTUP,
        "Milvus connection established",
        collection=cfg.collection_name,
    )


def disconnect_milvus() -> None:
    """Disconnects from Milvus. Called at application shutdown."""
    connections.disconnect("default")
    logger.info(LogEvent.APP_SHUTDOWN, "Milvus connection closed")


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    """
    Returns the cached Milvus Collection object.
    Collection loading is expensive — cache it for the process lifetime.
    """
    cfg = settings.milvus
    collection = Collection(cfg.collection_name)

    # Load collection into memory — required before search
    # Idempotent: safe if already loaded
    collection.load()

    logger.info(
        LogEvent.APP_STARTUP,
        "Milvus collection loaded into memory",
        collection=cfg.collection_name,
    )

    return collection


# ---------------------------------------------------------------------------
# Search result schema
# ---------------------------------------------------------------------------

class MilvusSearchResult:
    """
    Strongly typed wrapper around a single Milvus hit.
    Decouples downstream code from pymilvus internal types.
    """

    __slots__ = ("product_id", "score", "entity")

    def __init__(
        self,
        product_id: str,
        score: float,
        entity: dict[str, Any],
    ) -> None:
        self.product_id = product_id
        self.score = score        # Cosine similarity [0.0 – 1.0]
        self.entity = entity      # All stored fields for this vector

    def __repr__(self) -> str:
        return (
            f"MilvusSearchResult("
            f"product_id={self.product_id!r}, "
            f"score={self.score:.4f})"
        )


# ---------------------------------------------------------------------------
# Core search — async wrapper over sync pymilvus
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(MilvusException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
async def vector_search(
    query_vector: list[float],
    top_k: int | None = None,
    output_fields: list[str] | None = None,
    filter_expr: str | None = None,
) -> list[MilvusSearchResult]:
    """
    Executes ANN vector search against the Milvus collection.

    Runs the synchronous pymilvus call in a thread pool executor to avoid
    blocking the event loop. pymilvus is not async-native.

    Args:
        query_vector:  Embedding vector from the query encoder.
                       Must match the dimension in EmbeddingSettings.
        top_k:         Number of nearest neighbours to retrieve.
                       Defaults to settings.milvus.top_k (50).
        output_fields: Milvus fields to return alongside the score.
                       Defaults to ["global_product_id", "entity_id", "business_unit_id", "category"].
        filter_expr:   Optional Milvus boolean expression for pre-filtering.
                       e.g. 'entity_id == "entity_001" and business_unit_id == "bu_001" and in_stock == true'
                       Pre-filter reduces candidate set BEFORE ANN —
                       use only for high-selectivity filters (entity_id, business_unit_id, in_stock).
                       Low-selectivity filters (price) → handle in PSQL post-filter.

    Returns:
        List of MilvusSearchResult sorted by score descending (most similar first).

    Raises:
        MilvusException: After 3 retry attempts with exponential backoff.
    """
    cfg = settings.milvus
    effective_top_k = top_k or cfg.top_k
    effective_output_fields = [
        "global_product_id",
        "category_path",
        "bu_ids",
        "updated_at",
    ]

    search_params = {
        "metric_type": cfg.metric_type,
        "params": {"ef": cfg.search_ef},
    }

    async with logger.timed(
        LogEvent.RETRIEVAL_MILVUS_END,
        "milvus_vector_search",
        top_k=effective_top_k,
        has_filter=filter_expr is not None,
    ):
        # Run blocking pymilvus call in thread pool
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _sync_search(
                query_vector=query_vector,
                top_k=effective_top_k,
                output_fields=effective_output_fields,
                search_params=search_params,
                filter_expr=filter_expr,
            ),
        )

    parsed = _parse_results(results)

    logger.info(
        LogEvent.RETRIEVAL_MILVUS_END,
        "Milvus ANN search complete",
        candidates_returned=len(parsed),
        top_k_requested=effective_top_k,
        filter_applied=filter_expr or "none",
    )

    return parsed


def _sync_search(
    query_vector: list[float],
    top_k: int,
    output_fields: list[str],
    search_params: dict[str, Any],
    filter_expr: str | None,
) -> Any:
    """
    Synchronous pymilvus search — runs inside thread pool executor.
    Isolated here so the async layer stays clean.
    """
    collection = _get_collection()

    kwargs: dict[str, Any] = {
        "data": [query_vector],
        "anns_field": "embedding",       # Must match your Milvus schema field name
        "param": search_params,
        "limit": top_k,
        "output_fields": output_fields,
    }

    if filter_expr:
        kwargs["expr"] = filter_expr

    return collection.search(**kwargs)


def _parse_results(raw_results: Any) -> list[MilvusSearchResult]:
    """
    Converts pymilvus SearchResult into clean MilvusSearchResult objects.
    Handles the nested pymilvus result structure safely.
    """
    parsed: list[MilvusSearchResult] = []

    # pymilvus returns results as list of hits per query vector
    # We always send one query vector, so results[0] is our hits
    if not raw_results or len(raw_results) == 0:
        return parsed

    for hit in raw_results[0]:
        try:
            entity = hit.entity.to_dict() if hasattr(hit.entity, "to_dict") else {}
            parsed.append(
                MilvusSearchResult(
                    product_id=str(hit.id),
                    score=float(hit.score),
                    entity=entity,
                )
            )
        except Exception as e:
            # Log and skip malformed hits — don't fail entire search
            logger.warning(
                LogEvent.RETRIEVAL_MILVUS_END,
                "Skipping malformed Milvus hit",
                error=str(e),
                hit_id=getattr(hit, "id", "unknown"),
            )
            continue

    return parsed


# ---------------------------------------------------------------------------
# Milvus filter expression builder
# ---------------------------------------------------------------------------

class MilvusFilterBuilder:
    """
    Builds Milvus boolean filter expressions from structured inputs.

    Only use for HIGH-SELECTIVITY filters that dramatically reduce the
    candidate set before ANN search:
      ✅ entity_id    — one entity vs millions of products
      ✅ in_stock    — typically filters 10-30% of catalog
      ❌ price_range — low selectivity, handled in PSQL post-filter
      ❌ color       — too many values, semantic search handles this better

    Milvus filter expressions use a subset of Python syntax.
    Reference: https://milvus.io/docs/boolean.md
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def business_unit(self, business_unit_id: str):
        self._parts.append(f'array_contains(bu_ids, "{business_unit_id}")')
        return self

    # def entity(self, entity_id: str) -> MilvusFilterBuilder:
    #     self._parts.append(f'entity_id == "{entity_id}"')
    #     return self

    def category(self, category: str):
        self._parts.append(f'category_path like "{category}%"')
        return self

    def build(self) -> str | None:
        """Returns the composed filter expression, or None if no filters added."""
        if not self._parts:
            return None
        return " && ".join(self._parts)