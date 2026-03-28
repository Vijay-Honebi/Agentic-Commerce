from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pymilvus import Collection, MilvusException

from embedding_strategy.config.settings import get_settings
from embedding_strategy.observability.logger import StructuredLogger, trace_stage

logger = StructuredLogger("embedding.repository")
settings = get_settings()


# Data Contracts

@dataclass(frozen=True)
class ProductVector:
    """
    The exact unit of data written to and read from Milvus.

    This is the contract between the ingestion pipeline and
    the vector store. Nothing outside this dataclass is
    ever written to Milvus products_collection.

    Fields map 1:1 to the schema defined in schema.py.
    Any schema change must be reflected here first.
    """
    global_product_id: str
    embedding: list[float]          # 512-dim, pre-validated before insert
    category_path: str
    bu_ids: list[str]
    updated_at: int                 # Unix timestamp


@dataclass(frozen=True)
class VectorSearchRequest:
    """
    Input contract for a semantic product search.

    query_vector    : 512-dim embedding of the user's query.
    bu_id           : Required. Every search is BU-scoped.
    category_path   : Optional. Narrows ANN search to a category subtree.
    top_k           : Number of candidates to return from Milvus.
                      Deliberately larger than final result set —
                      downstream PSQL + business ranking will reduce this.
    ef              : HNSW query-time search depth.
                      Higher = better recall, higher latency.
                      Tunable per request if needed.
    """
    query_vector: list[float]
    bu_ids: list[str]
    category_path: Optional[str] = None
    top_k: int = 50                 # Return 50 candidates to PSQL stage
    ef: int = 64                    # HNSW ef — matches HNSWIndexConfig.SEARCH_PARAMS


@dataclass(frozen=True)
class VectorSearchResult:
    """
    A single candidate returned from Milvus semantic search.

    global_product_id : Bridge key — passed to PSQL for full product fetch.
    score             : Cosine similarity [0.0, 1.0]. Higher = more similar.
    category_path     : Returned for observability and adaptive relaxation logic.
    rank              : 1-based position in Milvus result set (before re-ranking).
    """
    global_product_id: str
    score: float
    category_path: str
    rank: int


# Insert Payload Builder

class MilvusInsertPayloadBuilder:
    """
    Converts a list[ProductVector] into the columnar format
    Milvus expects for batch insert.

    Milvus insert format:
        {
            "field_name": [value_for_row_0, value_for_row_1, ...],
            ...
        }

    This is NOT row-oriented. Each field is a list of all values
    across the batch. This class owns that transformation.
    """

    @staticmethod
    def build(vectors: list[ProductVector]) -> list[list]:
        """
        Returns Milvus-ready columnar insert data.
        Column order must match schema field definition order exactly.
        """
        return [
            [v.global_product_id for v in vectors],   # global_product_id
            [v.embedding for v in vectors],            # embedding
            [v.category_path for v in vectors],        # category_path
            [v.bu_ids for v in vectors],                # bu_ids
            [v.updated_at for v in vectors],           # updated_at
        ]


# Vector Validator

class ProductVectorValidator:
    """
    Validates ProductVector instances before they reach Milvus.

    Validation is a strict gate — invalid vectors are rejected
    and logged. They are NEVER silently inserted with bad data.

    Catches:
    - Wrong embedding dimensions (schema mismatch)
    - Empty or null required fields
    - Non-finite float values in embedding (NaN, Inf)
    """

    EXPECTED_DIM = settings.embedding_dimensions   # 512

    @classmethod
    def validate_batch(
        cls, vectors: list[ProductVector]
    ) -> tuple[list[ProductVector], list[str]]:
        """
        Returns (valid_vectors, rejected_ids).
        Caller decides whether rejected_ids warrant an alert.
        """
        valid = []
        rejected = []

        for v in vectors:
            error = cls._validate_one(v)
            if error:
                logger.warning(
                    "ProductVector rejected",
                    global_product_id=v.global_product_id,
                    reason=error,
                )
                rejected.append(v.global_product_id)
            else:
                valid.append(v)

        if rejected:
            logger.warning(
                "Batch validation summary",
                total=len(vectors),
                valid=len(valid),
                rejected=len(rejected),
                rejected_ids=rejected,
            )

        return valid, rejected

    @classmethod
    def _validate_one(cls, v: ProductVector) -> Optional[str]:
        if not v.global_product_id or not v.global_product_id.strip():
            return "global_product_id is empty"

        if not v.bu_ids:
            return "bu_ids is empty — product has no BU assignment"

        if not all(isinstance(b, str) and b.strip() for b in v.bu_ids):
            return "bu_ids contains empty or non-string elements"

        if not v.category_path or not v.category_path.strip():
            return "category_path is empty"

        if not v.embedding:
            return "embedding is empty"

        if len(v.embedding) != cls.EXPECTED_DIM:
            return (
                f"embedding dimension mismatch: "
                f"expected {cls.EXPECTED_DIM}, got {len(v.embedding)}"
            )

        if not all(isinstance(x, float) and _is_finite(x) for x in v.embedding):
            return "embedding contains non-finite values (NaN or Inf)"

        return None


def _is_finite(value: float) -> bool:
    import math
    return not (math.isnan(value) or math.isinf(value))


# Repository

class ProductVectorRepository:
    """
    The ONLY interface between the application and Milvus
    products_collection.

    Responsibilities:
    - Batched upsert with validation
    - Semantic search with filter expression building
    - Flush lifecycle management
    - Collection stats for observability

    Design rules:
    - Never exposes raw Milvus Collection to callers
    - Never builds filter expressions outside _build_search_filter()
    - Never swallows MilvusException silently — always re-raises
      after structured logging
    - Upsert is idempotent — safe to re-run on same product IDs
      (Milvus upsert on primary key = update if exists, insert if not)
    """

    def __init__(self, collection: Collection):
        self._collection = collection
        self._validator = ProductVectorValidator()
        self._payload_builder = MilvusInsertPayloadBuilder()

    # Write

    @trace_stage("milvus_upsert_batch")
    def upsert_batch(self, vectors: list[ProductVector]) -> UpsertResult:
        """
        Validates, then upserts a batch of ProductVectors into Milvus.

        Returns UpsertResult with counts for pipeline observability.
        Raises MilvusException on storage failure after logging.

        Why upsert (not insert):
        - Incremental sync re-processes products that already exist
        - Upsert on primary key guarantees idempotency
        - No need to pre-check existence before writing
        """
        if not vectors:
            return UpsertResult(attempted=0, inserted=0, rejected=0, rejected_ids=[])

        valid_vectors, rejected_ids = self._validator.validate_batch(vectors)

        if not valid_vectors:
            logger.error(
                "Entire batch rejected by validator",
                attempted=len(vectors),
                rejected=len(rejected_ids),
            )
            return UpsertResult(
                attempted=len(vectors),
                inserted=0,
                rejected=len(rejected_ids),
                rejected_ids=rejected_ids,
            )

        try:
            payload = self._payload_builder.build(valid_vectors)
            mutation_result = self._collection.upsert(payload)

            result = UpsertResult(
                attempted=len(vectors),
                inserted=len(valid_vectors),
                rejected=len(rejected_ids),
                rejected_ids=rejected_ids,
            )

            logger.info(
                "Milvus upsert successful",
                attempted=result.attempted,
                inserted=result.inserted,
                rejected=result.rejected,
                milvus_insert_count=mutation_result.insert_count,
                milvus_delete_count=mutation_result.delete_count,
            )

            return result

        except MilvusException as e:
            logger.error(
                "Milvus upsert failed",
                error=str(e),
                error_code=e.code,
                batch_size=len(valid_vectors),
            )
            raise

    @trace_stage("milvus_flush")
    def flush(self):
        """
        Flushes buffered segments to persistent storage.

        When to call:
        - After the final batch of a full ingestion run
        - After the final batch of an incremental sync run
        - NOT after every batch — flush is expensive

        Flush seals the current segment and makes data
        queryable by scalar filters (not just ANN search).
        """
        self._collection.flush()
        row_count = self._collection.num_entities

        logger.info(
            "Milvus flush complete",
            collection=settings.milvus_collection_name,
            row_count_after_flush=row_count,
        )

    # Read

    @trace_stage("milvus_semantic_search")
    def search(self, request: VectorSearchRequest) -> list[VectorSearchResult]:
        """
        Executes ANN search against the products_collection.

        Pipeline position:
            User Query → LLM → Embedding → HERE → PSQL → Ranking

        Returns top_k candidates as VectorSearchResult list.
        These are candidate IDs only — full product data is
        fetched from PSQL in the next pipeline stage.

        Filter expression is built dynamically based on request.
        BU scoping is ALWAYS applied — it is never optional.
        """
        filter_expr = self._build_search_filter(request)

        logger.info(
            "Semantic search initiated",
            bu_id=request.bu_id,
            category_path=request.category_path,
            top_k=request.top_k,
            ef=request.ef,
            filter_expr=filter_expr,
        )

        try:
            results = self._collection.search(
                data=[request.query_vector],
                anns_field="embedding",
                param={
                    "metric_type": "COSINE",
                    "params": {"ef": request.ef},
                },
                limit=request.top_k,
                expr=filter_expr,
                output_fields=["global_product_id", "category_path"],
                # Only fetch what the next pipeline stage needs.
                # Never fetch embedding back — wasteful and unused.
            )

            candidates = self._map_search_results(results)

            logger.info(
                "Semantic search complete",
                returned=len(candidates),
                top_score=candidates[0].score if candidates else None,
                bottom_score=candidates[-1].score if candidates else None,
            )

            return candidates

        except MilvusException as e:
            logger.error(
                "Milvus search failed",
                error=str(e),
                error_code=e.code,
                bu_id=request.bu_id,
            )
            raise

    def get_by_ids(self, product_ids: list[str]) -> list[dict]:
        """
        Point lookup by primary key list.

        Use cases:
        - Verify specific products are indexed
        - Debugging and observability tooling
        - NOT for search — use search() for that
        """
        if not product_ids:
            return []

        id_list = ", ".join(f'"{pid}"' for pid in product_ids)
        expr = f"global_product_id in [{id_list}]"

        try:
            results = self._collection.query(
                expr=expr,
                output_fields=[
                    "global_product_id",
                    "category_path",
                    "bu_ids",
                    "updated_at",
                ],
            )
            logger.debug(
                "Point lookup complete",
                requested=len(product_ids),
                found=len(results),
            )
            return results

        except MilvusException as e:
            logger.error(
                "Milvus point lookup failed",
                error=str(e),
                product_ids=product_ids,
            )
            raise

    # Filter Builder

    @staticmethod
    def _build_search_filter(request: VectorSearchRequest) -> str:
        """
        Builds a Milvus boolean filter expression from a VectorSearchRequest.

        Rules:
        - bu_id filter is ALWAYS present — never optional
        - category_path filter uses 'like' prefix match to capture
          all subcategories under a given path segment
          e.g. "Shoes > Sports" matches "Shoes > Sports > Badminton"
               and "Shoes > Sports > Running"
        - Expressions are AND-joined — all conditions must be satisfied

        Milvus filter syntax reference:
        https://milvus.io/docs/boolean.md
        """
        expressions = [f'array_contains(bu_ids, "{request.bu_id}")']

        if request.category_path:
            # Prefix match to capture entire category subtrees.
            # "Shoes > Sports" like pattern matches all children.
            safe_path = request.category_path.replace('"', '\\"')
            expressions.append(f'category_path like "{safe_path}%"')

        return " && ".join(expressions)

    # Result Mapper

    @staticmethod
    def _map_search_results(
        raw_results,
    ) -> list[VectorSearchResult]:
        """
        Maps raw Milvus SearchResult to typed VectorSearchResult list.
        Milvus returns a list of hits per query vector.
        We always send one query vector — so we consume index 0.
        """
        candidates = []

        hits = raw_results[0] if raw_results else []

        for rank, hit in enumerate(hits, start=1):
            candidates.append(
                VectorSearchResult(
                    global_product_id=hit.fields.get("global_product_id"),
                    score=round(float(hit.score), 6),
                    category_path=hit.fields.get("category_path", ""),
                    rank=rank,
                )
            )

        return candidates


# Result Types

@dataclass(frozen=True)
class UpsertResult:
    """
    Returned by upsert_batch() for pipeline observability.
    Carries enough information to decide whether to alert.

    rejected_rate > 0.05 (5%) should trigger an alert in production.
    """
    attempted: int
    inserted: int
    rejected: int
    rejected_ids: list[str]

    @property
    def rejected_rate(self) -> float:
        if self.attempted == 0:
            return 0.0
        return round(self.rejected / self.attempted, 4)

    @property
    def is_clean(self) -> bool:
        """True if zero rejections."""
        return self.rejected == 0
# ```

# ---

# ## What This Step Establishes

# Five clean, separated responsibilities:
# ```
# ProductVector               → Write contract: exactly what enters Milvus
# VectorSearchRequest         → Read contract: exactly what a search needs
# VectorSearchResult          → Read output: exactly what the pipeline receives
# ProductVectorValidator      → Guards the write path — bad data never enters
# ProductVectorRepository     → The single Milvus interface — no raw Collection elsewhere


# Four decisions:
# _build_search_filter uses prefix like on category_path — "Shoes > Sports" matches "Shoes > Sports > Badminton" and "Shoes > Sports > Running". This means a broad category query naturally includes all subcategories without the caller needing to enumerate them. This is the correct behavior for discovery.
# output_fields never includes embedding — returning the 512-dim vector on search results is wasteful. The embedding is written once and only used internally for ANN. Callers only need the ID and score to proceed to PSQL.
# flush() is called once per run, not per batch — flushing after every 500-record batch would make ingestion 100x slower. One flush at the end of the full run seals all segments cleanly.
# UpsertResult.rejected_rate — the pipeline can inspect this after every batch. Anything above 5% is a signal that the embedding composer or PSQL mapper has a systematic problem worth alerting on.