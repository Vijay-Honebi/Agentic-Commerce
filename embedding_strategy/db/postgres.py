from __future__ import annotations

import atexit
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from embedding_strategy.config.settings import get_settings
from embedding_strategy.embedding.composer import GlobalProduct
from embedding_strategy.observability.logger import StructuredLogger, trace_stage

logger = StructuredLogger("db.postgres")
settings = get_settings()


# Sync Watermark

@dataclass(frozen=True)
class SyncWatermark:
    """
    Represents the last successful incremental sync checkpoint.

    last_synced_at : Unix timestamp of the last sync run.
                     The next incremental fetch will query
                     WHERE updated_at > last_synced_at.

    total_synced   : Cumulative count of products synced in
                     this watermark's run — for observability only.
    """
    last_synced_at: int
    total_synced: int


# Connection Pool

class PostgresConnectionPool:
    """
    Manages a ThreadedConnectionPool for safe concurrent access.

    Why ThreadedConnectionPool over a single connection:
    - Ingestion pipeline may parallelize batch fetching in future phases
    - Prevents connection exhaustion under load
    - Automatic connection reuse — no reconnect overhead per batch

    Pool sizing:
    - minconn=2  : Always keep 2 warm connections
    - maxconn=10 : Cap at 10 — ingestion is batch-sequential,
                   not massively concurrent
    """

    _pool: ThreadedConnectionPool | None = None

    @classmethod
    def initialize(cls):
        if cls._pool is not None:
            return

        logger.info(
            "Initializing Postgres connection pool",
            minconn=2,
            maxconn=10,
        )

        cls._pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=settings.postgres_dsn,
            cursor_factory=psycopg2.extras.RealDictCursor,
            # RealDictCursor: rows returned as dicts keyed by column name.
            # Never positional indexing — resilient to column order changes.
            options=f"-c statement_timeout={settings.postgres_statement_timeout_ms}",
            # 30 second statement timeout — batch fetches should never
            # take longer. Prevents runaway queries blocking the pipeline.
        )

        logger.info("Postgres connection pool initialized")

    @classmethod
    @contextmanager
    def get_connection(cls) -> Generator:
        if cls._pool is None:
            cls.initialize()

        conn = cls._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cls._pool.putconn(conn)

    @classmethod
    def close_all(cls):
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            logger.info("Postgres connection pool closed")


# Queries

class ProductQueries:
    """
    Single source of truth for all SQL queries in the ingestion pipeline.

    Schema alignment:
    - Watermark column  : p.modified_at (not updated_at)
    - Attributes        : JSONB_OBJECT_AGG(a.label, s.field_values)
    - Category levels   : category → sub_category → classification (l3 optional)
    - BU membership     : ARRAY_AGG via master_catalogues junction
    """

    FETCH_ALL_PRODUCTS = """
        SELECT
            p.id                                            AS global_product_id,
            p.name                                          AS name,
            ARRAY_AGG(DISTINCT mc.business_unit_id)
                FILTER (WHERE mc.business_unit_id IS NOT NULL)
                                                            AS bu_ids,
            c.name                                          AS category_l1,
            sc.name                                         AS category_l2,
            cls.name                                        AS category_l3,
            b.name                                          AS brand_name,
            COALESCE(
                (
                    SELECT JSONB_OBJECT_AGG(attr_agg.label, attr_agg.values)
                    FROM (
                        SELECT
                            a2.label AS label,
                            STRING_AGG(
                                DISTINCT unnested.val::text,
                                ', ' ORDER BY unnested.val::text
                            ) AS values
                        FROM master_variants v2
                        JOIN specifications s2
                            ON s2.variant_id = v2.variant_id
                        JOIN attributes a2
                            ON a2.id = s2.attribute_id
                        CROSS JOIN LATERAL UNNEST(s2.field_values) AS unnested(val)
                        WHERE v2.product_id = p.id
                        AND a2.label IS NOT NULL
                        AND s2.field_values IS NOT NULL
                        AND unnested.val IS NOT NULL
                        AND unnested.val::text <> ''
                        GROUP BY a2.label
                    ) attr_agg
                ),
                '{}'::jsonb
            )
            ||
            COALESCE(
                (
                    SELECT JSONB_OBJECT_AGG(cond_attr_agg.label, cond_attr_agg.values)
                    FROM (
                        SELECT
                            a3.label AS label,
                            STRING_AGG(
                                DISTINCT unnested.val::text,
                                ', ' ORDER BY unnested.val::text
                            ) AS values
                        FROM master_variants v3
                        JOIN condition_specifications cs
                            ON cs.variant_id = v3.variant_id
                        JOIN attributes a3
                            ON a3.id = cs.condition_attribute_id
                        CROSS JOIN LATERAL UNNEST(cs.field_values) AS unnested(val)
                        WHERE v3.product_id = p.id
                        AND a3.label IS NOT NULL
                        AND cs.field_values IS NOT NULL
                        AND unnested.val IS NOT NULL
                        AND unnested.val::text <> ''
                        GROUP BY a3.label
                    ) cond_attr_agg
                ),
                '{}'::jsonb
            ) AS attributes,
            p.short_description                             AS description,
            EXTRACT(EPOCH FROM p.modified_at)::BIGINT       AS updated_at
        FROM products p
        JOIN master_products mp        ON mp.product_id = p.id
        LEFT JOIN classifications cls       ON cls.id = mp.classification_id
        LEFT JOIN master_catalogues mc      ON mc.id = mp.master_catalogue_id
        LEFT JOIN sub_categories sc         ON sc.id = mp.sub_category_id
        LEFT JOIN categories c              ON c.id = COALESCE(sc.category_id, mp.category_id)
        LEFT JOIN brands b                  ON b.id = p.brand_id
        WHERE
            mp.is_publish_to_customer = True
        GROUP BY
            p.id,
            p.name,
            c.name,
            sc.name,
            cls.name,
            b.name,
            p.short_description,
            p.modified_at
        ORDER BY p.id
        LIMIT %(limit)s
        OFFSET %(offset)s
    """

    FETCH_UPDATED_PRODUCTS = """
        SELECT
            p.id                                            AS global_product_id,
            p.name                                          AS name,
            ARRAY_AGG(DISTINCT mc.business_unit_id)
                FILTER (WHERE mc.business_unit_id IS NOT NULL)
                                                            AS bu_ids,
            c.name                                          AS category_l1,
            sc.name                                         AS category_l2,
            cls.name                                        AS category_l3,
            b.name                                          AS brand_name,
            COALESCE(
                (
                    SELECT JSONB_OBJECT_AGG(attr_agg.label, attr_agg.values)
                    FROM (
                        SELECT
                            a2.label AS label,
                            STRING_AGG(
                                DISTINCT unnested.val::text,
                                ', ' ORDER BY unnested.val::text
                            ) AS values
                        FROM master_variants v2
                        JOIN specifications s2
                            ON s2.variant_id = v2.variant_id
                        JOIN attributes a2
                            ON a2.id = s2.attribute_id
                        CROSS JOIN LATERAL UNNEST(s2.field_values) AS unnested(val)
                        WHERE v2.product_id = p.id
                        AND a2.label IS NOT NULL
                        AND s2.field_values IS NOT NULL
                        AND unnested.val IS NOT NULL
                        AND unnested.val::text <> ''
                        GROUP BY a2.label
                    ) attr_agg
                ),
                '{}'::jsonb
            )
            ||
            COALESCE(
                (
                    SELECT JSONB_OBJECT_AGG(cond_attr_agg.label, cond_attr_agg.values)
                    FROM (
                        SELECT
                            a3.label AS label,
                            STRING_AGG(
                                DISTINCT unnested.val::text,
                                ', ' ORDER BY unnested.val::text
                            ) AS values
                        FROM master_variants v3
                        JOIN condition_specifications cs
                            ON cs.variant_id = v3.variant_id
                        JOIN attributes a3
                            ON a3.id = cs.condition_attribute_id
                        CROSS JOIN LATERAL UNNEST(cs.field_values) AS unnested(val)
                        WHERE v3.product_id = p.id
                        AND a3.label IS NOT NULL
                        AND cs.field_values IS NOT NULL
                        AND unnested.val IS NOT NULL
                        AND unnested.val::text <> ''
                        GROUP BY a3.label
                    ) cond_attr_agg
                ),
                '{}'::jsonb
            ) AS attributes,
            p.short_description                             AS description,
            EXTRACT(EPOCH FROM p.modified_at)::BIGINT       AS updated_at
        FROM products p
        JOIN master_products mp        ON mp.product_id = p.id
        LEFT JOIN classifications cls       ON cls.id = mp.classification_id
        LEFT JOIN master_catalogues mc      ON mc.id = mp.master_catalogue_id
        LEFT JOIN sub_categories sc         ON sc.id = mp.sub_category_id
        LEFT JOIN categories c              ON c.id = COALESCE(sc.category_id, mp.category_id)
        LEFT JOIN brands b                  ON b.id = p.brand_id
        WHERE
            mp.is_publish_to_customer = True AND EXTRACT(EPOCH FROM p.modified_at)::BIGINT > %(since_timestamp)s
        GROUP BY
            p.id,
            p.name,
            c.name,
            sc.name,
            cls.name,
            b.name,
            p.short_description,
            p.modified_at
        ORDER BY p.id
        LIMIT %(limit)s
        OFFSET %(offset)s
    """

    COUNT_ALL_PRODUCTS = """
        SELECT COUNT(*) AS total
        FROM products
    """

    COUNT_UPDATED_PRODUCTS = """
        SELECT COUNT(*) AS total
        FROM products
        WHERE EXTRACT(EPOCH FROM modified_at)::BIGINT > %(since_timestamp)s
    """

    UPSERT_SYNC_WATERMARK = """
        INSERT INTO embedding_sync_watermarks (
            collection_name,
            last_synced_at,
            total_synced,
            synced_at
        )
        VALUES (
            %(collection_name)s,
            %(last_synced_at)s,
            %(total_synced)s,
            NOW()
        )
        ON CONFLICT (collection_name)
        DO UPDATE SET
            last_synced_at = EXCLUDED.last_synced_at,
            total_synced   = EXCLUDED.total_synced,
            synced_at      = EXCLUDED.synced_at
    """

    FETCH_SYNC_WATERMARK = """
        SELECT last_synced_at, total_synced
        FROM embedding_sync_watermarks
        WHERE collection_name = %(collection_name)s
    """


# Row Mapper

class ProductRowMapper:
    """
    Maps a raw PSQL RealDictRow to a typed GlobalProduct dataclass.

    Isolated here so:
    - Query changes never touch business logic
    - Null handling is centralized and explicit
    - Easy to unit test with mock dicts
    """

    # Inline import to avoid circular dependency with composer
    @staticmethod
    def map(row: dict) -> GlobalProduct:
        return GlobalProduct(
            global_product_id=str(row["global_product_id"]),
            name=ProductRowMapper._clean(row.get("name")),
            bu_ids=ProductRowMapper._clean_array(row.get("bu_ids")),
            category_l1=ProductRowMapper._clean(row.get("category_l1")),
            category_l2=ProductRowMapper._clean(row.get("category_l2")),
            category_l3=ProductRowMapper._clean(row.get("category_l3")),
            brand_name=ProductRowMapper._clean(row.get("brand_name")),
            attributes=ProductRowMapper._clean_attributes(row.get("attributes")),
            description=ProductRowMapper._clean(row.get("description")),
            updated_at=int(row["updated_at"]),
        )

    @staticmethod
    def _clean_attributes(value) -> dict[str, str]:
        """
        Cleans JSONB_OBJECT_AGG result into a flat dict[str, str].

        psycopg2 with RealDictCursor deserializes JSONB columns
        automatically into Python dicts — no manual json.loads needed.

        Handles:
        - None / missing → empty dict
        - Non-string values → cast to str (handles int, float, bool specs)
        - Empty string keys or values → excluded
        """
        if not value:
            return {}
        if isinstance(value, str):
            # Fallback: if for any reason psycopg2 returns raw JSON string
            import json
            try:
                value = json.loads(value)
            except Exception:
                return {}
        return {
            str(k).strip(): str(v).strip()
            for k, v in value.items()
            if k and str(k).strip() and v and str(v).strip()
        }

    @staticmethod
    def _clean(value: Optional[str]) -> Optional[str]:
        """
        Strips whitespace and returns None for empty/null values.
        Ensures the EmbeddingComposer never sees empty strings.
        """
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped if stripped else None
    
    @staticmethod
    def _clean_array(value) -> list[str]:
        """
        Cleans a PSQL array result into a list of non-empty strings.
        ARRAY_AGG returns a Python list via psycopg2 — never None
        if JOIN is INNER, but guard anyway.
        """
        if not value:
            return []
        return [str(v).strip() for v in value if v and str(v).strip()]


# Product Fetcher

class GlobalProductFetcher:
    """
    Fetches GlobalProduct records from PSQL in controlled batches.

    Two fetch modes:
    - fetch_all_batched()      : Full catalog ingestion (initial load)
    - fetch_updated_batched()  : Incremental sync (daily delta)

    Both are generators — they yield one batch at a time.
    The pipeline never loads the full catalog into memory.

    Batch size is driven by settings.ingestion_batch_size (default: 500).
    Each batch is a list[GlobalProduct] ready for the embedding pipeline.
    """

    def __init__(self):
        self._batch_size = settings.ingestion_batch_size
        self._mapper = ProductRowMapper()

    def get_total_product_count(self) -> int:
        with PostgresConnectionPool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ProductQueries.COUNT_ALL_PRODUCTS)
                result = cur.fetchone()
                total = int(result["total"])   # ← fetch only, no return here

        # Test limit cap — now reachable
        if settings.test_product_limit is not None:
            capped = min(total, settings.test_product_limit)
            logger.info(
                "Test product limit applied",
                actual_total=total,
                capped_to=capped,
                test_mode=True,
            )
            return capped

        logger.info("Total product count fetched", total=total)
        return total

    def get_updated_product_count(self, since_timestamp: int) -> int:
        with PostgresConnectionPool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    ProductQueries.COUNT_UPDATED_PRODUCTS,
                    {"since_timestamp": since_timestamp},
                )
                result = cur.fetchone()
                total = int(result["total"])   # ← no return here

        # Cap logic is already outside the with block — correct
        if settings.test_product_limit is not None:
            capped = min(total, settings.test_product_limit)
            logger.info(
                "Test product limit applied on incremental",
                actual_total=total,
                capped_to=capped,
                test_mode=True,
            )
            return capped

        logger.info(
            "Updated product count fetched",
            since_timestamp=since_timestamp,
            total=total,
        )
        return total

    @trace_stage("psql_fetch_all_batched")
    def fetch_all_batched(self) -> Generator[list, None, None]:
        """
        Yields batches of GlobalProduct for full catalog ingestion.
        Logs progress every batch with offset/total for monitoring.
        """
        total = self.get_total_product_count()
        offset = 0
        batch_number = 0
        total_yielded = 0

        logger.info(
            "Starting full catalog fetch",
            total_products=total,
            batch_size=self._batch_size,
            estimated_batches=max(1, total // self._batch_size),
        )

        while offset < total:
            batch = self._fetch_batch(
                query=ProductQueries.FETCH_ALL_PRODUCTS,
                params={"limit": self._batch_size, "offset": offset},
            )

            if not batch:
                break

            batch_number += 1
            total_yielded += len(batch)

            logger.info(
                "Batch fetched",
                batch_number=batch_number,
                batch_size=len(batch),
                offset=offset,
                total=total,
                progress_pct=round((offset / total) * 100, 1),
            )

            yield batch
            offset += self._batch_size

            # Hard stop guard
            # Safety net: never exceed total regardless of batch math
            if total_yielded >= total:
                logger.info(
                    "Test limit reached — stopping fetch",
                    total_yielded=total_yielded,
                    limit=total,
                )
                break

        logger.info(
            "Full catalog fetch complete",
            total_batches=batch_number,
            total_products=total_yielded,
        )

    @trace_stage("psql_fetch_updated_batched")
    def fetch_updated_batched(
        self, since_timestamp: int
    ) -> Generator[list, None, None]:
        """
        Yields batches of GlobalProduct updated after since_timestamp.
        Used exclusively by the incremental daily sync.
        """
        total = self.get_updated_product_count(since_timestamp)

        if total == 0:
            logger.info(
                "No updated products found",
                since_timestamp=since_timestamp,
            )
            return

        offset = 0
        batch_number = 0

        logger.info(
            "Starting incremental fetch",
            since_timestamp=since_timestamp,
            total_updated=total,
            batch_size=self._batch_size,
        )

        while offset < total:
            batch = self._fetch_batch(
                query=ProductQueries.FETCH_UPDATED_PRODUCTS,
                params={
                    "since_timestamp": since_timestamp,
                    "limit": self._batch_size,
                    "offset": offset,
                },
            )

            if not batch:
                break

            batch_number += 1
            logger.info(
                "Incremental batch fetched",
                batch_number=batch_number,
                batch_size=len(batch),
                offset=offset,
                total=total,
            )

            yield batch
            offset += self._batch_size

        logger.info(
            "Incremental fetch complete",
            total_batches=batch_number,
            total_updated=total,
        )

    def _fetch_batch(self, query: str, params: dict) -> list:
        """
        Executes a single batch query and maps rows to GlobalProduct instances.
        Any row-level mapping failure is logged and skipped —
        one bad row must never abort the entire batch.
        """
        with PostgresConnectionPool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        products = []
        for row in rows:
            try:
                products.append(self._mapper.map(dict(row)))
            except Exception as e:
                logger.error(
                    "Row mapping failed — skipping",
                    row_id=row.get("global_product_id", "unknown"),
                    error=str(e),
                    error_type=type(e).__name__,
                )

        return products


# Watermark Repository

class SyncWatermarkRepository:
    """
    Persists and retrieves incremental sync watermarks in PSQL.

    Why store watermarks in PSQL (not a file or Redis):
    - Transactional: watermark update and product write are in the same DB
    - Durable: survives restarts, container recycles, and infra events
    - Auditable: full history of sync runs via synced_at column
    - No external dependency: no Redis, no filesystem assumptions

    Watermark tracks p.modified_at from the products table.
    Column: products.modified_at (TIMESTAMPTZ)
    Stored as: Unix epoch (BIGINT) in embedding_sync_watermarks.last_synced_at

    Required PSQL table (run once as migration):

        CREATE TABLE embedding_sync_watermarks (
            collection_name VARCHAR(128) PRIMARY KEY,
            last_synced_at  BIGINT       NOT NULL,
            total_synced    INTEGER      NOT NULL DEFAULT 0,
            synced_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        );
    """

    def __init__(self):
        self._collection_name = settings.milvus_collection_name

    def get_watermark(self) -> Optional[SyncWatermark]:
        """
        Returns the last watermark or None if this is the first sync run.
        Callers must handle None and default to full ingestion.
        """
        with PostgresConnectionPool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    ProductQueries.FETCH_SYNC_WATERMARK,
                    {"collection_name": self._collection_name},
                )
                row = cur.fetchone()

        if not row:
            logger.info(
                "No watermark found — first sync run",
                collection=self._collection_name,
            )
            return None

        watermark = SyncWatermark(
            last_synced_at=int(row["last_synced_at"]),
            total_synced=int(row["total_synced"]),
        )

        logger.info(
            "Watermark loaded",
            collection=self._collection_name,
            last_synced_at=watermark.last_synced_at,
            total_synced=watermark.total_synced,
        )

        return watermark

    @trace_stage("watermark_save")
    def save_watermark(self, last_synced_at: int, total_synced: int):
        """
        Upserts the watermark after a successful sync run.
        Must be called ONLY after Milvus upsert is confirmed successful.
        """
        with PostgresConnectionPool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    ProductQueries.UPSERT_SYNC_WATERMARK,
                    {
                        "collection_name": self._collection_name,
                        "last_synced_at": last_synced_at,
                        "total_synced": total_synced,
                    },
                )

        logger.info(
            "Watermark saved",
            collection=self._collection_name,
            last_synced_at=last_synced_at,
            total_synced=total_synced,
        )
# ```

# ---

# ## What This Step Establishes

# Four clean, separated responsibilities:
# ```
# PostgresConnectionPool      → Connection lifecycle, pooling, context manager
# ProductQueries              → All SQL in one place — zero SQL anywhere else
# GlobalProductFetcher        → Batch generator — never loads full catalog in memory
# SyncWatermarkRepository     → Checkpoint persistence — durable, transactional


# Three decisions:
# Generator pattern on fetchers — fetch_all_batched() and fetch_updated_batched() are generators. The pipeline never holds more than 500 products in memory regardless of catalog size. This scales to 10M products without change.
# Row-level fault isolation — a single malformed row in a batch of 500 logs an error and skips. It never aborts the batch or the pipeline. One bad product cannot block 499 good ones.
# Watermark saved after Milvus confirms — the ordering contract is strict. PSQL watermark is updated only after Milvus upsert succeeds. If Milvus fails mid-batch, the next run re-processes from the last safe checkpoint — no silent data loss.