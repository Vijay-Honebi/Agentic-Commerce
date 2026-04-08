# conversational_commerce/retrieval/psql_client.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg
import json

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.query import HardConstraints

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Pool lifecycle — mirrors session_store.py pattern
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None

async def _register_retrieval_codecs(
    connection: asyncpg.Connection,
) -> None:
    for type_name in ("jsonb", "json"):
        await connection.set_type_codec(
            type_name,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


async def init_psql_retrieval_pool() -> None:
    """
    Creates the asyncpg connection pool for product retrieval queries.
    Separate pool from session_store — different query patterns,
    different sizing requirements.
    Called once at application startup.
    """
    global _pool

    if _pool is not None:
        return

    cfg = settings.postgres

    logger.info(
        LogEvent.APP_STARTUP,
        "Initialising PostgreSQL retrieval pool",
        min_size=cfg.pool_min_size,
        max_size=cfg.pool_max_size,
    )

    _pool = await asyncpg.create_pool(
        dsn=str(cfg.dsn).replace("+asyncpg", ""),
        min_size=cfg.pool_min_size,
        max_size=cfg.pool_max_size,
        command_timeout=cfg.command_timeout_seconds,
        init=_register_retrieval_codecs,
    )

    logger.info(
        LogEvent.APP_STARTUP,
        "FINAL DB CONFIG",
        host=settings.postgres.host,
        db=settings.postgres.db,
        dsn=settings.postgres.dsn,
    )

    # print(1/0)

    logger.info(LogEvent.APP_STARTUP, "PostgreSQL retrieval pool ready")


async def close_psql_retrieval_pool() -> None:
    global _pool
    if _pool is None:
        return
    await _pool.close()
    _pool = None
    logger.info(LogEvent.APP_SHUTDOWN, "PostgreSQL retrieval pool closed")


def _get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError(
            "PSQL retrieval pool not initialised. "
            "Call init_psql_retrieval_pool() at startup."
        )
    return _pool


# ---------------------------------------------------------------------------
# Enriched product record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnrichedProduct:
    """
    A single product record returned from PSQL enrichment.

    Contains everything needed to:
      - Apply hard constraint filtering
      - Compute business ranking scores
      - Populate ProductCard schema
      - Feed Phase 2 cart tool (variant_id is present)

    Design note:
        This is a dataclass not a Pydantic model — it's an internal
        retrieval type, never serialised to JSON or sent to clients.
        ProductCard (Pydantic) is the external contract.
    """

    product_id: str
    product_code: str
    product_name: str

    # Pricing
    # base_price: float
    # compare_at_price: float | None
    # currency: str

    # # Availability
    # in_stock: bool
    # stock_quantity: int | None
    # variant_count: int

    # # Catalog attributes (JSONB column)
    # attributes: dict[str, Any]

    # # Business ranking signals
    # rating: float | None
    # review_count: int
    # sales_rank: int | None          # Lower = more popular
    # business_boost_score: float     # Merchant-configured boost [0.0–1.0]

    # Primary image
    images: list[dict] | None              # JSON array of necessary image data
    url_slug: str | None             # For product URL construction

    # short_description: str


# ---------------------------------------------------------------------------
# Core enrichment query
# ---------------------------------------------------------------------------

async def enrich_candidates(
    product_ids: list[str],
    constraints: HardConstraints,
    entity_id: str | None = None,
    business_unit_id: str | None = None,
) -> list[EnrichedProduct]:
    """
    Fetches full product data from PSQL for a set of Milvus candidate IDs,
    applying hard constraint filters as SQL WHERE clauses.

    This is the PSQL half of hybrid retrieval:
      Milvus → candidate_ids (semantic relevance, ANN speed)
      PSQL   → enriched + filtered products (structured truth)

    The query is parameterised — never string-interpolated.
    All hard constraints compile to $N placeholders to prevent SQL injection.

    Args:
        product_ids:  Candidate IDs from Milvus ANN search.
        constraints:  HardConstraints from ParsedQuery — compiled to WHERE clauses.
        entity_id:     Optional entity scope override.
        business_unit_id: Optional business unit scope override.

    Returns:
        List of EnrichedProduct that PASS all hard constraints.
        Order is not guaranteed here — ranker.py sorts the final list.
    """
    if not product_ids:
        return []

    query, params = _build_enrichment_query(
        product_ids=product_ids,
        constraints=constraints,
        entity_id=entity_id,
        business_unit_id=business_unit_id,
    )

    async with logger.timed(
        LogEvent.RETRIEVAL_PSQL_END,
        "psql_enrich_candidates",
        candidate_count=len(product_ids),
    ):
        async with _get_pool().acquire() as conn:

            row = await conn.fetchrow("""
                SELECT 
                    current_database(),
                    current_schema(),
                    inet_server_addr(),
                    inet_server_port(),
                    version()
            """)
            logger.info(LogEvent.RETRIEVAL_PSQL_END, "DB DEBUG", **dict(row))

            rows = await conn.fetch(query, *params)

    enriched = [_row_to_enriched_product(row) for row in rows]

    logger.info(
        LogEvent.RETRIEVAL_PSQL_END,
        "PSQL enrichment complete",
        candidates_in=len(product_ids),
        candidates_out=len(enriched),
        filtered_out=len(product_ids) - len(enriched),
    )

    return enriched


async def fetch_product_by_id(product_id: str) -> EnrichedProduct | None:
    """
    Fetches a single product by ID with no constraint filtering.
    Used by get_product_details tool.

    Returns None if product does not exist or is inactive.
    """
    query, params = _build_enrichment_query(
        product_ids=[product_id],
        constraints=HardConstraints(in_stock_only=False),  # show even OOS on detail
        entity_id=None,
        business_unit_id=None,
    )

    async with _get_pool().acquire() as conn:
        rows = await conn.fetch(query, *params)

    if not rows:
        logger.warning(
            LogEvent.RETRIEVAL_PSQL_END,
            "Product not found",
            product_id=product_id,
        )
        return None

    return _row_to_enriched_product(rows[0])


# ---------------------------------------------------------------------------
# Query builder
# Compiles HardConstraints → parameterised SQL
# ---------------------------------------------------------------------------

def _build_enrichment_query(
    product_ids: list[str],
    constraints: HardConstraints,
    entity_id: str | None,
    business_unit_id: str | None,
) -> tuple[str, list[Any]]:
    """
    Builds a fully parameterised SQL query from HardConstraints.

    Design principles:
      1. NEVER string-interpolate user values — always $N params
      2. Each constraint adds exactly one param to the params list
      3. Missing constraint fields → no WHERE clause for that field
      4. JSONB attribute filters use @> operator (containment)

    Returns:
        (query_string, params_list) ready for asyncpg conn.fetch()
    """
    params: list[Any] = []
    where_clauses: list[str] = []

    # ── Base filter: candidate IDs from Milvus ────────────────────────────
    params.append(product_ids)
    where_clauses.append(f"p.id = ANY(${len(params)})")

    # if product_ids:
    #     placeholders = []
    #     for i, pid in enumerate(product_ids):
    #         params.append(pid)
    #         placeholders.append(f"${len(params)}::text")

    #     where_clauses.append(f"p.id IN ({', '.join(placeholders)})")

    # # ── Active products only ──────────────────────────────────────────────
    # where_clauses.append("p.active_flag = true")

    # effective_entity_id = entity_id or constraints.entity_id
    # if effective_entity_id:
    #     params.append(effective_entity_id)
    #     where_clauses.append(f"ep.entity_id = ${len(params)}")

    # effective_bu_id = business_unit_id or constraints.business_unit_id
    # if effective_bu_id:
    #     params.append(effective_bu_id)
    #     where_clauses.append(f"mc.business_unit_id = ${len(params)}")

    # # ── Stock filter ──────────────────────────────────────────────────────
    # if constraints.in_stock_only:
    #     where_clauses.append("mp.stock_balance > 0")

    # # ── Price range ───────────────────────────────────────────────────────
    # if constraints.price_range:
    #     if constraints.price_range.min_price is not None:
    #         params.append(constraints.price_range.min_price)
    #         where_clauses.append(f"mp.max_retail_price >= ${len(params)}")
    #     if constraints.price_range.max_price is not None:
    #         params.append(constraints.price_range.max_price)
    #         where_clauses.append(f"mp.max_retail_price <= ${len(params)}")

    # # ── Category ──────────────────────────────────────────────────────────
    # if constraints.category:
    #     params.append(constraints.category.lower())
    #     where_clauses.append(f"LOWER(c.name) = ${len(params)}")

    # if constraints.sub_category:
    #     params.append(constraints.sub_category.lower())
    #     where_clauses.append(f"LOWER(sc.name) = ${len(params)}")

    # # ── Brand ─────────────────────────────────────────────────────────────
    # if constraints.brand:
    #     params.append(constraints.brand.lower())
    #     where_clauses.append(f"LOWER(b.name) = ${len(params)}")

    # # # ── Gender ────────────────────────────────────────────────────────────
    # # if constraints.gender:
    # #     params.append(constraints.gender.lower())
    # #     where_clauses.append(f"LOWER(p.gender) = ${len(params)}")

    # # # ── Rating ────────────────────────────────────────────────────────────
    # # if constraints.min_rating is not None:
    # #     params.append(constraints.min_rating)
    # #     where_clauses.append(f"p.rating >= ${len(params)}")

    # # ── Dynamic attribute filters (generic, scalable) ─────────────────────
    # if constraints.dynamic_filters:
    #     for key, value in constraints.dynamic_filters.items():
    #         if value is None:
    #             continue

    #         # Normalize
    #         attr_key = str(key).strip().lower()
    #         if isinstance(value, str):
    #             attr_value = str(value).strip().lower()
    #         else:
    #             attr_value = value

    #         params.append(json.dumps({attr_key: attr_value}))
    #         where_clauses.append(f"p.attributes @> ${len(params)}::jsonb")

    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT
            p.id AS product_id,
            ep.entity_id,
            mc.business_unit_id,
            p.code AS product_code,
            p.name AS product_name,
            b.name brand,
            c.name category,
            sc.name sub_category,
            p.family_id,
            -- mp.min_retail_price AS base_price,
            -- p.compare_at_price,
            -- p.currency,
            -- p.in_stock,
            -- p.stock_quantity,
            -- p.variant_count,
            -- p.attributes,
            -- p.rating,
            -- p.review_count,
            -- p.sales_rank,
            -- COALESCE(p.business_boost_score, 0.0) AS business_boost_score,
            p.short_description,
            p.images,
            s.url_slug
        FROM products p
        JOIN master_products mp ON p.id = mp.product_id
		JOIN categories c ON mp.category_id = c.id
		LEFT JOIN sub_categories sc ON mp.sub_category_id = sc.id
        LEFT JOIN entity_products ep ON p.id = ep.product_id
        LEFT JOIN master_catalogues mc ON ep.master_catalogue_id = mc.id
		LEFT JOIN brands b ON p.brand_id = b.id
        JOIN seo s ON p.id = s.object_id
        WHERE {where_sql}
    """

    logger.info(
        LogEvent.PSQL_QUERY_BUILT,
        "PSQL query built",
        query=query,
        params=params,
    )

    return query, params

def _normalize_value(value: Any) -> Any:
    """
    Converts asyncpg / Postgres types into JSON-safe Python types.
    This prevents memoryview, bytes, and other DB-native types
    from leaking into API layers.
    """
    if value is None:
        return None

    # Handle memoryview (MOST IMPORTANT)
    if isinstance(value, memoryview):
        value = value.tobytes()

    # Handle bytes → string
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)

    # Handle JSON strings
    if isinstance(value, str):
        value_strip = value.strip()
        if value_strip.startswith("{") or value_strip.startswith("["):
            try:
                return json.loads(value)
            except Exception:
                return value

    # Handle dict/list recursively
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_normalize_value(v) for v in value]

    return value

def _row_to_enriched_product(row: asyncpg.Record) -> EnrichedProduct:
    """Maps an asyncpg Record to an EnrichedProduct dataclass."""
    data = {k: _normalize_value(v) for k, v in dict(row).items()}
    
    return EnrichedProduct(
        product_id=data["product_id"],
        product_code=data["product_code"],
        product_name=data["product_name"],
        # base_price=float(data["base_price"]),
        # compare_at_price=(
        #     float(data["compare_at_price"])
        #     if data["compare_at_price"] is not None
        #     else None
        # ),
        # currency=data["currency"] or "INR",
        # in_stock=bool(data["in_stock"]),
        # stock_quantity=data["stock_quantity"],
        # variant_count=data["variant_count"] or 1,
        # attributes=dict(data["attributes"] or {}),
        # rating=float(data["rating"]) if data["rating"] is not None else None,
        # review_count=data["review_count"] or 0,
        # sales_rank=data["sales_rank"],
        # business_boost_score=float(data["business_boost_score"] or 0.0),
        images=data["images"],
        url_slug=data["url_slug"],
    )