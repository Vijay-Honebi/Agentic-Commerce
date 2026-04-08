# conversational_commerce/tools/discovery_tools/get_product_details.py

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.product import ProductDetail, ProductImage, ProductVariant
from retrieval.psql_client import EnrichedProduct, fetch_product_by_id

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class GetProductDetailsInput(BaseModel):
    """
    Input schema for the get_product_details tool.

    The agent calls this when:
      - User asks 'tell me more about [product]'
      - User refers to 'the first one', 'that blue shoe', etc.
        (Orchestrator resolves the reference → product_id before calling)
      - User is about to add to cart and needs variant information (Phase 2)
    """

    product_id: str = Field(
        description=(
            "The exact product_id from a previous search_products result. "
            "Never guess or construct a product_id — only use IDs from search results."
        )
    )
    entity_id: str | None = Field(
        default=None,
        description="Entity ID from session context. Pass this for access control.",
    )
    business_unit_id: str | None = Field(
        default=None,
        description="Business Unit ID from session context. Pass this for access control.",
    )
    requesting_agent: str = Field(
        default="discovery_agent",
        description=(
            "Name of the agent requesting details. "
            "Used for access control logging. "
            "Example: 'discovery_agent', 'cart_agent'."
        ),
    )


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

async def _get_product_details_impl(
    product_id: str,
    entity_id: str | None = None,
    business_unit_id: str | None = None,
    requesting_agent: str = "discovery_agent",
) -> dict[str, Any]:
    """
    Fetches full product details for a single product.

    Pipeline:
      1. Fetch EnrichedProduct from PSQL (no constraint filtering)
      2. Fetch all variants for this product
      3. Fetch full attribute set
      4. Convert → ProductDetail → dict

    Returns a structured dict with everything needed for:
      - Rendering a full product detail page
      - Phase 2: selecting a variant_id for add-to-cart
      - Phase 3: identifying cross-sell and upsell candidates

    Returns error dict (never raises) if product not found.
    """
    logger.info(
        LogEvent.AGENT_TOOL_CALL,
        "get_product_details called",
        product_id=product_id,
        entity_id=entity_id,
        business_unit_id=business_unit_id,
        requesting_agent=requesting_agent,
    )

    # ── Step 1: Fetch base product ────────────────────────────────────────
    enriched = await fetch_product_by_id(product_id)

    if enriched is None:
        logger.warning(
            LogEvent.AGENT_TOOL_RESULT,
            "Product not found",
            product_id=product_id,
        )
        return {
            "success": False,
            "error_code": "PRODUCT_NOT_FOUND",
            "error_detail": f"No product found with ID '{product_id}'.",
            "product_id": product_id,
        }

    # ── Step 2: Entity-level access control ───────────────────────────────
    # Ensure the product belongs to the requesting entity and business unit.
    if entity_id and enriched.entity_id != entity_id:
        logger.warning(
            LogEvent.GUARDRAIL_VIOLATION,
            "Cross-entity product access attempt blocked",
            requested_product_id=product_id,
            product_entity_id=enriched.entity_id,
            requesting_entity_id=entity_id,
            requesting_agent=requesting_agent,
        )
        return {
            "success": False,
            "error_code": "PRODUCT_NOT_FOUND",  # Intentionally vague — don't leak entity data
            "error_detail": "Product not available in this entity.",
            "product_id": product_id,
        }
    
    if business_unit_id and enriched.business_unit_id != business_unit_id:
        logger.warning(
            LogEvent.GUARDRAIL_VIOLATION,
            "Cross-business-unit product access attempt blocked",
            requested_product_id=product_id,
            product_business_unit_id=enriched.business_unit_id,
            requesting_business_unit_id=business_unit_id,
            requesting_agent=requesting_agent,
        )
        return {
            "success": False,
            "error_code": "PRODUCT_NOT_FOUND",  # Intentionally vague — don't leak business unit data
            "error_detail": "Product not available in this business unit.",
            "product_id": product_id,
        }

    # ── Step 3: Fetch full variants ───────────────────────────────────────
    variants = await _fetch_product_variants(product_id)

    # ── Step 4: Fetch full images ─────────────────────────────────────────
    images = await _fetch_product_images(product_id)

    # ── Step 5: Build ProductDetail ───────────────────────────────────────
    detail = _build_product_detail(
        enriched=enriched,
        variants=variants,
        images=images,
    )

    result = _detail_to_dict(detail)

    logger.info(
        LogEvent.AGENT_TOOL_RESULT,
        "get_product_details complete",
        product_id=product_id,
        variant_count=len(variants),
        image_count=len(images),
    )

    return result


async def _fetch_product_variants(product_id: str) -> list[ProductVariant]:
    """
    Fetches all variants for a product from PSQL.
    Returns empty list if no variants table or no variants found.

    Phase 2 note: variant_id from these results feeds directly into
    the add_to_cart tool. The schema is intentionally Phase 2 ready.
    """
    from retrieval.psql_client import _get_pool

    try:
        async with _get_pool().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    variant_id,
                    sku,
                    attributes,
                    price,
                    compare_at_price,
                    in_stock,
                    stock_quantity
                FROM product_variants
                WHERE product_id = $1
                ORDER BY price ASC
                """,
                product_id,
            )

        return [
            ProductVariant(
                variant_id=str(row["variant_id"]),
                sku=row["sku"] or "",
                attributes=dict(row["attributes"] or {}),
                price=float(row["price"]),
                compare_at_price=(
                    float(row["compare_at_price"])
                    if row["compare_at_price"] is not None
                    else None
                ),
                in_stock=bool(row["in_stock"]),
                stock_quantity=row["stock_quantity"],
            )
            for row in rows
        ]

    except Exception as e:
        logger.warning(
            LogEvent.RETRIEVAL_PSQL_END,
            "Failed to fetch product variants — returning empty list",
            product_id=product_id,
            error=str(e),
        )
        return []


async def _fetch_product_images(product_id: str) -> list[ProductImage]:
    """
    Fetches all images for a product from PSQL.
    Ordered: primary image first, then by display_order.
    """
    from retrieval.psql_client import _get_pool

    try:
        async with _get_pool().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT url, alt_text, is_primary
                FROM product_images
                WHERE product_id = $1
                ORDER BY is_primary DESC, display_order ASC
                """,
                product_id,
            )

        return [
            ProductImage(
                url=row["url"],
                alt_text=row["alt_text"] or "",
                is_primary=bool(row["is_primary"]),
            )
            for row in rows
        ]

    except Exception as e:
        logger.warning(
            LogEvent.RETRIEVAL_PSQL_END,
            "Failed to fetch product images — returning empty list",
            product_id=product_id,
            error=str(e),
        )
        return []


def _build_product_detail(
    enriched: EnrichedProduct,
    variants: list[ProductVariant],
    images: list[ProductImage],
) -> ProductDetail:
    """Assembles ProductDetail from its constituent parts."""
    return ProductDetail(
        product_id=enriched.product_id,
        entity_id=enriched.entity_id,
        business_unit_id=enriched.business_unit_id,
        name=enriched.name,
        brand=enriched.brand,
        category=enriched.category,
        sub_category=enriched.sub_category,
        family=enriched.family,
        description=enriched.short_description,
        short_description=enriched.short_description,
        images=images,
        currency=enriched.currency,
        variants=variants,
        attributes=enriched.attributes,
        specifications={},              # Extended in Phase 1 as catalog matures
        rating=enriched.rating,
        review_count=enriched.review_count,
        is_active=True,
        related_product_ids=[],        # Phase 3: populated from recommendation engine
        frequently_bought_with=[],     # Phase 3: populated from co-purchase model
    )


def _detail_to_dict(detail: ProductDetail) -> dict[str, Any]:
    """
    Converts ProductDetail → dict for LangGraph serialisation.

    Includes full variant information so the synthesizer can communicate
    available options to the user, and Phase 2 can immediately act on variant_id.
    """
    return {
        "success": True,
        "product_id": detail.product_id,
        "entity_id": detail.entity_id,
        "business_unit_id": detail.business_unit_id,
        "name": detail.name,
        "brand": detail.brand,
        "category": detail.category,
        "sub_category": detail.sub_category,
        "family": detail.family,
        "description": detail.description,
        "short_description": detail.short_description,
        "currency": detail.currency,
        "rating": detail.rating,
        "review_count": detail.review_count,
        "attributes": detail.attributes,
        "specifications": detail.specifications,
        "images": [
            {
                "url": img.url,
                "alt_text": img.alt_text,
                "is_primary": img.is_primary,
            }
            for img in detail.images
        ],
        "variants": [
            {
                "variant_id": v.variant_id,    # Phase 2: this feeds add_to_cart
                "sku": v.sku,
                "attributes": v.attributes,    # e.g. {"color": "red", "size": "42"}
                "price": v.price,
                "compare_at_price": v.compare_at_price,
                "in_stock": v.in_stock,
                "stock_quantity": v.stock_quantity,
            }
            for v in detail.variants
        ],
        "available_sizes": sorted({
            v.attributes.get("size", "")
            for v in detail.variants
            if v.in_stock and v.attributes.get("size")
        }),
        "available_colors": sorted({
            v.attributes.get("color", "")
            for v in detail.variants
            if v.in_stock and v.attributes.get("color")
        }),
        # Phase 3 hooks — empty now, populated when recommendation engine is live
        "related_product_ids": detail.related_product_ids,
        "frequently_bought_with": detail.frequently_bought_with,
    }


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def create_get_product_details_tool() -> StructuredTool:
    """
    Creates the get_product_details StructuredTool.
    Called by bootstrap_tools() at startup.
    """
    from tools.registry import make_instrumented_tool

    return make_instrumented_tool(
        func=_get_product_details_impl,
        name="get_product_details",
        description=(
            "Retrieve complete details for a specific product. "
            "Use this when the user asks for more information about a product "
            "they have already seen in search results, or when they refer to "
            "'the first one', 'that shoe', 'the red one', etc. "
            "Also use this before Phase 2 add-to-cart to get available variant_ids. "
            "Requires a product_id from a previous search_products result — "
            "never guess or construct a product_id."
        ),
        args_schema=GetProductDetailsInput,
    )