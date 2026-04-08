# conversational_commerce/tools/discovery_tools/search_products.py

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.product import ProductCard
from schemas.query import HardConstraints, ParsedQuery, SortOrder
from retrieval.hybrid_retrieval import HybridRetriever
from retrieval.ranker import BusinessRanker

logger = get_logger(__name__)
settings = get_settings()

# Module-level singletons — initialised once, reused across all requests
_hybrid_retriever = HybridRetriever()
_business_ranker = BusinessRanker()
_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai.api_key)
    return _openai_client


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class SearchProductsInput(BaseModel):
    """
    Input schema for the search_products tool.

    The Discovery Agent's LLM produces these values via structured output.
    Every field maps directly to a component of the retrieval pipeline.
    The LLM description on each field IS the prompt — write it carefully.
    """

    semantic_query: str = Field(
        description=(
            "The enriched natural language search query for semantic matching. "
            "Expand abbreviations, add category context, and include key attributes "
            "the user mentioned. "
            "Example: 'lightweight non-marking indoor badminton sports shoes'"
        )
    )
    category: str | None = Field(
        default=None,
        description=(
            "Product category if clearly stated or strongly implied by the user. "
            "Use lowercase. Examples: 'shoes', 'apparel', 'saree', 'handbags'. "
            "Do NOT infer if uncertain."
        ),
    )
    sub_category: str | None = Field(
        default=None,
        description=(
            "Sub-category within the main category. "
            "Example: 'sports' within 'shoes', 'ethnic' within 'apparel'. "
            "Only populate if explicitly mentioned."
        ),
    )
    brand: str | None = Field(
        default=None,
        description=(
            "Brand name if user specified one. "
            "Example: 'Nike', 'Adidas'. Do NOT infer brand from style."
        ),
    )
    min_price: float | None = Field(
        default=None,
        description="Minimum price in store currency. Only set if user stated a lower bound.",
    )
    max_price: float | None = Field(
        default=None,
        description="Maximum price in store currency. Only set if user stated an upper bound.",
    )
    gender: str | None = Field(
        default=None,
        description=(
            "Gender filter. One of: 'men', 'women', 'unisex', 'boys', 'girls', 'kids'. "
            "Only set if explicitly mentioned."
        ),
    )
    min_rating: float | None = Field(
        default=None,
        description="Minimum product rating (0.0–5.0) if user stated a quality threshold.",
    )
    sort_order: str = Field(
        default="relevance",
        description=(
            "How to sort results. One of: "
            "'relevance' (default), 'price_asc', 'price_desc', 'popularity', 'newest'. "
            "Only change from default if user explicitly requests a sort order."
        ),
    )
    result_limit: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of products to return. Default 10. Increase only if user asks for 'more'.",
    )
    entity_id: str | None = Field(
        default=None,
        description="Entity ID from session context. Always pass this if available.",
    )
    business_unit_id: str | None = Field(
        default=None,
        description="Business unit ID from session context. Always pass this if available.",
    )

    exclude_product_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Product IDs already shown to the user in this session. "
            "These will be excluded from results to avoid repetition. "
            "Always pass this from session context."
        ),
    )
    inference_notes: str = Field(
        default="",
        description=(
            "Your reasoning for the filters you chose. "
            "Example: 'User said badminton → inferred sports sub_category. "
            "No price stated → left price null.' "
            "This is for audit logging only — not shown to the user."
        ),
    )


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

async def _search_products_impl(
    semantic_query: str,
    category: str | None = None,
    sub_category: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    gender: str | None = None,
    min_rating: float | None = None,
    sort_order: str = "relevance",
    result_limit: int = 10,
    entity_id: str | None = None,
    business_unit_id: str | None = None,
    exclude_product_ids: list[str] | None = None,
    inference_notes: str = "",
    dynamic_filters: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Core implementation of the search_products tool.

    Pipeline:
      1. Build ParsedQuery from tool inputs
      2. Encode semantic_query → embedding vector
      3. HybridRetriever → enriched candidates with relaxation
      4. BusinessRanker → scored, sorted, deduplicated ProductCards
      5. Return structured result dict

    Returns a dict (not a Pydantic model) because LangGraph
    serialises tool results to JSON for the agent message history.
    """
    from schemas.query import PriceRange

    # ── Step 1: Build ParsedQuery ─────────────────────────────────────────
    price_range = None
    if min_price is not None or max_price is not None:
        price_range = PriceRange(min_price=min_price, max_price=max_price)

    try:
        sort = SortOrder(sort_order)
    except ValueError:
        logger.warning(
            LogEvent.AGENT_TOOL_CALL,
            "Invalid sort_order received — defaulting to RELEVANCE",
            received=sort_order,
        )
        sort = SortOrder.RELEVANCE

    constraints = HardConstraints(
        price_range=price_range,
        category=category,
        sub_category=sub_category,
        brand=brand,
        gender=gender,
        min_rating=min_rating,
        entity_id=entity_id,
        business_unit_id=business_unit_id,
        in_stock_only=True,         # Always — never show unavailable products
        dynamic_filters=dynamic_filters or {}
    )

    parsed_query = ParsedQuery(
        semantic_query=semantic_query,
        hard_constraints=constraints,
        sort_order=sort,
        result_limit=result_limit,
        inference_notes=inference_notes,
        original_query=semantic_query,
    )

    logger.info(
        LogEvent.AGENT_TOOL_CALL,
        "search_products called",
        semantic_query=semantic_query[:100],
        category=category,
        brand=brand,
        price_range={"min": min_price, "max": max_price},
        sort_order=sort_order,
        result_limit=result_limit,
        inference_notes=inference_notes[:200],
    )

    # ── Step 2: Encode semantic query → embedding vector ─────────────────
    async with logger.timed(
        LogEvent.RETRIEVAL_MILVUS_START,
        "query_embedding_encoding",
        model=settings.embedding.model_name,
    ):
        embedding_response = await _get_openai_client().embeddings.create(
            model=settings.embedding.model_name,
            input=semantic_query,
            dimensions=settings.embedding.dimensions,
        )
    query_vector = embedding_response.data[0].embedding

    # ── Step 3: Hybrid retrieval ──────────────────────────────────────────
    retrieval_result = await _hybrid_retriever.retrieve(
        parsed_query=parsed_query,
        query_vector=query_vector,
        entity_id=entity_id,
        business_unit_id=business_unit_id,
    )

    # ── Step 4: Business ranking ──────────────────────────────────────────
    ranked_cards = _business_ranker.rank(
        retrieval_result=retrieval_result,
        parsed_query=parsed_query,
        exclude_product_ids=exclude_product_ids or [],
    )

    # ── Step 5: Build result dict ─────────────────────────────────────────
    result = {
        "success": True,
        "products": [_card_to_dict(card) for card in ranked_cards],
        "total_found": len(ranked_cards),
        "retrieval_metadata": {
            "milvus_candidates": retrieval_result.total_milvus_candidates,
            "after_filtering": retrieval_result.total_after_filtering,
            "relaxation_rounds": retrieval_result.relaxation_rounds_applied,
            "relaxation_applied": retrieval_result.relaxation_rounds_applied > 0,
            "retrieval_latency_ms": round(retrieval_result.retrieval_latency_ms, 2),
        },
    }

    logger.info(
        LogEvent.AGENT_TOOL_RESULT,
        "search_products complete",
        total_found=result["total_found"],
        milvus_candidates=retrieval_result.total_milvus_candidates,
        after_filtering=retrieval_result.total_after_filtering,
        relaxation_rounds=retrieval_result.relaxation_rounds_applied,
    )

    return result


def _card_to_dict(card: ProductCard) -> dict[str, Any]:
    """
    Converts ProductCard → dict for LangGraph message serialisation.

    Only includes fields the LLM synthesizer needs to compose a response.
    Full ProductCard fields are preserved — synthesizer picks what's relevant.
    """
    result: dict[str, Any] = {
        # "product_id": card.product_id,
        # "name": card.name,
        # "brand": card.brand,
        # "category": card.category,
        # "sub_category": card.sub_category,
        # "base_price": card.base_price,
        # "currency": card.currency,
        # "in_stock": card.in_stock,
        # "variant_count": card.variant_count,
        # "short_description": card.short_description,
        # "key_attributes": card.key_attributes,
        # "rating": card.rating,
        # "review_count": card.review_count,
        # "relevance_score": card.relevance_score,

        "product_id": card.product_id,
        "product_code": card.product_code,
        "product_name": card.product_name,
        "url_slug": card.url_slug,
        "images": card.images,
        "relevance_score": card.relevance_score,
    }

    # # Discount info — synthesizer uses this for sales messaging
    # if card.is_on_sale:
    #     result["compare_at_price"] = card.compare_at_price
    #     result["discount_percent"] = card.discount_percent
    #     result["is_on_sale"] = True
    # else:
    #     result["is_on_sale"] = False

    # # Image URL — for response rendering
    # if card.primary_image:
    #     result["primary_image_url"] = card.primary_image.url

    return result


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def create_search_products_tool() -> StructuredTool:
    """
    Creates the search_products StructuredTool.
    Called by bootstrap_tools() at startup — not at request time.
    """
    from tools.registry import make_instrumented_tool

    return make_instrumented_tool(
        func=_search_products_impl,
        name="search_products",
        description=(
            "Search for products in the Honebi catalog using natural language. "
            "Use this tool when the user is looking for products, asking for recommendations, "
            "or wants to browse a category. "
            "Always pass entity_id and business_unit_id and exclude_product_ids from the session context. "
            "The tool handles semantic search, filtering, and ranking automatically. "
            "Returns a ranked list of products with prices, images, and key attributes."
        ),
        args_schema=SearchProductsInput,
    )