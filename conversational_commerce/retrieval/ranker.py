# conversational_commerce/retrieval/ranker.py

from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.product import ProductCard
from schemas.query import ParsedQuery, SortOrder

from retrieval.hybrid_retrieval import HybridRetrievalResult, enrich_to_product_card
from retrieval.psql_client import EnrichedProduct

logger = get_logger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class RankingWeights:
    """
    Configurable weights for the business ranking formula.

    Formula:
        relevance_score = (
            semantic_weight     * semantic_score      +
            rating_weight       * normalised_rating   +
            popularity_weight   * normalised_popularity +
            boost_weight        * business_boost_score
        )

    Weights must sum to 1.0.
    Defaults are calibrated for Phase 1 discovery.
    Phase 3/4 will tune these per store/segment.
    """

    semantic_weight: float = 0.50      # Semantic relevance (Milvus score) — dominant
    rating_weight: float = 0.20        # Product rating signal
    popularity_weight: float = 0.20    # Sales rank signal
    boost_weight: float = 0.10         # Merchant-configured boost

    def __post_init__(self) -> None:
        total = (
            self.semantic_weight
            + self.rating_weight
            + self.popularity_weight
            + self.boost_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"RankingWeights must sum to 1.0, got {total:.4f}"
            )


# Default weights — used unless overridden by business settings
DEFAULT_WEIGHTS = RankingWeights()


class BusinessRanker:
    """
    Applies business ranking to hybrid retrieval candidates.

    Responsibility:
      - Compute a composite relevance_score for each candidate
      - Apply sort_order from ParsedQuery
      - Truncate to result_limit
      - Convert EnrichedProduct → ProductCard (the API type)
      - Populate key_attributes based on query context

    The ranker does NOT touch retrieval, session state, or LLM calls.
    It is a pure scoring and sorting function over the retrieval result.

    Stateless — one instance shared for the entire application lifecycle.
    """

    def __init__(self, weights: RankingWeights = DEFAULT_WEIGHTS) -> None:
        self._weights = weights

    def rank(
        self,
        retrieval_result: HybridRetrievalResult,
        parsed_query: ParsedQuery,
        exclude_product_ids: list[str] | None = None,
    ) -> list[ProductCard]:
        """
        Ranks enriched candidates and returns final ProductCard list.

        Args:
            retrieval_result:     Output from HybridRetriever.retrieve().
            parsed_query:         Query with sort_order and result_limit.
            exclude_product_ids:  Products already shown in this session.
                                  Excluded from results to avoid repetition.

        Returns:
            Ranked, deduplicated, truncated list of ProductCards.
        """
        products = retrieval_result.products
        semantic_scores = retrieval_result.semantic_scores
        exclude_set = set(exclude_product_ids or [])

        # logger.info(
        #     LogEvent.RANKER_START,
        #     "Business ranking started",
        #     candidate_count=len(products),
        #     sort_order=parsed_query.sort_order.value,
        #     result_limit=parsed_query.result_limit,
        #     exclusion_count=len(exclude_set),
        # )

        # # Step 1: Exclude already-shown products
        # products = [p for p in products if p.product_id not in exclude_set]

        # if not products:
        #     logger.warning(
        #         LogEvent.RANKER_END,
        #         "All candidates excluded — empty result after deduplication",
        #         excluded_count=len(exclude_set),
        #     )
        #     return []

        # # Step 2: Normalise signals across the candidate set
        # max_rating = max(
        #     (p.rating for p in products if p.rating is not None), default=5.0
        # )
        # min_sales_rank = min(
        #     (p.sales_rank for p in products if p.sales_rank is not None), default=1
        # )
        # max_sales_rank = max(
        #     (p.sales_rank for p in products if p.sales_rank is not None), default=1
        # )

        # # Step 3: Score each candidate
        # scored: list[tuple[float, EnrichedProduct]] = []

        # for product in products:
        #     score = self._compute_score(
        #         product=product,
        #         semantic_score=semantic_scores.get(product.product_id, 0.0),
        #         max_rating=max_rating,
        #         min_sales_rank=min_sales_rank,
        #         max_sales_rank=max_sales_rank,
        #     )
        #     scored.append((score, product))

        # # Step 4: Apply sort order
        # sorted_products = self._apply_sort(
        #     scored=scored,
        #     sort_order=parsed_query.sort_order,
        # )

        # # Step 5: Truncate to result_limit
        # sorted_products = sorted_products[: parsed_query.result_limit]

        # # Step 6: Convert to ProductCard with key_attributes
        # result: list[ProductCard] = []
        # for relevance_score, product in sorted_products:
        #     card = enrich_to_product_card(
        #         product=product,
        #         semantic_score=semantic_scores.get(product.product_id, 0.0),
        #         relevance_score=relevance_score,
        #     )
        #     # Populate key_attributes from query context
        #     card = self._attach_key_attributes(
        #         card=card,
        #         product=product,
        #         parsed_query=parsed_query,
        #     )
        #     result.append(card)

        # logger.info(
        #     LogEvent.RANKER_END,
        #     "Business ranking complete",
        #     final_count=len(result),
        #     top_score=result[0].relevance_score if result else 0.0,
        #     sort_order=parsed_query.sort_order.value,
        # )

        # return result
        products_sorted = sorted(
            products,
            key=lambda p: semantic_scores.get(p.product_id, 0.0),
            reverse=True
        )

        result = [
            enrich_to_product_card(
                product=p,
                semantic_score=semantic_scores.get(p.product_id, 0.0),
                relevance_score=semantic_scores.get(p.product_id, 0.0),
            )
            for p in products_sorted[: parsed_query.result_limit]
        ]

        return result

    def _compute_score(
        self,
        product: EnrichedProduct,
        semantic_score: float,
        max_rating: float,
        min_sales_rank: int,
        max_sales_rank: int,
    ) -> float:
        """
        Computes composite relevance score using weighted formula.
        All signals normalised to [0.0, 1.0] before weighting.
        """
        w = self._weights

        # Semantic score already in [0.0, 1.0] from Milvus cosine similarity
        s_semantic = max(0.0, min(1.0, semantic_score))

        # Rating: normalise to [0.0, 1.0]
        s_rating = 0.0
        if product.rating is not None and max_rating > 0:
            s_rating = product.rating / max_rating

        # Popularity: sales_rank is inverse (lower rank = more popular)
        # Normalise so rank=1 → score=1.0, rank=max → score=0.0
        s_popularity = 0.0
        if product.sales_rank is not None:
            rank_range = max_sales_rank - min_sales_rank
            if rank_range > 0:
                s_popularity = 1.0 - (
                    (product.sales_rank - min_sales_rank) / rank_range
                )
            else:
                s_popularity = 1.0  # All same rank

        # Business boost: already in [0.0, 1.0] from PSQL
        s_boost = max(0.0, min(1.0, product.business_boost_score))

        return (
            w.semantic_weight * s_semantic
            + w.rating_weight * s_rating
            + w.popularity_weight * s_popularity
            + w.boost_weight * s_boost
        )

    def _apply_sort(
        self,
        scored: list[tuple[float, EnrichedProduct]],
        sort_order: SortOrder,
    ) -> list[tuple[float, EnrichedProduct]]:
        """
        Sorts the scored candidate list by the requested sort order.
        RELEVANCE uses composite score. Others override with single signal.
        """
        if sort_order == SortOrder.RELEVANCE:
            return sorted(scored, key=lambda x: x[0], reverse=True)

        elif sort_order == SortOrder.PRICE_ASC:
            return sorted(scored, key=lambda x: x[1].base_price)

        elif sort_order == SortOrder.PRICE_DESC:
            return sorted(scored, key=lambda x: x[1].base_price, reverse=True)

        elif sort_order == SortOrder.POPULARITY:
            return sorted(
                scored,
                key=lambda x: x[1].sales_rank or 999_999,
            )

        elif sort_order == SortOrder.NEWEST:
            # Requires created_at in PSQL — fallback to relevance if not present
            return sorted(scored, key=lambda x: x[0], reverse=True)

        return sorted(scored, key=lambda x: x[0], reverse=True)

    def _attach_key_attributes(
        self,
        card: ProductCard,
        product: EnrichedProduct,
        parsed_query: ParsedQuery,
    ) -> ProductCard:
        """
        Selects and attaches the most query-relevant attributes to the card.

        Key attributes are shown on the product card in search results.
        We show attributes the user ASKED about, not all attributes.
        e.g. User asked about 'breathable shoes' → show 'material: mesh'
             Not: 'closure_type: lace-up, gender: unisex, season: all-season'
        """
        constraints = parsed_query.hard_constraints
        key_attrs: dict[str, str] = {}

        # Priority 1: attributes the user explicitly mentioned
        for field, attr_key in [
            (constraints.material, "material"),
            (constraints.color, "color"),
            (constraints.size, "size"),
            (constraints.gender, "gender"),
        ]:
            if field and attr_key in product.attributes:
                key_attrs[attr_key] = str(product.attributes[attr_key])

        # Priority 2: top 2 category-relevant attributes from the product
        category_defaults = _get_category_default_attributes(product.category)
        for attr_key in category_defaults:
            if len(key_attrs) >= 3:
                break
            if attr_key not in key_attrs and attr_key in product.attributes:
                key_attrs[attr_key] = str(product.attributes[attr_key])

        # Pydantic models are immutable by default — use model_copy
        return card.model_copy(update={"key_attributes": key_attrs})


def _get_category_default_attributes(category: str) -> list[str]:
    """
    Returns the most relevant attribute keys to surface for a given category.
    These are the attributes customers care about most when browsing that category.
    Extend this dict as you onboard new product categories.
    """
    defaults: dict[str, list[str]] = {
        "shoes": ["material", "sole_type", "closure_type", "gender"],
        "apparel": ["material", "fit_type", "gender", "sleeve_type"],
        "saree": ["fabric", "weave_type", "border_style", "occasion"],
        "handbags": ["material", "closure_type", "compartments", "style"],
        "electronics": ["brand", "warranty", "connectivity", "power"],
    }
    return defaults.get(category.lower(), ["material", "color", "brand"])