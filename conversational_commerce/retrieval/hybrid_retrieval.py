# conversational_commerce/retrieval/hybrid_retriever.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.query import HardConstraints, ParsedQuery, RelaxationRecord
from schemas.product import ProductCard, ProductImage

from retrieval.milvus_client import (
    MilvusFilterBuilder,
    MilvusSearchResult,
    vector_search,
)
from retrieval.psql_client import EnrichedProduct, enrich_candidates
from retrieval.relaxation_engine import RelaxationEngine

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Retrieval result bundle
# ---------------------------------------------------------------------------

@dataclass
class HybridRetrievalResult:
    """
    Complete output from hybrid retrieval.
    Passed to ranker.py and then to the discovery tool.
    """

    products: list[EnrichedProduct]         # Enriched, filtered candidates
    semantic_scores: dict[str, float]       # product_id → Milvus cosine score
    relaxation_records: list[RelaxationRecord]
    relaxation_rounds_applied: int
    total_milvus_candidates: int            # Before PSQL filtering
    total_after_filtering: int              # After PSQL filtering
    retrieval_latency_ms: float


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Orchestrates the full retrieval pipeline:

      1. Build Milvus pre-filter (high-selectivity constraints only)
      2. ANN vector search → candidate_ids + semantic scores
      3. PSQL enrichment + hard constraint filtering → enriched products
      4. Check candidate count against threshold
      5. If below threshold → trigger RelaxationEngine → retry (max 2 rounds)
      6. Return HybridRetrievalResult to ranker

    This class owns the relaxation loop.
    It never touches ranking — that's ranker.py's responsibility.
    It never touches session state — that's the Orchestrator's responsibility.

    One instance per application (stateless — safe to share across requests).
    """

    def __init__(self) -> None:
        self._relaxation_engine = RelaxationEngine()
        self._cfg = settings.milvus

    async def retrieve(
        self,
        parsed_query: ParsedQuery,
        query_vector: list[float],
        entity_id: str | None = None,
        business_unit_id: str | None = None,
    ) -> HybridRetrievalResult:
        """
        Executes hybrid retrieval with adaptive relaxation.

        Args:
            parsed_query:  Structured query from LLM parser node.
            query_vector:  Encoded embedding of semantic_query.
            entity_id:     Entity scope from request context.
                           Overrides constraints.entity_id if both present.
            business_unit_id: Business unit scope from request context.
                               Overrides constraints.business_unit_id if both present.

        Returns:
            HybridRetrievalResult with enriched products ready for ranking.
        """
        start_time = time.perf_counter()

        # Working copy of constraints — relaxation mutates this across rounds
        active_constraints = parsed_query.hard_constraints.model_copy(deep=True)
        relaxation_records: list[RelaxationRecord] = []
        relaxation_rounds = 0
        total_milvus_candidates = 0

        logger.info(
            LogEvent.RETRIEVAL_MILVUS_START,
            "Hybrid retrieval started",
            semantic_query=parsed_query.semantic_query[:100],
            entity_id=entity_id,
            business_unit_id=business_unit_id,
            constraints_summary=_summarise_constraints(active_constraints),
            dynamic_filters_count=active_constraints.dynamic_filters,
        )

        # ── Relaxation loop ───────────────────────────────────────────────
        # Round 0 = initial attempt (no relaxation)
        # Round 1 = price widening
        # Round 2 = attribute dropping
        # Max rounds enforced by RelaxationEngine

        milvus_results: list[MilvusSearchResult] = []
        enriched: list[EnrichedProduct] = []

        for attempt in range(settings.relaxation.max_rounds + 1):
            # Step 1: Build Milvus pre-filter
            milvus_filter = self._build_milvus_filter(
                constraints=active_constraints,
                business_unit_id=business_unit_id,
            )

            # Step 2: ANN vector search
            milvus_results = await vector_search(
                query_vector=query_vector,
                top_k=self._cfg.top_k,
                filter_expr=milvus_filter,
            )

            if attempt == 0:
                total_milvus_candidates = len(milvus_results)

            # Step 3: PSQL enrichment + hard constraint filtering
            candidate_ids = [r.product_id for r in milvus_results]
            enriched = await enrich_candidates(
                product_ids=candidate_ids,
                constraints=active_constraints,
                entity_id=entity_id,
                business_unit_id=business_unit_id,
            )

            logger.info(
                LogEvent.RETRIEVAL_HYBRID_END,
                "Retrieval attempt complete",
                attempt=attempt,
                milvus_hits=len(milvus_results),
                enriched_count=len(enriched),
                threshold=self._cfg.min_candidates_threshold,
            )

            # Step 4: Check threshold
            if not self._relaxation_engine.should_relax(len(enriched)):
                # Threshold met — no relaxation needed
                break

            # Step 5: Attempt relaxation
            next_round = attempt + 1
            relaxation_result = self._relaxation_engine.relax(
                parsed_query=parsed_query,
                candidate_count_before=len(enriched),
                round_number=next_round,
            )

            if relaxation_result is None:
                # No more relaxation possible — return best-effort
                logger.warning(
                    LogEvent.RETRIEVAL_RELAXATION_TRIGGERED,
                    "Relaxation exhausted — returning best-effort results",
                    final_candidate_count=len(enriched),
                )
                break

            new_constraints, new_records = relaxation_result
            active_constraints = new_constraints
            relaxation_rounds += 1

            # Fill in candidate_count_after on records from this round
            for record in new_records:
                object.__setattr__(record, "candidate_count_after", len(enriched))
            relaxation_records.extend(new_records)

        # ── Build semantic scores map ─────────────────────────────────────
        # Preserve Milvus scores for ranking — keyed by product_id
        semantic_scores = {
            r.product_id: r.score
            for r in milvus_results   # type: ignore[possibly-undefined]
        }

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            LogEvent.RETRIEVAL_HYBRID_END,
            "Hybrid retrieval complete",
            total_milvus_candidates=total_milvus_candidates,
            final_candidate_count=len(enriched),
            relaxation_rounds=relaxation_rounds,
            latency_ms=round(latency_ms, 2),
        )

        return HybridRetrievalResult(
            products=enriched,          # type: ignore[possibly-undefined]
            semantic_scores=semantic_scores,
            relaxation_records=relaxation_records,
            relaxation_rounds_applied=relaxation_rounds,
            total_milvus_candidates=total_milvus_candidates,
            total_after_filtering=len(enriched),  # type: ignore[possibly-undefined]
            retrieval_latency_ms=latency_ms,
        )

    def _build_milvus_filter(
        self,
        constraints: HardConstraints,
        business_unit_id: str | None,
    ) -> str | None:
        """
        Builds Milvus pre-filter expression from high-selectivity constraints.
        Low-selectivity constraints (price, color, material) go to PSQL only.
        """
        builder = MilvusFilterBuilder()

        # effective_entity_id = entity_id or constraints.entity_id
        # if effective_entity_id:
        #     builder.entity(effective_entity_id)

        # effective_business_unit_id = business_unit_id or constraints.business_unit_id
        # if effective_business_unit_id:
        #     builder.business_unit(effective_business_unit_id)

        # if constraints.category:
        #     builder.category(constraints.category)

        return builder.build()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise_constraints(constraints: HardConstraints) -> dict[str, Any]:
    """Compact constraint summary for logging — avoids dumping full object."""
    return {
        "price_min": constraints.price_range.min_price if constraints.price_range else None,
        "price_max": constraints.price_range.max_price if constraints.price_range else None,
        "category": constraints.category,
        "brand": constraints.brand,
        "gender": constraints.gender,
        "in_stock_only": constraints.in_stock_only,
        "dynamic_filters": list(constraints.dynamic_filters.keys()),
    }


def enrich_to_product_card(
    product: EnrichedProduct,
    semantic_score: float = 0.0,
    relevance_score: float = 0.0,
) -> ProductCard:
    """
    Converts EnrichedProduct (internal retrieval type) → ProductCard (API contract).
    Called by ranker.py after scoring — this is the final type conversion.
    """
    # from schemas.product import ProductImage

    # primary_image = None
    # if product.primary_image_url:
    #     primary_image = ProductImage(
    #         url=product.primary_image_url,
    #         alt_text=product.primary_image_alt or "",
    #         is_primary=True,
    #     )

    # return ProductCard(
    #     product_id=product.product_id,
    #     entity_id=product.entity_id,
    #     business_unit_id=product.business_unit_id,
    #     name=product.name,
    #     brand=product.brand,
    #     category=product.category,
    #     sub_category=product.sub_category,
    #     base_price=product.base_price,
    #     compare_at_price=product.compare_at_price,
    #     currency=product.currency,
    #     primary_image=primary_image,
    #     short_description=product.short_description,
    #     key_attributes={},              # Populated by ranker based on query context
    #     in_stock=product.in_stock,
    #     variant_count=product.variant_count,
    #     relevance_score=round(relevance_score, 4),
    #     semantic_score=round(semantic_score, 4),
    #     rating=product.rating,
    #     review_count=product.review_count,
    # )

    return ProductCard(
        product_id=product.product_id,
        product_code=product.product_code,
        product_name=product.product_name,
        url_slug=product.url_slug,
        images=product.images,
        relevance_score=round(relevance_score, 4),
    )