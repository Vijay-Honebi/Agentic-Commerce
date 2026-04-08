# conversational_commerce/schemas/query.py

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class PriceRange(BaseModel):
    """Closed interval [min_price, max_price] in the store's base currency."""

    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> PriceRange:
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError(
                f"min_price ({self.min_price}) cannot exceed "
                f"max_price ({self.max_price})"
            )
        return self


class SortOrder(str, Enum):
    """Controls Business Ranker's final sort after retrieval + scoring."""

    RELEVANCE = "relevance"       # Default — pure semantic + business score
    PRICE_ASC = "price_asc"       # Cheapest first
    PRICE_DESC = "price_desc"     # Most expensive first
    POPULARITY = "popularity"     # By sales rank
    NEWEST = "newest"             # By catalog creation date


class HardConstraints(BaseModel):
    """
    Filters extracted from the user query that MUST be satisfied.
    These are passed as SQL WHERE clauses — not semantic hints.

    "Hard" means: a product violating any of these is excluded from results,
    even if it is semantically very similar to the query.

    The relaxation engine may loosen these (by step_percent per round, max 2 rounds)
    only if candidate count falls below min_candidates_threshold.
    Every relaxation event is logged with full audit trail.

    Design note:
        All fields are Optional. LLM only populates fields it finds in the query.
        Missing fields = no filter on that dimension.
        This prevents the LLM from hallucinating constraints that weren't stated.
    """

    # Price
    price_range: PriceRange | None = Field(
        default=None,
        description="User-stated price range. e.g. 'under 500' → max_price=500",
    )

    # Product identity
    category: str | None = Field(
        default=None,
        description="Top-level product category. e.g. 'shoes', 'apparel'",
    )
    sub_category: str | None = Field(
        default=None,
        description="Sub-category within category. e.g. 'sports', 'casual'",
    )
    brand: str | None = Field(
        default=None,
        description="Specific brand name if stated. e.g. 'Nike', 'Adidas'",
    )

    # Product attributes (flexible — accommodates SMB + Enterprise catalog models)
    gender: str | None = Field(
        default=None,
        description="Gender target. e.g. 'men', 'women', 'unisex'",
    )
    size: str | None = Field(
        default=None,
        description="Size specification. e.g. 'XL', '42', 'free size'",
    )
    color: str | None = Field(
        default=None,
        description="Color filter. e.g. 'black', 'red'",
    )
    material: str | None = Field(
        default=None,
        description="Material specification. e.g. 'cotton', 'leather'",
    )

    # Availability & ratings
    in_stock_only: bool = Field(
        default=True,
        description="Exclude out-of-stock products. Default True — "
                    "never show unavailable products unless user asks.",
    )
    min_rating: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Minimum product rating filter.",
    )

    # Store / business unit scope

    business_unit_id: str | None = Field(
        default=None,
        description="Business unit scope for this query.",
    )
    entity_id: str | None = Field(
        default=None,
        description=(
            "Entity scope within the business unit. "
            "Narrows catalog to this specific store/branch/channel."
        ),
    )

    # Flexible catch-all for catalog-specific attributes not in the fixed schema
    # e.g. {"sole_type": "rubber", "closure_type": "lace-up"}
    dynamic_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional product attributes that don't fit fixed fields. "
                    "Passed as dynamic PSQL filters.",
    )

    @field_validator("gender", mode="before")
    @classmethod
    def normalise_gender(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalised = v.strip().lower()
        allowed = {"men", "women", "unisex", "boys", "girls", "kids"}
        if normalised not in allowed:
            # Don't raise — just drop invalid gender values silently.
            # LLM can hallucinate unusual gender strings; we prefer no filter
            # over a filter that blocks all results.
            return None
        return normalised


class RelaxationRecord(BaseModel):
    """
    Immutable audit record for a single relaxation round.
    Stored in session state and emitted to observability layer.
    """

    round_number: int
    original_value: Any
    relaxed_value: Any
    field_relaxed: str
    candidate_count_before: int
    candidate_count_after: int
    relaxation_percent: float


class ParsedQuery(BaseModel):
    """
    Structured output of the LLM query parser node.

    The LLM parser receives the raw user message and produces this object.
    It is the single point where natural language becomes structured data.

    Two outputs serve different retrieval systems:
      semantic_query   → fed to the embedding model → Milvus ANN search
      hard_constraints → compiled to SQL WHERE clause → PSQL filter

    Invariants the LLM parser must respect (enforced by guardrails):
      1. semantic_query must not be empty
      2. hard_constraints must not contain fields not mentioned in user message
      3. inferred attributes must have inference_notes explaining the reasoning
    """

    # ── Semantic retrieval input ──────────────────────────────────────────
    semantic_query: str = Field(
        description="Cleaned, enriched query text for embedding. "
                    "Should expand abbreviations and add relevant context. "
                    "e.g. 'lightweight badminton shoes indoor non-marking sole'",
        min_length=1,
    )

    # ── Structured retrieval input ────────────────────────────────────────
    hard_constraints: HardConstraints = Field(
        default_factory=HardConstraints,
        description="SQL-compilable filters. Strictly from user's stated requirements.",
    )

    # ── Result control ────────────────────────────────────────────────────
    sort_order: SortOrder = Field(
        default=SortOrder.RELEVANCE,
        description="Requested sort order. Default RELEVANCE unless user specifies.",
    )
    result_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return to user. "
                    "Orchestrator may override for UI context.",
    )

    # ── LLM reasoning transparency ────────────────────────────────────────
    # These fields are for observability and debugging — not shown to the user.
    # They allow engineers to audit LLM parsing decisions without replaying queries.
    inference_notes: str = Field(
        default="",
        description="LLM's explanation of parsing decisions. "
                    "e.g. 'Inferred sports category from badminton context. "
                    "No price stated — left price_range null.'",
    )
    original_query: str = Field(
        default="",
        description="Raw user message before any processing. "
                    "Preserved for audit trail.",
    )
    relaxation_history: list[RelaxationRecord] = Field(
        default_factory=list,
        description="Populated by relaxation_engine.py if triggered. "
                    "Records every constraint relaxation for audit.",
    )

    @field_validator("semantic_query", mode="before")
    @classmethod
    def clean_semantic_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("semantic_query cannot be empty or whitespace")
        return cleaned