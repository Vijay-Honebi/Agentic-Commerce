# conversational_commerce/schemas/product.py

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class ProductImage(BaseModel):
    """A single product image with its role in the catalog."""

    url: str = Field(description="Publicly accessible image URL.")
    alt_text: str = Field(default="", description="Accessibility alt text.")
    is_primary: bool = Field(
        default=False,
        description="True for the hero image shown in search results.",
    )


class ProductVariant(BaseModel):
    """
    A single purchasable variant of a product.
    e.g. 'Red / Size 42' or 'Blue / XL'

    Discovery returns variants so the Orchestrator knows what's actually
    available to add to cart in Phase 2 — no second lookup needed.
    """

    variant_id: str
    sku: str
    attributes: dict[str, str] = Field(
        description="Variant-defining attributes. e.g. {'color': 'red', 'size': '42'}",
    )
    price: float
    compare_at_price: float | None = Field(
        default=None,
        description="Original price before discount. None if no active discount.",
    )
    in_stock: bool
    stock_quantity: int | None = Field(
        default=None,
        description="Exact stock count if available. None if only in/out tracked.",
    )


# class ProductCard(BaseModel):
#     """
#     Minimal product representation for search result lists.

#     Designed for Discovery Agent's search results — contains exactly
#     what the UI needs to render a product card without further API calls.
#     The Orchestrator synthesizes these into the user-facing response.

#     Phase 2 contract: variant_id from ProductCard → add_to_cart tool input.
#     This means discovery results are directly actionable in Phase 2 without
#     any additional product lookup.
#     """

#     product_id: str = Field(description="Canonical product identifier.")
#     entity_id: str = Field(description="Honebi entity this product belongs to.")
#     business_unit_id: str = Field(description="Honebi business unit this product belongs to.")
#     name: str
#     brand: str | None = None
#     category: str
#     sub_category: str | None = None

#     # Pricing — reflects the best available variant price
#     base_price: float = Field(description="Lowest available variant price.")
#     compare_at_price: float | None = Field(
#         default=None,
#         description="Original price for discount display. None if no active discount.",
#     )
#     currency: str = Field(default="INR")

#     # Discovery metadata
#     primary_image: ProductImage | None = None
#     short_description: str = Field(
#         default="",
#         description="1-2 sentence product summary for result display.",
#     )
#     key_attributes: dict[str, str] = Field(
#         default_factory=dict,
#         description="Most relevant attributes for this query. "
#                     "e.g. {'material': 'mesh', 'sole': 'non-marking'}. "
#                     "Populated by ranker based on query context.",
#     )

#     # Availability summary
#     in_stock: bool
#     variant_count: int = Field(
#         default=1,
#         description="Total number of purchasable variants.",
#     )

#     # Ranking metadata (not shown to user — used by Orchestrator)
#     relevance_score: float = Field(
#         default=0.0,
#         ge=0.0,
#         le=1.0,
#         description="Combined semantic + business rank score. "
#                     "Higher = more relevant to this specific query.",
#     )
#     semantic_score: float = Field(
#         default=0.0,
#         ge=0.0,
#         le=1.0,
#         description="Raw Milvus cosine similarity score.",
#     )
#     rating: float | None = Field(default=None, ge=0.0, le=5.0)
#     review_count: int = Field(default=0)

#     @property
#     def discount_percent(self) -> float | None:
#         """Computed discount percentage for display. None if no active discount."""
#         if self.compare_at_price and self.compare_at_price > self.base_price:
#             return round(
#                 (1 - self.base_price / self.compare_at_price) * 100, 1
#             )
#         return None

#     @property
#     def is_on_sale(self) -> bool:
#         return self.discount_percent is not None

class ProductCard(BaseModel):
    product_id: str
    product_code: str
    product_name: str
    url_slug: str
    images: list[dict]
    relevance_score: float


class ProductDetail(BaseModel):
    """
    Full product representation for the product detail view.

    Returned by the get_product_details tool.
    Superset of ProductCard — contains everything needed for
    a full product page including all variants, full description,
    specifications, and related product IDs for cross-sell (Phase 3).
    """

    # Core identity
    product_id: str
    entity_id: str
    business_unit_id: str
    name: str
    brand: str | None = None
    category: str
    sub_category: str | None = None
    family: str | None = Field(
        default=None,
        description="Enterprise catalog family. e.g. 'Footwear', 'Saree'",
    )

    # Content
    description: str = Field(description="Full product description.")
    short_description: str = Field(default="")
    images: list[dict] = Field(default_factory=list)

    # Pricing
    currency: str = Field(default="INR")

    # All purchasable variants
    variants: list[ProductVariant] = Field(
        default_factory=list,
        description="All available variants. Phase 2 cart agent uses variant_id.",
    )

    # Full attribute set
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete product attributes from catalog. "
                    "Includes both SMB fixed and Enterprise family attributes.",
    )
    specifications: dict[str, Any] = Field(
        default_factory=dict,
        description="Technical specifications. Varies by category.",
    )

    # Social proof
    rating: float | None = Field(default=None, ge=0.0, le=5.0)
    review_count: int = Field(default=0)

    # Catalog metadata
    created_at: datetime | None = None
    updated_at: datetime | None = None
    is_active: bool = Field(default=True)

    # Phase 3 readiness — cross-sell hooks
    related_product_ids: list[str] = Field(
        default_factory=list,
        description="Pre-computed related product IDs for cross-sell. "
                    "Populated from catalog data. Used in Phase 3.",
    )
    frequently_bought_with: list[str] = Field(
        default_factory=list,
        description="Co-purchase product IDs. Used in Phase 3.",
    )

    def to_card(self) -> ProductCard:
        """
        Downcasts a ProductDetail to a ProductCard.
        Used when detail fetch result needs to appear in a search result list.
        """
        # cheapest_variant = min(
        #     self.variants, key=lambda v: v.price, default=None
        # )
        return ProductCard(
            product_id=self.product_id,
            product_code=self.attributes.get("product_code", ""),
            product_name=self.name,
            url_slug=self.attributes.get("url_slug", ""),
            images=self.images,
            relevance_score=0.0,  # To be populated by ranker if used in search results
        )