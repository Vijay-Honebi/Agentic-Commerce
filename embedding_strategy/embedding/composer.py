from dataclasses import dataclass
import re
from typing import Optional
from embedding_strategy.observability.logger import StructuredLogger

logger = StructuredLogger("embedding.composer")


@dataclass(frozen=True)
class GlobalProduct:
    global_product_id: str
    name: str
    bu_ids: list[str]
    category_l1: Optional[str]      # category.name
    category_l2: Optional[str]      # sub_category.name
    category_l3: Optional[str]      # classification.name — NULL for current clients
    brand_name: Optional[str]
    attributes: dict[str, str]      # dynamic JSONB — all specs as key:value
    description: Optional[str]      # p.short_description
    updated_at: int                 # EXTRACT(EPOCH FROM p.modified_at)

class EmbeddingComposer:
    """
    Composes a rich, semantically dense text string from a GlobalProduct.

    Design principles:
    - Ordered from most → least discriminative signal
    - Pipe-separated sections for clear boundary encoding
    - None values are gracefully excluded — no "None" strings in vectors
    - Deterministic: same input always produces same output
    """

    _SKU_PATTERN = re.compile(r'\b[A-Z0-9]{2,}-[A-Z0-9\-]{3,}\b')

    def _clean_name(self, name: str) -> str:
        """
        Strips SKU codes from product names before embedding.
        'Turtle Casual T-SHIRT OTTS-84173-1002-H-POLO'
        → 'Turtle Casual T-SHIRT'
        """
        cleaned = self._SKU_PATTERN.sub('', name)
        return ' '.join(cleaned.split()).strip()

    # Fixed fields ordered by semantic weight
    _FIXED_FIELD_ORDER = [
        "name",
        "category_path",
        "brand_name",
        "description",
    ]

    def compose(self, product: GlobalProduct) -> str:
        category_path = self._build_category_path(product)

        fixed_map = {
            "name": self._clean_name(product.name) if product.name else None,
            "category_path": category_path,
            "brand_name": product.brand_name,
            "description": product.description,
        }

        # Fixed fields first
        parts = [
            fixed_map[field]
            for field in self._FIXED_FIELD_ORDER
            if fixed_map.get(field)
        ]

        # Dynamic attributes appended after fixed fields
        # Format: "material: Synthetic | style: Casual | sole type: Rubber"
        # Key-value format preserves semantic meaning of each attribute
        if product.attributes:
            attr_parts = [
                f"{k}: {v}"
                for k, v in sorted(product.attributes.items())
            ]
            parts.extend(attr_parts)

        composed = "\n".join(parts)

        logger.debug(
            "Embedding text composed",
            global_product_id=product.global_product_id,
            composed_length=len(composed),
            fixed_fields=len([f for f in self._FIXED_FIELD_ORDER if fixed_map.get(f)]),
            dynamic_attributes=len(product.attributes),
        )

        return composed
# ```

# Example output for a badminton shoe with dynamic attributes:
# ```
# Lightweight Badminton Shoe | Shoes > Sports | Yonex | Breathable indoor sports shoe | material: Synthetic | sole type: Rubber | closure type: Lace-up | gender: Unisex | style: Athletic

    def _build_category_path(self, product: GlobalProduct) -> str:
        levels = [
            product.category_l1,
            product.category_l2,
            product.category_l3,        # NULL for current clients — excluded cleanly
        ]
        return " > ".join(level for level in levels if level)

    def build_category_path_for_filter(self, product: GlobalProduct) -> str:
        """Exposed separately for Milvus scalar field population."""
        return self._build_category_path(product)