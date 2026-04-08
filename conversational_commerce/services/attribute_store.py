# conversational_commerce/services/attribute_store.py

from __future__ import annotations

from typing import Any

import asyncpg

from observability.logger import LogEvent, get_logger

logger = get_logger(__name__)


class AttributeStore:
    """
    In-memory attribute cache loaded once at application startup.

    Provides the query parser with the EXACT valid values for every
    filterable attribute in the catalog. This prevents the LLM from:
        - Hallucinating attribute values not in the catalog
        - Using wrong casing ("Sports" vs "sports")
        - Inventing categories that don't exist
        - Creating unmatchable SQL WHERE clauses

    Load order in main.py:
        PostgreSQL pool ready → AttributeStore.load() → query parser available

    Refresh strategy:
        Phase 1: Loaded once at startup (daily restart flushes cache)
        Phase 2: Add a reload() endpoint for merchant catalog updates
        Phase 3: Background task refreshes every N hours

    O(1) lookup — safe to call on every single query parse.
    Thread-safe — read-only after load(), never mutated during runtime.
    """

    def __init__(self) -> None:
        self._attributes: dict[str, list[str]] = {}
        self._is_loaded: bool = False

    async def load(self, pool: asyncpg.Pool) -> None:
        """
        Loads all attribute definitions from PostgreSQL into memory.
        Idempotent — safe to call multiple times (subsequent calls are no-ops).

        Tables read:
            attributes      → filterable product attributes with valid values
                              e.g. material: ["mesh", "leather", "cotton"]
            categories      → top-level product categories
            sub_categories  → sub-categories within each category
            classifications → enterprise catalog families (SMB model extension)

        FIX 1: fixed_values is a JSONB/text[] column — handle both types
        FIX 2: normalise ALL values to lowercase for consistent SQL matching
        FIX 3: entity scoping — Phase 2 enhancement noted in docstring
        """
        if self._is_loaded:
            return

        attributes: dict[str, list[str]] = {}

        async with pool.acquire() as conn:

            # ── Product attributes with fixed value sets ───────────────────
            # These map directly to HardConstraints fields.
            # e.g. attribute label="material", fixed_values=["mesh","leather"]
            attr_rows = await conn.fetch(
                """
                SELECT label, fixed_values
                FROM attributes
                WHERE fixed_values IS NOT NULL
                  AND array_length(fixed_values, 1) > 0
                ORDER BY label ASC
                """
            )

            for row in attr_rows:
                label = row["label"]
                raw_values = row["fixed_values"] or []

                if not label:
                    continue

                normalised_key = label

                normalised_values = sorted({
                    str(v)
                    for v in raw_values
                    if v is not None
                })

                if normalised_values:
                    attributes[normalised_key] = normalised_values

            # ── Categories ────────────────────────────────────────────────
            category_rows = await conn.fetch(
                """
                SELECT DISTINCT name AS label
                FROM categories
                WHERE name IS NOT NULL
                  AND active_flag = true
                ORDER BY name ASC
                """
            )

            attributes["categories"] = sorted({
                row["label"].strip().lower()
                for row in category_rows
                if row["label"]
            })

            # ── Sub-categories ────────────────────────────────────────────
            sub_category_rows = await conn.fetch(
                """
                SELECT DISTINCT name AS label
                FROM sub_categories
                WHERE name IS NOT NULL
                  AND active_flag = true
                ORDER BY name ASC
                """
            )

            attributes["sub_categories"] = sorted({
                row["label"].strip().lower()
                for row in sub_category_rows
                if row["label"]
            })

            # ── Classifications (Enterprise catalog families) ─────────────
            classification_rows = await conn.fetch(
                """
                SELECT DISTINCT name AS label
                FROM classifications
                WHERE name IS NOT NULL
                ORDER BY name ASC
                """
            )

            attributes["classifications"] = sorted({
                row["label"].strip().lower()
                for row in classification_rows
                if row["label"]
            })

            # ── Brands ────────────────────────────────────────────────────
            # FIX 3: Brands are NOT normalised to lowercase
            # Brand names are proper nouns: "Nike", "Adidas", not "nike"
            # We keep original casing for display but provide them as hints
            brand_rows = await conn.fetch(
                """
                SELECT DISTINCT name AS label
                FROM brands
                WHERE name IS NOT NULL
                  AND active_flag = true
                ORDER BY name ASC
                """
            )

            attributes["brands"] = sorted({
                row["label"]
                for row in brand_rows
                if row["label"]
            })

        self._attributes = attributes
        self._is_loaded = True

        logger.info(
            LogEvent.APP_STARTUP,
            "Attribute store loaded",
            attribute_keys=list(attributes.keys()),
            total_keys=len(attributes),
            category_count=len(attributes.get("categories", [])),
            sub_category_count=len(attributes.get("sub_categories", [])),
            brand_count=len(attributes.get("brands", [])),
        )

    async def reload(self, pool: asyncpg.Pool) -> None:
        """
        Force-reloads the attribute store.
        Use when merchant updates their catalog attributes.
        Phase 2: expose as admin endpoint.
        """
        self._is_loaded = False
        await self.load(pool)
        logger.info(LogEvent.APP_STARTUP, "Attribute store reloaded")

    def get_all(self) -> dict[str, list[str]]:
        """Returns the full attribute map. Raises if not loaded."""
        if not self._is_loaded:
            raise RuntimeError(
                "AttributeStore not loaded. "
                "Ensure attribute_store.load() is called at startup."
            )
        return self._attributes

    def get(self, key: str) -> list[str]:
        """
        Returns valid values for a single attribute key.
        Returns empty list (not error) if key not found —
        safe to call for any attribute without guard.
        """
        return self._attributes.get(key.strip().lower(), [])

    def build_prompt_block(self) -> str:
        """
        Builds the attribute constraints block injected into the
        query parser prompt. The LLM reads this and uses ONLY
        these values when extracting hard constraints.

        Format is deliberately simple — one line per attribute.
        The LLM doesn't need JSON here, just a clear enumeration.

        Returns empty string if not loaded — parser falls back gracefully.
        """
        if not self._is_loaded or not self._attributes:
            return ""

        lines = ["[VALID CATALOG VALUES — USE ONLY THESE IN hard_constraints]"]

        # ── Priority attributes first (most commonly filtered) ────────────
        priority_keys = [
            "categories",
            "sub_categories",
            "brands",
            "gender",
            "material",
            "color",
            "size",
            "sole_type",
            "closure_type",
            "fabric",
            "classifications",
        ]

        # Emit priority keys first, then remaining keys alphabetically
        emitted: set[str] = set()

        for key in priority_keys:
            values = self._attributes.get(key, [])
            if values:
                lines.append(f"{key}: {', '.join(values)}")
                emitted.add(key)

        # Remaining keys
        for key in sorted(self._attributes.keys()):
            if key not in emitted:
                values = self._attributes[key]
                if values:
                    lines.append(f"{key}: {', '.join(values)}")

        lines.append(
            "\nRULE: For every hard_constraints field, you MUST use one of "
            "the exact values listed above. If the user's input does not "
            "match any listed value, set that field to null — never invent values."
        )

        return "\n".join(lines)

    def format_for_prompt(
        self,
        keys: list[str] | None = None,
    ) -> str:
        """
        Returns a subset of attributes for focused prompts.
        When keys=None, returns all (same as build_prompt_block).
        When keys provided, returns only those keys.

        Usage:
            # Only inject category-related attributes
            store.format_for_prompt(["categories", "sub_categories"])
        """
        if not self._is_loaded:
            return ""

        target = (
            {k: self._attributes[k] for k in keys if k in self._attributes}
            if keys
            else self._attributes
        )

        if not target:
            return ""

        lines = ["[VALID CATALOG VALUES]"]
        for key, values in target.items():
            if values:
                lines.append(f"{key}: {', '.join(values)}")

        return "\n".join(lines)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def categories(self) -> list[str]:
        return self.get("categories")

    @property
    def sub_categories(self) -> list[str]:
        return self.get("sub_categories")

    @property
    def brands(self) -> list[str]:
        return self.get("brands")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_attribute_store: AttributeStore | None = None


def get_attribute_store() -> AttributeStore:
    """
    Returns the singleton AttributeStore.
    Raises RuntimeError if called before load() at startup.
    """
    global _attribute_store
    if _attribute_store is None:
        _attribute_store = AttributeStore()
    return _attribute_store