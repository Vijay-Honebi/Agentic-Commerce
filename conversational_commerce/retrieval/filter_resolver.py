# conversational_commerce/retrieval/filter_resolver.py

from dataclasses import dataclass
from typing import Any

from schemas.query import HardConstraints


@dataclass
class SQLFilter:
    joins: list[str]
    where_clauses: list[str]
    params: list[Any]


class FilterResolver:

    def resolve(self, constraints: HardConstraints, start_index: int = 1) -> SQLFilter:
        joins: list[str] = []
        where_clauses: list[str] = []
        params: list[Any] = []

        param_index = start_index

        # ── Dynamic filters (attribute resolution) ─────────────────────
        for key, value in constraints.dynamic_filters.items():
            if value is None:
                continue

            attr_key = str(key).strip().lower()

            if isinstance(value, str):
                attr_value = value.strip().lower()
            else:
                attr_value = value

            alias = f"pa_{attr_key}"

            joins.append(
                f"""
                JOIN product_attributes {alias}
                  ON {alias}.product_id = p.id
                 AND LOWER({alias}.attribute_name) = ${param_index}
                """
            )
            params.append(attr_key)
            param_index += 1

            where_clauses.append(f"LOWER({alias}.attribute_value) = ${param_index}")
            params.append(attr_value)
            param_index += 1

        return SQLFilter(
            joins=joins,
            where_clauses=where_clauses,
            params=params,
        )