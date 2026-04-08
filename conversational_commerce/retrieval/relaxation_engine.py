# conversational_commerce/retrieval/relaxation_engine.py

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Awaitable

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.query import HardConstraints, ParsedQuery, PriceRange, RelaxationRecord

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RelaxationResult:
    """
    Outcome of a relaxation attempt.
    Returned to hybrid_retriever.py which decides whether to retry.
    """

    relaxed_constraints: HardConstraints
    records: list[RelaxationRecord]
    rounds_applied: int
    was_triggered: bool         # False if threshold was met without relaxation


# ---------------------------------------------------------------------------
# Relaxation Engine
# ---------------------------------------------------------------------------

class RelaxationEngine:
    """
    Adaptively loosens HardConstraints when candidate count falls below
    settings.milvus.min_candidates_threshold.

    Relaxation strategy (priority order):
      Round 1 → Loosen price range by step_percent on each bound
      Round 2 → Drop lowest-priority attribute constraints
                (color → material → size in that order)

    Hard limits:
      - Maximum rounds = settings.relaxation.max_rounds (default: 2)
      - Price never goes below 0
      - in_stock_only is NEVER relaxed — business rule
      - entity_id is NEVER relaxed — tenant isolation rule
      - business_unit_id is NEVER relaxed — tenant isolation rule
      - Every relaxation is logged with full audit trail

    The engine does NOT re-run retrieval itself.
    It returns relaxed constraints and the hybrid_retriever re-runs.
    This keeps the engine pure and testable.
    """

    def __init__(self) -> None:
        self._cfg = settings.relaxation
        self._milvus_cfg = settings.milvus

    def should_relax(self, candidate_count: int) -> bool:
        """True if candidate count is below the configured threshold."""
        return candidate_count < self._milvus_cfg.min_candidates_threshold

    def relax(
        self,
        parsed_query: ParsedQuery,
        candidate_count_before: int,
        round_number: int,
    ) -> tuple[HardConstraints, list[RelaxationRecord]] | None:
        """
        Attempts one round of constraint relaxation.

        Args:
            parsed_query:           Current query with constraints to relax.
            candidate_count_before: Candidate count that triggered relaxation.
            round_number:           Which round this is (1-indexed).

        Returns:
            (relaxed_constraints, records) if relaxation was possible.
            None if no further relaxation is possible (all constraints exhausted).
        """
        if round_number > self._cfg.max_rounds:
            logger.warning(
                LogEvent.RETRIEVAL_RELAXATION_TRIGGERED,
                "Max relaxation rounds reached — returning best-effort results",
                max_rounds=self._cfg.max_rounds,
                candidate_count=candidate_count_before,
            )
            return None

        logger.info(
            LogEvent.RETRIEVAL_RELAXATION_TRIGGERED,
            "Relaxation triggered",
            round_number=round_number,
            candidate_count_before=candidate_count_before,
            threshold=self._milvus_cfg.min_candidates_threshold,
        )

        # Deep copy — never mutate the original constraints
        constraints = copy.deepcopy(parsed_query.hard_constraints)
        records: list[RelaxationRecord] = []

        if round_number == 1:
            record = self._relax_price(constraints, candidate_count_before)
            if record:
                records.append(record)

        elif round_number == 2:
            # Drop attributes in order of least impact on purchase intent
            for field in ("color", "material", "size", "sub_category"):
                record = self._drop_attribute(
                    constraints, field, candidate_count_before
                )
                if record:
                    records.append(record)
                    break  # Drop one attribute per round, then re-evaluate

        if not records:
            logger.info(
                LogEvent.RETRIEVAL_RELAXATION_ROUND,
                "No relaxable constraints remain",
                round_number=round_number,
            )
            return None

        for record in records:
            logger.info(
                LogEvent.RETRIEVAL_RELAXATION_ROUND,
                "Constraint relaxed",
                round_number=round_number,
                field=record.field_relaxed,
                original=record.original_value,
                relaxed=record.relaxed_value,
                step_percent=self._cfg.step_percent,
            )

        return constraints, records

    # ── Private relaxation strategies ─────────────────────────────────────

    def _relax_price(
        self,
        constraints: HardConstraints,
        candidate_count_before: int,
    ) -> RelaxationRecord | None:
        """
        Widens the price range by step_percent on each bound.
        min_price is lowered. max_price is raised.
        """
        if constraints.price_range is None:
            return None

        step = self._cfg.step_percent
        original_range = copy.deepcopy(constraints.price_range)

        new_min = None
        new_max = None

        if constraints.price_range.min_price is not None:
            new_min = max(0.0, constraints.price_range.min_price * (1 - step))

        if constraints.price_range.max_price is not None:
            new_max = constraints.price_range.max_price * (1 + step)

        constraints.price_range = PriceRange(
            min_price=new_min,
            max_price=new_max,
        )

        return RelaxationRecord(
            round_number=1,
            field_relaxed="price_range",
            original_value={
                "min": original_range.min_price,
                "max": original_range.max_price,
            },
            relaxed_value={
                "min": new_min,
                "max": new_max,
            },
            candidate_count_before=candidate_count_before,
            candidate_count_after=0,  # Filled by hybrid_retriever after re-run
            relaxation_percent=step,
        )

    def _drop_attribute(
        self,
        constraints: HardConstraints,
        field: str,
        candidate_count_before: int,
    ) -> RelaxationRecord | None:
        """
        Drops a single attribute constraint entirely.
        Returns None if the field was already None (nothing to drop).
        """
        original_value = getattr(constraints, field, None)
        if original_value is None:
            return None

        # Set field to None — removes it from PSQL WHERE clause
        object.__setattr__(constraints, field, None)

        return RelaxationRecord(
            round_number=2,
            field_relaxed=field,
            original_value=original_value,
            relaxed_value=None,
            candidate_count_before=candidate_count_before,
            candidate_count_after=0,  # Filled by hybrid_retriever after re-run
            relaxation_percent=1.0,   # 100% — full drop
        )