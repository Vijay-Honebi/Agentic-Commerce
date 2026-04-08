# conversational_commerce/guardrails/discovery_guard.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config.settings import get_settings
from observability.logger import LogEvent, get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Violation taxonomy
# ---------------------------------------------------------------------------

class DiscoveryViolationType(str, Enum):
    """
    Every possible guardrail violation the Discovery Agent can produce.

    Each violation type maps to a specific sanitisation strategy.
    Violations are logged with full context for engineering review.
    """

    # Product integrity violations
    HALLUCINATED_PRODUCT_ID = "hallucinated_product_id"
    # Agent returned a product_id not present in the retrieval result set.
    # Root cause: LLM invented a product ID instead of using tool output.

    PRODUCT_NOT_IN_RETRIEVAL_SET = "product_not_in_retrieval_set"
    # Agent referenced a product that was not returned by search_products tool.
    # Could indicate prompt injection or context bleed between sessions.

    # Pricing violations
    HALLUCINATED_DISCOUNT = "hallucinated_discount"
    # Agent stated a discount % not present in the product data.
    # Critical: could cause merchant financial loss if acted upon.

    PRICE_MISMATCH = "price_mismatch"
    # Agent stated a price different from the catalog price.
    # Tolerance: ±0.01 (floating point rounding only).

    UNAVAILABLE_PRODUCT_PROMOTED = "unavailable_product_promoted"
    # Agent recommended an out-of-stock product as available.

    # Business logic violations
    BUSINESS_RULE_BYPASS = "business_rule_bypass"
    # Agent result violates a hard business constraint from AgentRequest.
    # e.g. Result includes products outside the allowed store scope.

    STORE_SCOPE_VIOLATION = "store_scope_violation"
    # Results include products from a different store than requested.
    # Tenant isolation breach — highest severity.

    # Response quality violations
    EMPTY_RESULT_WITHOUT_FLAG = "empty_result_without_flag"
    # Agent returned zero products without flagging it as an empty result.
    # Synthesizer needs to know so it can ask clarifying questions.

    RESULT_LIMIT_EXCEEDED = "result_limit_exceeded"
    # Agent returned more products than result_limit requested.
    # Truncated automatically — logged as warning not error.

    CONFIDENCE_TOO_LOW = "confidence_too_low"
    # AgentResult.confidence below threshold — result quality unreliable.


class ViolationSeverity(str, Enum):
    """
    Severity determines what happens when a violation fires:
      CRITICAL → Block response entirely, return safe fallback
      HIGH     → Sanitise the specific violation, log error
      MEDIUM   → Sanitise and log warning
      LOW      → Log only, pass through
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Severity mapping — determines response for each violation type
VIOLATION_SEVERITY: dict[DiscoveryViolationType, ViolationSeverity] = {
    DiscoveryViolationType.HALLUCINATED_PRODUCT_ID:         ViolationSeverity.CRITICAL,
    DiscoveryViolationType.PRODUCT_NOT_IN_RETRIEVAL_SET:    ViolationSeverity.CRITICAL,
    DiscoveryViolationType.HALLUCINATED_DISCOUNT:           ViolationSeverity.CRITICAL,
    DiscoveryViolationType.STORE_SCOPE_VIOLATION:           ViolationSeverity.CRITICAL,
    DiscoveryViolationType.PRICE_MISMATCH:                  ViolationSeverity.HIGH,
    DiscoveryViolationType.UNAVAILABLE_PRODUCT_PROMOTED:    ViolationSeverity.HIGH,
    DiscoveryViolationType.BUSINESS_RULE_BYPASS:            ViolationSeverity.HIGH,
    DiscoveryViolationType.EMPTY_RESULT_WITHOUT_FLAG:       ViolationSeverity.MEDIUM,
    DiscoveryViolationType.RESULT_LIMIT_EXCEEDED:           ViolationSeverity.LOW,
    DiscoveryViolationType.CONFIDENCE_TOO_LOW:              ViolationSeverity.MEDIUM,
}


# ---------------------------------------------------------------------------
# Guardrail result
# ---------------------------------------------------------------------------

@dataclass
class GuardrailViolation:
    """A single detected violation with full context for logging and audit."""

    violation_type: DiscoveryViolationType
    severity: ViolationSeverity
    description: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "context": self.context,
        }


@dataclass
class DiscoveryGuardrailResult:
    """
    Output of the Discovery guardrail check.

    passed:     True if no CRITICAL or HIGH violations found.
                Orchestrator proceeds with sanitised result.
    violations: All detected violations (any severity).
    sanitised_products: Product list after removing violating items.
                        May be smaller than input if violations removed products.
    should_block: True if any CRITICAL violation found.
                  Orchestrator returns safe fallback, not the agent result.
    """

    passed: bool
    violations: list[GuardrailViolation]
    sanitised_products: list[dict[str, Any]]
    should_block: bool
    block_reason: str | None = None

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    @property
    def critical_violations(self) -> list[GuardrailViolation]:
        return [
            v for v in self.violations
            if v.severity == ViolationSeverity.CRITICAL
        ]

    @property
    def high_violations(self) -> list[GuardrailViolation]:
        return [
            v for v in self.violations
            if v.severity == ViolationSeverity.HIGH
        ]


# ---------------------------------------------------------------------------
# Discovery Guardrail
# ---------------------------------------------------------------------------

class DiscoveryGuardrail:
    """
    Validates Discovery Agent output before the Orchestrator synthesizes
    it into a user-facing response.

    Checks performed (in order):
      1. Confidence threshold
      2. Product ID integrity (no hallucinated IDs)
      3. Store scope (tenant isolation)
      4. Price integrity (no price mismatches)
      5. Discount integrity (no hallucinated discounts)
      6. Stock status (no unavailable products promoted)
      7. Result limit enforcement
      8. Empty result detection

    Stateless — one instance shared across all requests.
    """

    # Minimum confidence for agent results to be accepted
    MIN_CONFIDENCE: float = 0.50

    # Price mismatch tolerance (floating point rounding only)
    PRICE_TOLERANCE: float = 0.01

    def validate(
        self,
        agent_result_data: dict[str, Any],
        retrieval_product_ids: set[str],
        requested_entity_id: str | None,
        requested_business_unit_id: str | None,
        requested_result_limit: int,
        agent_confidence: float = 1.0,
    ) -> DiscoveryGuardrailResult:
        """
        Validates the raw data dict from the Discovery Agent's AgentResult.

        Args:
            agent_result_data:      AgentResult.data from DiscoveryAgent.
            retrieval_product_ids:  Set of product_ids returned by search_products tool.
                                    Ground truth — any ID not in this set is hallucinated.
            requested_entity_id:   Entity scope from the original AgentRequest. All results must match this if set.,
            requested_business_unit_id: Business unit scope from the original AgentRequest. All results must match this if set.
            requested_result_limit: From ParsedQuery — enforced here.
            agent_confidence:       AgentResult.confidence.

        Returns:
            DiscoveryGuardrailResult with pass/fail, violations, and sanitised data.
        """
        violations: list[GuardrailViolation] = []
        products: list[dict[str, Any]] = agent_result_data.get("products", [])

        # ── Check 1: Confidence threshold ─────────────────────────────────
        if agent_confidence < self.MIN_CONFIDENCE:
            violations.append(GuardrailViolation(
                violation_type=DiscoveryViolationType.CONFIDENCE_TOO_LOW,
                severity=VIOLATION_SEVERITY[DiscoveryViolationType.CONFIDENCE_TOO_LOW],
                description=(
                    f"Agent confidence {agent_confidence:.2f} below "
                    f"minimum threshold {self.MIN_CONFIDENCE}"
                ),
                context={"confidence": agent_confidence, "threshold": self.MIN_CONFIDENCE},
            ))

        # ── Check 2: Empty result detection ───────────────────────────────
        if not products:
            # Empty result is valid — but must be explicitly flagged
            # so the synthesizer asks a clarifying question
            if not agent_result_data.get("is_empty_result", False):
                violations.append(GuardrailViolation(
                    violation_type=DiscoveryViolationType.EMPTY_RESULT_WITHOUT_FLAG,
                    severity=VIOLATION_SEVERITY[
                        DiscoveryViolationType.EMPTY_RESULT_WITHOUT_FLAG
                    ],
                    description="Empty product list returned without is_empty_result flag",
                    context={},
                ))
            # No products to validate — return early
            return self._build_result(
                violations=violations,
                sanitised_products=[],
            )

        # ── Check 3: Result limit enforcement ─────────────────────────────
        if len(products) > requested_result_limit:
            violations.append(GuardrailViolation(
                violation_type=DiscoveryViolationType.RESULT_LIMIT_EXCEEDED,
                severity=VIOLATION_SEVERITY[DiscoveryViolationType.RESULT_LIMIT_EXCEEDED],
                description=(
                    f"Agent returned {len(products)} products, "
                    f"limit was {requested_result_limit}. Auto-truncated."
                ),
                context={
                    "returned": len(products),
                    "limit": requested_result_limit,
                },
            ))
            products = products[:requested_result_limit]

        # ── Per-product validation ─────────────────────────────────────────
        sanitised: list[dict[str, Any]] = []

        for product in products:
            product_id = product.get("product_id", "")
            product_violations: list[GuardrailViolation] = []

            # Check 4: Product ID integrity
            if not product_id:
                product_violations.append(GuardrailViolation(
                    violation_type=DiscoveryViolationType.HALLUCINATED_PRODUCT_ID,
                    severity=VIOLATION_SEVERITY[
                        DiscoveryViolationType.HALLUCINATED_PRODUCT_ID
                    ],
                    description="Product with empty product_id detected",
                    context={"product_name": product.get("name", "unknown")},
                ))

            elif product_id not in retrieval_product_ids:
                product_violations.append(GuardrailViolation(
                    violation_type=DiscoveryViolationType.PRODUCT_NOT_IN_RETRIEVAL_SET,
                    severity=VIOLATION_SEVERITY[
                        DiscoveryViolationType.PRODUCT_NOT_IN_RETRIEVAL_SET
                    ],
                    description=(
                        f"Product ID '{product_id}' was not in the retrieval result set. "
                        f"Possible hallucination or context bleed."
                    ),
                    context={
                        "product_id": product_id,
                        "product_name": product.get("name", "unknown"),
                    },
                ))

            # Check 5: Entity and business unit scope violations
            # If the request specified an entity_id or business_unit_id, all results must match
            if requested_entity_id and product.get("entity_id") != requested_entity_id:
                product_violations.append(GuardrailViolation(
                    violation_type=DiscoveryViolationType.BUSINESS_RULE_BYPASS,
                    severity=VIOLATION_SEVERITY[
                        DiscoveryViolationType.BUSINESS_RULE_BYPASS
                    ],
                    description=(
                        f"Product '{product_id}' has entity_id '{product.get('entity_id')}' "
                        f"which does not match requested entity_id '{requested_entity_id}'"
                    ),
                    context={
                        "product_id": product_id,
                        "product_entity_id": product.get("entity_id"),
                        "requested_entity_id": requested_entity_id,
                    },
                ))

            # Check 6: Discount integrity
            # The synthesizer must not state discount percentages
            # not present in the catalog data
            if product.get("is_on_sale"):
                discount_pct = product.get("discount_percent")
                base_price = product.get("base_price")
                compare_at = product.get("compare_at_price")

                if (
                    discount_pct is not None
                    and base_price is not None
                    and compare_at is not None
                    and compare_at > 0
                ):
                    expected_discount = round(
                        (1 - base_price / compare_at) * 100, 1
                    )
                    if abs(discount_pct - expected_discount) > 1.0:
                        product_violations.append(GuardrailViolation(
                            violation_type=DiscoveryViolationType.HALLUCINATED_DISCOUNT,
                            severity=VIOLATION_SEVERITY[
                                DiscoveryViolationType.HALLUCINATED_DISCOUNT
                            ],
                            description=(
                                f"Discount mismatch for '{product_id}': "
                                f"stated {discount_pct}%, "
                                f"computed {expected_discount}%"
                            ),
                            context={
                                "product_id": product_id,
                                "stated_discount": discount_pct,
                                "computed_discount": expected_discount,
                                "base_price": base_price,
                                "compare_at_price": compare_at,
                            },
                        ))
                        # Correct the discount in-place — don't drop the product
                        product = {
                            **product,
                            "discount_percent": expected_discount,
                        }

            # Check 7: Stock status
            if not product.get("in_stock", True):
                product_violations.append(GuardrailViolation(
                    violation_type=DiscoveryViolationType.UNAVAILABLE_PRODUCT_PROMOTED,
                    severity=VIOLATION_SEVERITY[
                        DiscoveryViolationType.UNAVAILABLE_PRODUCT_PROMOTED
                    ],
                    description=(
                        f"Out-of-stock product '{product_id}' "
                        f"included in discovery results"
                    ),
                    context={"product_id": product_id},
                ))

            violations.extend(product_violations)

            # Drop product if it has any CRITICAL violations
            # Keep product (possibly sanitised) for HIGH/MEDIUM/LOW
            has_critical = any(
                v.severity == ViolationSeverity.CRITICAL
                for v in product_violations
            )
            if not has_critical:
                sanitised.append(product)

        return self._build_result(
            violations=violations,
            sanitised_products=sanitised,
        )

    def _build_result(
        self,
        violations: list[GuardrailViolation],
        sanitised_products: list[dict[str, Any]],
    ) -> DiscoveryGuardrailResult:
        """
        Builds the guardrail result and emits structured logs.
        """
        has_critical = any(
            v.severity == ViolationSeverity.CRITICAL
            for v in violations
        )
        should_block = has_critical
        passed = not has_critical and not any(
            v.severity == ViolationSeverity.HIGH for v in violations
        )

        if violations:
            log_level = "error" if should_block else "warning"
            getattr(logger, log_level)(
                LogEvent.GUARDRAIL_VIOLATION,
                "Discovery guardrail violations detected",
                violation_count=len(violations),
                critical_count=sum(
                    1 for v in violations
                    if v.severity == ViolationSeverity.CRITICAL
                ),
                high_count=sum(
                    1 for v in violations
                    if v.severity == ViolationSeverity.HIGH
                ),
                should_block=should_block,
                violations=[v.to_dict() for v in violations],
            )
        else:
            logger.info(
                LogEvent.GUARDRAIL_PASS,
                "Discovery guardrail passed",
                product_count=len(sanitised_products),
            )

        return DiscoveryGuardrailResult(
            passed=passed,
            violations=violations,
            sanitised_products=sanitised_products,
            should_block=should_block,
            block_reason=(
                f"{len([v for v in violations if v.severity == ViolationSeverity.CRITICAL])} "
                f"critical violations detected"
                if should_block else None
            ),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_discovery_guardrail: DiscoveryGuardrail | None = None


def get_discovery_guardrail() -> DiscoveryGuardrail:
    """Returns the singleton DiscoveryGuardrail instance."""
    global _discovery_guardrail
    if _discovery_guardrail is None:
        _discovery_guardrail = DiscoveryGuardrail()
    return _discovery_guardrail