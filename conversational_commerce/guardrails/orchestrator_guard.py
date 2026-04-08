# conversational_commerce/guardrails/orchestrator_guard.py

from __future__ import annotations

import re
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

class OrchestratorViolationType(str, Enum):
    """
    Violations at the Orchestrator response level.
    These fire AFTER the Discovery guardrail — final check before the user.

    These violations catch issues the synthesizer LLM might introduce
    when converting structured agent data into natural language.
    """

    # Fabrication violations
    PRICE_FABRICATED_IN_RESPONSE = "price_fabricated_in_response"
    # Synthesizer stated a price not present in any product in the result set.

    DISCOUNT_FABRICATED_IN_RESPONSE = "discount_fabricated_in_response"
    # Synthesizer stated a discount % not present in any product data.
    # e.g. "This product is 40% off!" when catalog says 25%.

    PRODUCT_NAME_FABRICATED = "product_name_fabricated"
    # Synthesizer mentioned a product name not in the result set.

    PROMOTION_NOT_IN_CATALOG = "promotion_not_in_catalog"
    # Synthesizer invented a promotion (BOGO, free shipping, etc.)
    # not present in the active_promotions passed to it.

    # Business logic violations
    ILLEGAL_PROMOTION_PROMISE = "illegal_promotion_promise"
    # Synthesizer promised a promotion that exceeds merchant-set limits.
    # Phase 3/4: discount floor enforcement lives here.

    COMPETITOR_MENTIONED = "competitor_mentioned"
    # Synthesizer mentioned a competitor brand in a promotional context.

    # Safety violations
    PERSONAL_DATA_LEAK = "personal_data_leak"
    # Response contains patterns matching PII (email, phone, address).

    INTERNAL_SYSTEM_LEAK = "internal_system_leak"
    # Response contains internal identifiers (product_id, session_id, tool names).
    # Users should never see raw system IDs in conversational responses.

    # Quality violations
    RESPONSE_TOO_SHORT = "response_too_short"
    # Synthesizer returned a one-word or empty response.

    RESPONSE_TOO_LONG = "response_too_long"
    # Synthesizer exceeded reasonable response length — likely hallucinating.

    REFUSAL_WITHOUT_ALTERNATIVE = "refusal_without_alternative"
    # Synthesizer said "I can't help" without offering a related alternative.
    # We are a salesman — we never give up without offering something.


# ---------------------------------------------------------------------------
# Promotion limit enforcement (Phase 3/4 seed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionLimits:
    """
    Merchant-configured promotion limits.
    The Orchestrator guardrail enforces these — the synthesizer cannot exceed them.

    Phase 3: these are loaded dynamically from merchant settings.
    Phase 4: these have per-customer overrides with hard floors.
    For Phase 1: sensible defaults, no promotions active.
    """
    max_discount_percent: float = 70.0      # Absolute maximum any promotion can offer
    min_discount_percent: float = 0.0       # Minimum (floor) — Phase 4 enforces this
    allow_free_shipping_promise: bool = False
    allow_bogo_promise: bool = False


@dataclass
class OrchestratorViolation:
    """A single detected violation in the Orchestrator's synthesized response."""

    violation_type: OrchestratorViolationType
    severity: str                           # "critical", "high", "medium", "low"
    description: str
    context: dict[str, Any] = field(default_factory=dict)
    auto_sanitised: bool = False            # True if we fixed it automatically

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity,
            "description": self.description,
            "context": self.context,
            "auto_sanitised": self.auto_sanitised,
        }


@dataclass
class OrchestratorGuardrailResult:
    """
    Output of the Orchestrator guardrail check.

    If should_block=True: return safe_response to user, log violation.
    If passed=False but not blocked: return sanitised_response.
    If passed=True: return original response unchanged.
    """
    passed: bool
    violations: list[OrchestratorViolation]
    sanitised_response: str
    should_block: bool
    safe_fallback_response: str | None = None

    @property
    def final_response(self) -> str:
        """
        Returns the appropriate response based on guardrail outcome.
        Orchestrator calls this — never accesses sanitised_response directly.
        """
        if self.should_block and self.safe_fallback_response:
            return self.safe_fallback_response
        return self.sanitised_response


# ---------------------------------------------------------------------------
# Pattern library — compiled once at module load
# ---------------------------------------------------------------------------

# PII detection patterns
_PII_PATTERNS = [
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # email
    re.compile(r'\b(?:\+91|0)?[6-9]\d{9}\b'),                               # IN phone
    re.compile(r'\b\d{12}\b'),                                               # Aadhaar-like
    re.compile(r'\b[4-6]\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),       # card-like
]

# Internal system identifiers that should never appear in user responses
_SYSTEM_ID_PATTERNS = [
    re.compile(r'\bproduct_id\b', re.IGNORECASE),
    re.compile(r'\bsession_id\b', re.IGNORECASE),
    re.compile(r'\btrace_id\b', re.IGNORECASE),
    re.compile(r'\bvariant_id\b', re.IGNORECASE),
    re.compile(r'\bsearch_products\b', re.IGNORECASE),
    re.compile(r'\bget_product_details\b', re.IGNORECASE),
    re.compile(r'\btool_call\b', re.IGNORECASE),
    re.compile(r'\bDiscoveryAgent\b'),
    re.compile(r'\bOrchestrator\b'),
    re.compile(r'\bLangGraph\b', re.IGNORECASE),
    re.compile(r'\bMilvus\b', re.IGNORECASE),
]

# Fabricated promotion language patterns
_ILLEGAL_PROMO_PATTERNS = [
    re.compile(r'\bbuy one get one\b', re.IGNORECASE),
    re.compile(r'\bbogo\b', re.IGNORECASE),
    re.compile(r'\bfree shipping\b', re.IGNORECASE),
    re.compile(r'\bfree delivery\b', re.IGNORECASE),
    re.compile(r'\bcashback\b', re.IGNORECASE),
    re.compile(r'\bno cost emi\b', re.IGNORECASE),
]

# Refusal language patterns — the synthesizer should NEVER use these
_REFUSAL_PATTERNS = [
    re.compile(r"\bi (can|cannot|can't|couldn't) (help|assist|find|search)\b", re.IGNORECASE),
    re.compile(r"\bi don't have (access|information|data)\b", re.IGNORECASE),
    re.compile(r"\bnot (able|possible) to\b", re.IGNORECASE),
    re.compile(r"\bsorry,? (i|we) (can|cannot)\b", re.IGNORECASE),
]

# Discount percentage extraction
_DISCOUNT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%\s*off', re.IGNORECASE)

# Price extraction (INR patterns)
_PRICE_PATTERN = re.compile(
    r'(?:₹|rs\.?|inr)\s*([0-9,]+(?:\.\d{1,2})?)', re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Orchestrator Guardrail
# ---------------------------------------------------------------------------

class OrchestratorGuardrail:
    """
    Final validation layer on the synthesized user-facing response.

    Runs AFTER the Discovery guardrail and AFTER response synthesis.
    This is the last check before the response reaches the user.

    Validates:
      1. No internal system IDs leaked into response text
      2. No PII patterns present
      3. No illegal promotion promises
      4. No discount percentages exceeding merchant limits
      5. No refusal language without an alternative offer
      6. Response length sanity check

    Sanitisation strategy:
      CRITICAL violations → block, return safe fallback
      HIGH violations     → sanitise specific pattern, return cleaned response
      MEDIUM/LOW          → log only, pass through

    Stateless — one instance shared across all requests.
    """

    # Response length bounds
    MIN_RESPONSE_CHARS: int = 20
    MAX_RESPONSE_CHARS: int = 3000

    def validate(
        self,
        response_text: str,
        product_data: list[dict[str, Any]],
        promotion_limits: PromotionLimits | None = None,
        active_promotion_ids: set[str] | None = None,
    ) -> OrchestratorGuardrailResult:
        """
        Validates the synthesizer's natural language response.

        Args:
            response_text:          Full text response from the synthesizer LLM.
            product_data:           Sanitised product dicts from DiscoveryGuardrail.
                                    Used to validate prices and discounts in text.
            promotion_limits:       Merchant-configured promotion boundaries.
            active_promotion_ids:   IDs of promotions currently active in the store.
                                    Any promotion not in this set = fabricated.

        Returns:
            OrchestratorGuardrailResult with final response text.
        """
        limits = promotion_limits or PromotionLimits()
        violations: list[OrchestratorViolation] = []
        working_response = response_text

        # ── Check 1: Internal system ID leak ──────────────────────────────
        system_leak_violations, working_response = self._check_system_id_leak(
            working_response
        )
        violations.extend(system_leak_violations)

        # ── Check 2: PII detection ─────────────────────────────────────────
        pii_violations = self._check_pii(working_response)
        violations.extend(pii_violations)

        # ── Check 3: Illegal promotion language ───────────────────────────
        if not limits.allow_free_shipping_promise or not limits.allow_bogo_promise:
            promo_violations, working_response = self._check_illegal_promotions(
                working_response, limits
            )
            violations.extend(promo_violations)

        # ── Check 4: Discount percentage validation ────────────────────────
        discount_violations, working_response = self._check_discount_claims(
            working_response, product_data, limits
        )
        violations.extend(discount_violations)

        # ── Check 5: Refusal language detection ───────────────────────────
        refusal_violations = self._check_refusal_language(working_response)
        violations.extend(refusal_violations)

        # ── Check 6: Response length sanity ───────────────────────────────
        length_violations = self._check_response_length(working_response)
        violations.extend(length_violations)

        return self._build_result(
            violations=violations,
            sanitised_response=working_response,
            original_response=response_text,
        )

    # ── Private check methods ──────────────────────────────────────────────

    def _check_system_id_leak(
        self,
        response: str,
    ) -> tuple[list[OrchestratorViolation], str]:
        """
        Detects and removes internal system identifiers from the response.
        Users should never see 'product_id', 'session_id', tool names, etc.
        """
        violations = []
        working = response

        for pattern in _SYSTEM_ID_PATTERNS:
            if pattern.search(working):
                violations.append(OrchestratorViolation(
                    violation_type=OrchestratorViolationType.INTERNAL_SYSTEM_LEAK,
                    severity="high",
                    description=(
                        f"Internal system identifier detected in response: "
                        f"pattern '{pattern.pattern}'"
                    ),
                    context={"pattern": pattern.pattern},
                    auto_sanitised=True,
                ))
                # Remove the leaked term from the response
                working = pattern.sub("[ref]", working)

        return violations, working

    def _check_pii(self, response: str) -> list[OrchestratorViolation]:
        """Detects PII patterns. PII leaks are CRITICAL — block entirely."""
        violations = []
        for pattern in _PII_PATTERNS:
            match = pattern.search(response)
            if match:
                violations.append(OrchestratorViolation(
                    violation_type=OrchestratorViolationType.PERSONAL_DATA_LEAK,
                    severity="critical",
                    description="PII pattern detected in synthesized response",
                    context={"pattern_type": pattern.pattern[:30]},
                    auto_sanitised=False,   # Can't safely sanitise PII — must block
                ))
        return violations

    def _check_illegal_promotions(
        self,
        response: str,
        limits: PromotionLimits,
    ) -> tuple[list[OrchestratorViolation], str]:
        """
        Detects promotion language not enabled by merchant settings.
        Removes the offending phrase and logs the violation.
        """
        violations = []
        working = response

        for pattern in _ILLEGAL_PROMO_PATTERNS:
            if pattern.search(working):
                matched_text = pattern.search(working).group(0)  # type: ignore
                violations.append(OrchestratorViolation(
                    violation_type=OrchestratorViolationType.ILLEGAL_PROMOTION_PROMISE,
                    severity="high",
                    description=(
                        f"Illegal promotion language detected: '{matched_text}'. "
                        f"This promotion is not enabled for this store."
                    ),
                    context={"matched_text": matched_text},
                    auto_sanitised=True,
                ))
                working = pattern.sub("", working)

        return violations, working

    def _check_discount_claims(
        self,
        response: str,
        product_data: list[dict[str, Any]],
        limits: PromotionLimits,
    ) -> tuple[list[OrchestratorViolation], str]:
        """
        Validates every discount percentage mentioned in the response.

        Two checks:
          1. Discount exceeds merchant-configured maximum
          2. Discount not present in any product in the result set

        Phase 4: adds floor enforcement — discount cannot be
        below min_discount_percent for a targeted customer.
        """
        violations = []
        working = response

        # Build valid discount set from product data
        valid_discounts: set[float] = set()
        for product in product_data:
            d = product.get("discount_percent")
            if d is not None:
                valid_discounts.add(float(d))

        matches = _DISCOUNT_PATTERN.finditer(working)
        for match in matches:
            stated_discount = float(match.group(1))

            # Check 1: Exceeds merchant maximum
            if stated_discount > limits.max_discount_percent:
                violations.append(OrchestratorViolation(
                    violation_type=OrchestratorViolationType.ILLEGAL_PROMOTION_PROMISE,
                    severity="critical",
                    description=(
                        f"Stated discount {stated_discount}% exceeds "
                        f"merchant maximum {limits.max_discount_percent}%"
                    ),
                    context={
                        "stated_discount": stated_discount,
                        "max_allowed": limits.max_discount_percent,
                    },
                    auto_sanitised=False,
                ))

            # Check 2: Not in product data (fabricated discount)
            elif valid_discounts and not any(
                abs(stated_discount - valid) < 1.0
                for valid in valid_discounts
            ):
                violations.append(OrchestratorViolation(
                    violation_type=OrchestratorViolationType.DISCOUNT_FABRICATED_IN_RESPONSE,
                    severity="high",
                    description=(
                        f"Synthesizer stated {stated_discount}% discount "
                        f"not present in product data. Valid discounts: {valid_discounts}"
                    ),
                    context={
                        "stated": stated_discount,
                        "valid_discounts": list(valid_discounts),
                    },
                    auto_sanitised=True,
                ))
                # Remove the fabricated discount claim
                working = working.replace(match.group(0), "")

        return violations, working

    def _check_refusal_language(
        self,
        response: str,
    ) -> list[OrchestratorViolation]:
        """
        Detects refusal language. We are a salesman — never refuse without offering
        an alternative. The synthesizer prompt enforces this, but we validate here.
        """
        violations = []
        for pattern in _REFUSAL_PATTERNS:
            if pattern.search(response):
                violations.append(OrchestratorViolation(
                    violation_type=OrchestratorViolationType.REFUSAL_WITHOUT_ALTERNATIVE,
                    severity="medium",
                    description=(
                        "Refusal language detected in response. "
                        "Synthesizer should always offer an alternative."
                    ),
                    context={"matched_pattern": pattern.pattern},
                    auto_sanitised=False,
                ))
                break  # One violation is enough — don't stack refusal violations

        return violations

    def _check_response_length(
        self,
        response: str,
    ) -> list[OrchestratorViolation]:
        """Length sanity check — catches empty and runaway responses."""
        violations = []
        length = len(response.strip())

        if length < self.MIN_RESPONSE_CHARS:
            violations.append(OrchestratorViolation(
                violation_type=OrchestratorViolationType.RESPONSE_TOO_SHORT,
                severity="high",
                description=(
                    f"Response length {length} chars below "
                    f"minimum {self.MIN_RESPONSE_CHARS}"
                ),
                context={"length": length, "minimum": self.MIN_RESPONSE_CHARS},
                auto_sanitised=False,
            ))

        elif length > self.MAX_RESPONSE_CHARS:
            violations.append(OrchestratorViolation(
                violation_type=OrchestratorViolationType.RESPONSE_TOO_LONG,
                severity="low",
                description=(
                    f"Response length {length} chars exceeds "
                    f"maximum {self.MAX_RESPONSE_CHARS}"
                ),
                context={"length": length, "maximum": self.MAX_RESPONSE_CHARS},
                auto_sanitised=False,
            ))

        return violations

    def _build_result(
        self,
        violations: list[OrchestratorViolation],
        sanitised_response: str,
        original_response: str,
    ) -> OrchestratorGuardrailResult:
        """Builds result and emits structured logs."""
        has_critical = any(v.severity == "critical" for v in violations)
        has_high = any(v.severity == "high" for v in violations)

        passed = not has_critical and not has_high
        should_block = has_critical

        safe_fallback = None
        if should_block:
            safe_fallback = (
                "I found some great products for you! "
                "Let me pull up the details — "
                "could you tell me a bit more about what you're looking for?"
            )

        if violations:
            log_fn = logger.error if should_block else logger.warning
            log_fn(
                LogEvent.GUARDRAIL_VIOLATION,
                "Orchestrator guardrail violations detected",
                violation_count=len(violations),
                should_block=should_block,
                response_length=len(original_response),
                violations=[v.to_dict() for v in violations],
            )
        else:
            logger.info(
                LogEvent.GUARDRAIL_PASS,
                "Orchestrator guardrail passed",
                response_length=len(sanitised_response),
            )

        return OrchestratorGuardrailResult(
            passed=passed,
            violations=violations,
            sanitised_response=sanitised_response,
            should_block=should_block,
            safe_fallback_response=safe_fallback,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_orchestrator_guardrail: OrchestratorGuardrail | None = None


def get_orchestrator_guardrail() -> OrchestratorGuardrail:
    """Returns the singleton OrchestratorGuardrail instance."""
    global _orchestrator_guardrail
    if _orchestrator_guardrail is None:
        _orchestrator_guardrail = OrchestratorGuardrail()
    return _orchestrator_guardrail