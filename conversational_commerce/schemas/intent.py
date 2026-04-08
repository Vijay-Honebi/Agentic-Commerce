# conversational_commerce/schemas/intent.py

from __future__ import annotations

from enum import Enum


class IntentType(str, Enum):
    """
    Canonical intent taxonomy for the entire system.

    The Orchestrator classifies every user message into exactly one IntentType.
    This classification drives all routing decisions — which agent gets invoked,
    which tools are available, and how the response is synthesized.

    Design rules:
      - Each intent maps to exactly ONE specialist agent
      - Intents are mutually exclusive at classification time
      - New phases add new intents here — never modify existing ones
      - UNKNOWN is a valid production state, not an error

    Phase mapping:
      Phase 1  → PRODUCT_SEARCH, PRODUCT_DETAIL, CLARIFICATION, UNKNOWN
      Phase 2  → ADD_TO_CART, REMOVE_FROM_CART, VIEW_CART,
                 UPDATE_CART, CHECKOUT, ORDER_STATUS
      Phase 3  → PROMOTION_INQUIRY, RECOMMENDATION_REQUEST
      Phase 4  → PERSONALIZED_DISCOVERY (fully autonomous)
    """

    # ── Phase 1: Discovery ────────────────────────────────────────────────
    # User is searching for products by attribute, category, or natural language
    PRODUCT_SEARCH = "product_search"

    # User wants full details on a specific product they already know about
    PRODUCT_DETAIL = "product_detail"

    # User's query is ambiguous — orchestrator needs to ask a clarifying question
    # before dispatching to any agent. No agent is called for this intent.
    CLARIFICATION = "clarification"

    # ── Phase 2: Cart & Checkout ──────────────────────────────────────────
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    VIEW_CART = "view_cart"
    UPDATE_CART = "update_cart"
    CHECKOUT = "checkout"
    ORDER_STATUS = "order_status"

    # ── Phase 3: Promotions & Recommendations ─────────────────────────────
    PROMOTION_INQUIRY = "promotion_inquiry"
    RECOMMENDATION_REQUEST = "recommendation_request"

    # ── Phase 4: Personalized Commerce ───────────────────────────────────
    PERSONALIZED_DISCOVERY = "personalized_discovery"

    # ── Fallback ──────────────────────────────────────────────────────────
    # Orchestrator could not classify with sufficient confidence.
    # Triggers a graceful "I can help you find products..." fallback response.
    UNKNOWN = "unknown"


class IntentClassification(object):
    """
    Metadata about an intent classification decision.
    Produced by the Orchestrator's router node.
    Logged for every request — enables intent accuracy monitoring.
    """

    def __init__(
        self,
        intent: IntentType,
        confidence: float,
        reasoning: str,
        fallback_intent: IntentType | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {confidence}"
            )

        self.intent = intent
        self.confidence = confidence
        self.reasoning = reasoning  # LLM's explanation — stored in session for debug
        self.fallback_intent = fallback_intent  # second-best if confidence is low

    # Threshold below which orchestrator asks for clarification
    # rather than proceeding with a low-confidence routing decision
    LOW_CONFIDENCE_THRESHOLD: float = 0.70

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < self.LOW_CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "fallback_intent": (
                self.fallback_intent.value if self.fallback_intent else None
            ),
            "is_low_confidence": self.is_low_confidence,
        }