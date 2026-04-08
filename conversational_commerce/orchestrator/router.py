# conversational_commerce/orchestrator/router.py

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from orchestrator.prompts import (
    INTENT_ROUTER_SYSTEM_PROMPT,
    INTENT_ROUTER_PROMPT_VERSION,
)
from schemas.intent import IntentClassification, IntentType
from schemas.session import SessionState

logger = get_logger(__name__)
settings = get_settings()

_llm: ChatOpenAI | None = None


def _get_router_llm() -> ChatOpenAI:
    """
    Singleton LLM for intent routing.
    Separate instance from agent LLMs — different temperature,
    potentially different model in the future.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openai.chat_model,
            temperature=0.0,       # Zero temperature — deterministic classification
            max_tokens=256,        # Classification needs very few tokens
            timeout=settings.openai.request_timeout_seconds,
            api_key=settings.openai.api_key,
        )
    return _llm


class IntentRouter:
    """
    Classifies user messages into IntentType using the LLM.

    The router is the FIRST thing the Orchestrator calls on every request.
    Its output determines which specialist agent is dispatched.

    Design decisions:
        - Uses a dedicated LLM call with max_tokens=256 — fast and cheap
        - Low-confidence results (< 0.70) → CLARIFICATION intent
          Prevents dispatching to an agent with wrong context
        - Session context is included → enables follow-up understanding
          e.g. "yes" after seeing products → product_detail, not unknown
        - Every classification is logged for intent accuracy monitoring

    Stateless — one instance per application.
    """

    async def classify(
        self,
        user_message: str,
        session_state: SessionState,
    ) -> IntentClassification:
        """
        Classifies a user message into an IntentType.

        Args:
            user_message:   Raw message from the user.
            session_state:  Current session — provides context for
                            ambiguous messages like "yes" or "the first one".

        Returns:
            IntentClassification with intent, confidence, reasoning.
        """
        # Build context-aware classification prompt
        context_block = self._build_router_context(session_state)
        full_message = f"{context_block}\n\nCustomer message: {user_message}"

        logger.info(
            LogEvent.ORCHESTRATOR_INTENT_CLASSIFIED,
            "Intent classification started",
            message_preview=user_message[:100],
            session_turns=session_state.total_turns,
            prompt_version=INTENT_ROUTER_PROMPT_VERSION,
        )

        async with logger.timed(
            LogEvent.ORCHESTRATOR_INTENT_CLASSIFIED,
            "intent_classification_llm_call",
        ):
            response = await _get_router_llm().ainvoke([
                SystemMessage(content=INTENT_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=full_message),
            ])

        classification = self._parse_classification(
            response.content,
            user_message,
        )

        # Override: low-confidence → force CLARIFICATION
        # Better to ask than to dispatch with wrong intent
        if classification.is_low_confidence and classification.intent not in (
            IntentType.CLARIFICATION,
            IntentType.UNKNOWN,
        ):
            logger.info(
                LogEvent.ORCHESTRATOR_INTENT_CLASSIFIED,
                "Low confidence classification — overriding to CLARIFICATION",
                original_intent=classification.intent.value,
                confidence=classification.confidence,
                threshold=IntentClassification.LOW_CONFIDENCE_THRESHOLD,
            )
            return IntentClassification(
                intent=IntentType.CLARIFICATION,
                confidence=classification.confidence,
                reasoning=(
                    f"Low confidence ({classification.confidence:.2f}) on "
                    f"'{classification.intent.value}' — asking for clarification"
                ),
                fallback_intent=classification.intent,
            )

        logger.info(
            LogEvent.ORCHESTRATOR_INTENT_CLASSIFIED,
            "Intent classified",
            intent=classification.intent.value,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            is_low_confidence=classification.is_low_confidence,
        )

        return classification

    def _build_router_context(self, session_state: SessionState) -> str:
        """
        Builds session context for the router LLM.

        Provides just enough context to resolve ambiguous messages
        without overwhelming the classification prompt.
        """
        parts = ["[SESSION CONTEXT FOR CLASSIFICATION]"]
        parts.append(f"turn_count: {session_state.total_turns}")

        # Last assistant message — helps classify follow-ups
        recent = session_state.get_recent_turns(3)
        if recent:
            last_assistant = next(
                (t for t in reversed(recent) if t.role.value == "assistant"),
                None,
            )
            if last_assistant:
                parts.append(
                    f"last_assistant_message: {last_assistant.content[:200]}"
                )

        # Products shown — helps classify "add to cart" vs "more details"
        shown_ids = session_state.agent_context.discovery.shown_product_ids
        if shown_ids:
            parts.append(f"products_shown_count: {len(shown_ids)}")
            parts.append("products_have_been_shown: true")

        # Cart state — helps classify checkout intent
        cart = session_state.agent_context.cart
        if cart.item_count > 0:
            parts.append(f"cart_item_count: {cart.item_count}")
            parts.append(f"cart_total: {cart.cart_total}")

        return "\n".join(parts)

    def _parse_classification(
        self,
        content: str | list,
        original_message: str,
    ) -> IntentClassification:
        """
        Parses the router LLM's JSON response into IntentClassification.
        Falls back to UNKNOWN on any parse error.
        """
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )

        content = str(content).strip()

        # Strip markdown fences
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )

        try:
            data = json.loads(content)

            intent_str = data.get("intent", "unknown")
            try:
                intent = IntentType(intent_str)
            except ValueError:
                logger.warning(
                    LogEvent.ORCHESTRATOR_INTENT_CLASSIFIED,
                    "Unknown intent value from LLM — defaulting to UNKNOWN",
                    intent_str=intent_str,
                )
                intent = IntentType.UNKNOWN

            fallback_str = data.get("fallback_intent")
            fallback = None
            if fallback_str:
                try:
                    fallback = IntentType(fallback_str)
                except ValueError:
                    fallback = None

            return IntentClassification(
                intent=intent,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                fallback_intent=fallback,
            )

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(
                LogEvent.ORCHESTRATOR_INTENT_CLASSIFIED,
                "Failed to parse intent classification response",
                error=str(e),
                content_preview=content[:200],
            )
            return IntentClassification(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                reasoning=f"Parse error: {str(e)}",
            )


# Module-level singleton
_router: IntentRouter | None = None


def get_intent_router() -> IntentRouter:
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router