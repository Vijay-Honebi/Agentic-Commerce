# conversational_commerce/orchestrator/response_synthesizer.py

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from guardrails.orchestrator_guard import PromotionLimits, get_orchestrator_guardrail
from observability.logger import LogEvent, get_logger
from orchestrator.prompts import (
    CLARIFICATION_SYSTEM_PROMPT,
    RESPONSE_SYNTHESIZER_SYSTEM_PROMPT,
    RESPONSE_SYNTHESIZER_PROMPT_VERSION,
)
from schemas.agent_io import AgentResult
from schemas.intent import IntentType
from schemas.session import SessionState

logger = get_logger(__name__)
settings = get_settings()

_llm: ChatOpenAI | None = None


def _get_synthesizer_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openai.chat_model,
            temperature=0.3,    # Slight creativity for natural prose
            max_tokens=settings.openai.max_tokens,
            timeout=settings.openai.request_timeout_seconds,
            api_key=settings.openai.api_key,
        )
    return _llm


class ResponseSynthesizer:
    """
    Converts structured agent results into user-facing conversational prose.

    This is the Orchestrator's voice — the only component that speaks
    to the user. Every response passes through here and then through
    the OrchestratorGuardrail before reaching the API layer.

    Responsibilities:
        - Convert product data dicts → natural language product descriptions
        - Apply salesman tone and purchase-intent framing
        - Handle clarification responses (no agent involved)
        - Handle agent failures gracefully (no raw error messages to users)
        - Pass every response through OrchestratorGuardrail

    Stateless — one instance per application.
    """

    async def synthesize(
        self,
        agent_result: AgentResult,
        intent: IntentType,
        session_state: SessionState,
        promotion_limits: PromotionLimits | None = None,
    ) -> str:
        """
        Synthesizes a user-facing response from an AgentResult.

        Args:
            agent_result:      Output from a specialist agent.
            intent:            Classified intent (affects synthesis style).
            session_state:     Current session (provides conversation history context).
            promotion_limits:  Merchant promotion boundaries for guardrail.

        Returns:
            Clean, validated, user-ready response string.
        """
        # Handle agent failures gracefully
        if not agent_result.success:
            return await self._synthesize_failure(
                agent_result=agent_result,
                intent=intent,
            )

        # Build synthesis prompt from agent data
        synthesis_prompt = self._build_synthesis_prompt(
            agent_result=agent_result,
            intent=intent,
            session_state=session_state,
        )

        logger.info(
            LogEvent.ORCHESTRATOR_RESPONSE_SYNTHESIZED,
            "Response synthesis started",
            intent=intent.value,
            product_count=len(
                agent_result.data.get("products", [])
            ),
            prompt_version=RESPONSE_SYNTHESIZER_PROMPT_VERSION,
        )

        async with logger.timed(
            LogEvent.ORCHESTRATOR_RESPONSE_SYNTHESIZED,
            "response_synthesis_llm_call",
        ):
            response = await _get_synthesizer_llm().ainvoke([
                SystemMessage(content=RESPONSE_SYNTHESIZER_SYSTEM_PROMPT),
                HumanMessage(content=synthesis_prompt),
            ])

        raw_response = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Run Orchestrator guardrail on the synthesized text
        validated_response = self._run_guardrail(
            response_text=raw_response,
            agent_result=agent_result,
            promotion_limits=promotion_limits,
        )

        logger.info(
            LogEvent.ORCHESTRATOR_RESPONSE_SYNTHESIZED,
            "Response synthesis complete",
            response_length=len(validated_response),
            intent=intent.value,
        )

        return validated_response

    async def synthesize_clarification(
        self,
        user_message: str,
        session_state: SessionState,
    ) -> str:
        """
        Synthesizes a clarifying question.
        Called when intent classification returns CLARIFICATION.
        No agent is dispatched — Orchestrator handles this directly.
        """
        context = ""
        if session_state.total_turns > 0:
            recent = session_state.get_recent_turns(3)
            context = "\n".join(
                f"{t.role.value}: {t.content[:100]}"
                for t in recent
            )

        prompt = (
            f"Recent conversation:\n{context}\n\n"
            if context
            else ""
        ) + f"Customer message: {user_message}"

        logger.info(
            LogEvent.ORCHESTRATOR_RESPONSE_SYNTHESIZED,
            "Clarification synthesis started",
            message_preview=user_message[:100],
        )

        response = await _get_synthesizer_llm().ainvoke([
            SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        return (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

    def _build_synthesis_prompt(
        self,
        agent_result: AgentResult,
        intent: IntentType,
        session_state: SessionState,
    ) -> str:
        """
        Builds the data prompt sent to the synthesizer LLM.

        Provides:
            1. Recent conversation history (for continuity)
            2. Structured product data (ground truth — synthesizer cannot exceed this)
            3. Response guidance (tone, format hints based on result shape)
        """
        parts: list[str] = []

        # ── 1. Conversation history ───────────────────────────────────────
        cfg = settings.session
        recent_turns = session_state.get_recent_turns(
            cfg.max_turns_in_memory
        )
        if recent_turns:
            history_lines = [
                f"{turn.role.value.upper()}: {turn.content}"
                for turn in recent_turns[-4:]  # Last 4 turns for synthesis context
            ]
            parts.append("[CONVERSATION HISTORY]")
            parts.extend(history_lines)
            parts.append("")

        # ── 2. Agent result data ──────────────────────────────────────────
        data = agent_result.data
        products = data.get("products", [])

        parts.append("[PRODUCTS DATA]")
        if products:
            for i, product in enumerate(products, 1):
                parts.append(self._format_product_for_synthesis(i, product))
        else:
            parts.append("no_products_found: true")

        parts.append("")

        # ── 3. Response guidance ──────────────────────────────────────────
        parts.append("[RESPONSE GUIDANCE]")

        if not products:
            parts.append("situation: No products found for this query")
            parts.append(
                "instruction: Acknowledge the empty result, "
                "suggest a related search direction, "
                "ask ONE specific question to help narrow down"
            )

        elif intent == IntentType.PRODUCT_DETAIL:
            parts.append("situation: Customer asked for product details")
            parts.append(
                "instruction: Lead with the most compelling aspect. "
                "Cover available variants (sizes, colours), exact price. "
                "End with purchase-intent question."
            )

        elif intent == IntentType.PRODUCT_SEARCH:
            count = len(products)
            if count == 1:
                parts.append("situation: Found exactly one matching product")
                parts.append(
                    "instruction: Present it confidently with key features. "
                    "Create urgency if in_stock is limited. "
                    "End with a specific question about size/colour/fit."
                )
            elif count <= 3:
                parts.append(f"situation: Found {count} matching products")
                parts.append(
                    "instruction: Describe each briefly with its key differentiator. "
                    "Help customer compare. "
                    "End with ONE question that helps them choose."
                )
            else:
                parts.append(f"situation: Found {count} matching products")
                parts.append(
                    "instruction: Highlight top 2-3 most relevant products. "
                    f"Mention {count - 3} more are available. "
                    "Ask ONE narrowing question."
                )

        # Relaxation notice — be transparent if results were broadened
        if agent_result.metadata.get("relaxation_applied"):
            parts.append(
                "note: Results are slightly broader than the exact request "
                "— mention this naturally if relevant"
            )

        return "\n".join(parts)

    def _format_product_for_synthesis(
        self,
        position: int,
        product: dict[str, Any],
    ) -> str:
        """
        Formats a single product dict for the synthesis prompt.
        Structured clearly so the LLM extracts accurate data.
        """
        lines = [f"PRODUCT {position}:"]
        lines.append(f"  name: {product.get('name', 'Unknown')}")

        if brand := product.get("brand"):
            lines.append(f"  brand: {brand}")

        lines.append(f"  price: ₹{product.get('base_price', 0):,.0f}")

        if product.get("is_on_sale") and product.get("discount_percent"):
            lines.append(
                f"  discount: {product['discount_percent']}% off "
                f"(was ₹{product.get('compare_at_price', 0):,.0f})"
            )

        lines.append(f"  in_stock: {product.get('in_stock', False)}")

        if rating := product.get("rating"):
            lines.append(
                f"  rating: {rating}/5 ({product.get('review_count', 0)} reviews)"
            )

        if attrs := product.get("key_attributes"):
            attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
            lines.append(f"  key_attributes: {attr_str}")

        if desc := product.get("short_description"):
            lines.append(f"  description: {desc[:150]}")

        return "\n".join(lines)

    async def _synthesize_failure(
        self,
        agent_result: AgentResult,
        intent: IntentType,
    ) -> str:
        """
        Produces a graceful user-facing response for agent failures.
        Never exposes error codes or technical details to the user.
        """
        error_code = agent_result.error_code or "UNKNOWN"

        logger.warning(
            LogEvent.ORCHESTRATOR_RESPONSE_SYNTHESIZED,
            "Synthesizing failure response",
            error_code=error_code,
            agent_name=agent_result.agent_name,
            intent=intent.value,
        )

        # Map error codes to user-friendly responses
        failure_responses: dict[str, str] = {
            "NOT_IMPLEMENTED": (
                "I'm still learning that feature! "
                "For now, let me help you discover products. "
                "What are you looking for today?"
            ),
            "RETRIEVAL_EMPTY": (
                "I couldn't find products matching those exact criteria. "
                "Could you tell me more about what you're looking for? "
                "For example, a price range or specific style?"
            ),
            "GUARDRAIL_BLOCKED": (
                "Let me search for the best options for you. "
                "Could you describe what you're looking for in a bit more detail?"
            ),
        }

        return failure_responses.get(
            error_code,
            (
                "I'm having a moment — let me try again. "
                "What product are you looking for today?"
            ),
        )

    def _run_guardrail(
        self,
        response_text: str,
        agent_result: AgentResult,
        promotion_limits: PromotionLimits | None,
    ) -> str:
        """
        Runs OrchestratorGuardrail on the synthesized response.
        Returns final_response (safe fallback if blocked, sanitised if violations).
        """
        guardrail = get_orchestrator_guardrail()

        guardrail_result = guardrail.validate(
            response_text=response_text,
            product_data=agent_result.data.get("products", []),
            promotion_limits=promotion_limits or PromotionLimits(),
        )

        return guardrail_result.final_response


# Module-level singleton
_synthesizer: ResponseSynthesizer | None = None


def get_response_synthesizer() -> ResponseSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = ResponseSynthesizer()
    return _synthesizer