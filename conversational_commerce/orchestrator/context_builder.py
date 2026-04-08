# conversational_commerce/orchestrator/context_builder.py

from __future__ import annotations

from typing import Any
from uuid import uuid4

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.agent_io import AgentRequest
from schemas.intent import IntentType
from schemas.session import SessionState

logger = get_logger(__name__)
settings = get_settings()


class ContextBuilder:
    """
    Assembles AgentRequest objects from session state + user input.

    The Orchestrator never passes raw SessionState to agents.
    This class extracts the minimal relevant context slice
    each agent needs — enforcing information isolation.

    Rules:
        - Discovery Agent receives: store scope, shown products, last query
        - Cart Agent receives:      store scope, cart state, last shown products
        - Checkout Agent receives:  cart state, user identity
        - Promotion Agent receives: purchase history, session signals

    Agents only see what they need — nothing more.
    This prevents context bleed between agents.
    """

    def build_discovery_request(
        self,
        user_message: str,
        session_state: SessionState,
        intent: IntentType,
    ) -> AgentRequest:
        """
        Builds AgentRequest for the Discovery Agent.

        Context slice includes:
            - entity_id: scope all searches to the correct entity
            - business_unit_id: scope all searches to the correct business unit
            - shown_product_ids: exclusion list to avoid repetition
            - last_query: enables refinement ("show me red ones" inherits category)
            - last_filters: carry forward constraints from previous turn
            - result_limit: configurable per session/store settings
        """
        discovery_ctx = session_state.agent_context.discovery
        cfg = settings.session

        context: dict[str, Any] = {
            "business_unit_id": session_state.business_unit_id,
            "entity_id": session_state.entity_id,
            "shown_product_ids": list(discovery_ctx.shown_product_ids),
            "last_query": discovery_ctx.last_query,
            "last_filters": discovery_ctx.last_filters,
            "result_limit": 10,                     # Phase 3: tunable per store
            "total_searches_this_session": discovery_ctx.total_searches,
        }

        # Include last search results for reference resolution
        # e.g. "tell me more about the first one" →
        # agent can reference last_results[0].product_id
        if discovery_ctx.last_results:
            context["last_results_summary"] = [
                {
                    "position": i + 1,
                    "product_id": card.product_id,
                    "name": card.name,
                    "price": card.base_price,
                }
                for i, card in enumerate(discovery_ctx.last_results[:5])
            ]

        # Build task description — tells the agent exactly what to do
        task = self._build_discovery_task(
            user_message=user_message,
            intent=intent,
            discovery_ctx_has_results=bool(discovery_ctx.last_results),
        )

        # Business constraints from store/BU settings
        # Phase 3: loaded from merchant settings service
        constraints: dict[str, Any] = {
            "in_stock_only": True,
            "business_unit_id": session_state.business_unit_id,
            "entity_id": session_state.entity_id,
        }

        logger.debug(
            LogEvent.ORCHESTRATOR_AGENT_DISPATCHED,
            "Discovery request built",
            session_id=session_state.session_id,
            shown_count=len(discovery_ctx.shown_product_ids),
            has_last_results=bool(discovery_ctx.last_results),
            intent=intent.value,
        )

        return AgentRequest(
            request_id=str(uuid4()),
            session_id=session_state.session_id,
            intent=intent,
            user_message=user_message,
            task=task,
            context=context,
            constraints=constraints,
        )

    def build_cart_request(
        self,
        user_message: str,
        session_state: SessionState,
        intent: IntentType,
    ) -> AgentRequest:
        """
        Phase 2: Builds AgentRequest for the Cart Agent.

        Context slice: cart state + last shown products (for variant resolution).
        """
        cart_ctx = session_state.agent_context.cart
        discovery_ctx = session_state.agent_context.discovery

        context: dict[str, Any] = {
            "store_id": session_state.store_id,
            "cart_id": cart_ctx.cart_id,
            "cart_items": cart_ctx.items,
            "cart_total": cart_ctx.cart_total,
            "currency": cart_ctx.currency,
            # Last shown products — needed to resolve "add the first one"
            "last_results_summary": [
                {
                    "position": i + 1,
                    "product_id": card.product_id,
                    "name": card.name,
                    "price": card.base_price,
                }
                for i, card in enumerate(discovery_ctx.last_results[:5])
            ],
        }

        return AgentRequest(
            request_id=str(uuid4()),
            session_id=session_state.session_id,
            intent=intent,
            user_message=user_message,
            task=f"Handle cart operation: {user_message}",
            context=context,
            constraints={"store_id": session_state.store_id},
        )

    def build_checkout_request(
        self,
        user_message: str,
        session_state: SessionState,
        intent: IntentType,
    ) -> AgentRequest:
        """Phase 2: Builds AgentRequest for the Checkout Agent."""
        cart_ctx = session_state.agent_context.cart
        user_ctx = session_state.agent_context.user

        context: dict[str, Any] = {
            "store_id": session_state.store_id,
            "cart_id": cart_ctx.cart_id,
            "cart_total": cart_ctx.cart_total,
            "currency": cart_ctx.currency,
            "user_id": user_ctx.user_id,
            "is_authenticated": user_ctx.is_authenticated,
        }

        return AgentRequest(
            request_id=str(uuid4()),
            session_id=session_state.session_id,
            intent=intent,
            user_message=user_message,
            task="Process checkout for current cart",
            context=context,
            constraints={},
        )

    def build_promotion_request(
        self,
        user_message: str,
        session_state: SessionState,
        intent: IntentType,
    ) -> AgentRequest:
        """Phase 3: Builds AgentRequest for the Promotion Agent."""
        user_ctx = session_state.agent_context.user

        context: dict[str, Any] = {
            "store_id": session_state.store_id,
            "user_id": user_ctx.user_id,
            "preferred_categories": user_ctx.preferred_categories,
            "price_sensitivity": user_ctx.price_sensitivity,
            "total_session_searches": (
                session_state.agent_context.discovery.total_searches
            ),
        }

        return AgentRequest(
            request_id=str(uuid4()),
            session_id=session_state.session_id,
            intent=intent,
            user_message=user_message,
            task="Generate personalised promotion or recommendation",
            context=context,
            constraints={},
        )

    def _build_discovery_task(
        self,
        user_message: str,
        intent: IntentType,
        discovery_ctx_has_results: bool,
    ) -> str:
        """
        Generates a precise task description for the Discovery Agent.
        More specific than the raw user message — removes ambiguity.
        """
        if intent == IntentType.PRODUCT_DETAIL and discovery_ctx_has_results:
            return (
                f"The customer is asking for more details about a product "
                f"they've already seen. Use last_results_summary to identify "
                f"which product they're referring to, then call get_product_details. "
                f"Customer message: '{user_message}'"
            )

        if intent == IntentType.PRODUCT_SEARCH:
            return (
                f"Search for products matching the customer's request. "
                f"Extract all filters from the message, build a semantic query, "
                f"and call search_products. "
                f"Always pass entity_id and business_unit_id and exclude_product_ids from context. "
                f"Customer message: '{user_message}'"
            )

        return (
            f"Help the customer with: '{user_message}'. "
            f"Use your tools to find the most relevant products."
        )


# Module-level singleton
_context_builder: ContextBuilder | None = None


def get_context_builder() -> ContextBuilder:
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder()
    return _context_builder