# conversational_commerce/agents/cart_agent.py

"""
Phase 2: Cart Agent

Responsibilities (when implemented):
    - Add items to cart (by variant_id from discovery results)
    - Remove items from cart
    - Update item quantities
    - View current cart contents
    - Apply promotion codes

Tools (Phase 2):
    - add_to_cart(variant_id, quantity, store_id)
    - remove_from_cart(cart_item_id)
    - update_cart_item(cart_item_id, quantity)
    - get_cart(cart_id)
    - apply_promotion(cart_id, promotion_code)

The variant_id needed by add_to_cart is already present in
ProductCard and ProductDetail from Phase 1 — no schema changes needed.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from observability.logger import LogEvent, get_logger
from schemas.agent_io import AgentRequest, AgentResult

logger = get_logger(__name__)


class CartAgent(BaseAgent):
    agent_name = "cart_agent"

    async def _execute(self, request: AgentRequest) -> AgentResult:
        logger.warning(
            LogEvent.AGENT_START,
            "CartAgent invoked but Phase 2 is not implemented",
            request_id=request.request_id,
        )
        return self._build_failure(
            request=request,
            error_code="NOT_IMPLEMENTED",
            error_detail="CartAgent is a Phase 2 feature. Not yet implemented.",
        )