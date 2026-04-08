# conversational_commerce/agents/checkout_agent.py

"""
Phase 2: Checkout Agent

Responsibilities (when implemented):
    - Initiate checkout from cart
    - Apply promotions and discount codes
    - Process payment (via payment gateway API)
    - Confirm order and trigger fulfillment
    - Handle payment failures with retry logic

The Orchestrator passes cart context (from CartContext in SessionState)
to this agent — no cross-agent communication needed.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from observability.logger import LogEvent, get_logger
from schemas.agent_io import AgentRequest, AgentResult

logger = get_logger(__name__)


class CheckoutAgent(BaseAgent):
    agent_name = "checkout_agent"

    async def _execute(self, request: AgentRequest) -> AgentResult:
        logger.warning(
            LogEvent.AGENT_START,
            "CheckoutAgent invoked but Phase 2 is not implemented",
            request_id=request.request_id,
        )
        return self._build_failure(
            request=request,
            error_code="NOT_IMPLEMENTED",
            error_detail="CheckoutAgent is a Phase 2 feature. Not yet implemented.",
        )