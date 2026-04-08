# conversational_commerce/agents/promotion_agent.py

"""
Phase 3: Promotion Agent

Responsibilities (when implemented):
    - Push personalised promotions to end customers
    - Recommend products based on purchase history
    - Generate merchant-level promotion recommendations
    - Enforce promotion limits (max/min discount floors)
    - Win-back campaigns for ABANDONED sessions

The PromotionLimits schema is already in guardrails/orchestrator_guard.py.
The ABANDONED session signal is already captured in memory/session_store.py.
Both are ready for this agent to consume in Phase 3.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from observability.logger import LogEvent, get_logger
from schemas.agent_io import AgentRequest, AgentResult

logger = get_logger(__name__)


class PromotionAgent(BaseAgent):
    agent_name = "promotion_agent"

    async def _execute(self, request: AgentRequest) -> AgentResult:
        logger.warning(
            LogEvent.AGENT_START,
            "PromotionAgent invoked but Phase 3 is not implemented",
            request_id=request.request_id,
        )
        return self._build_failure(
            request=request,
            error_code="NOT_IMPLEMENTED",
            error_detail="PromotionAgent is a Phase 3 feature. Not yet implemented.",
        )