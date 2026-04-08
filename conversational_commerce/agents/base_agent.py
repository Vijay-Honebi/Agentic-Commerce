# conversational_commerce/agents/base_agent.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from schemas.agent_io import AgentRequest, AgentResult
from observability.logger import LogEvent, get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for ALL specialist agents in the system.

    Contract:
        Input  → AgentRequest   (constructed exclusively by the Orchestrator)
        Output → AgentResult    (consumed exclusively by the Orchestrator)

    Rules every agent must follow:
        1. Never read or write SessionState directly
           — context arrives via AgentRequest.context
           — state is persisted by the Orchestrator after the run

        2. Never speak to the user
           — return structured data only
           — the Orchestrator's synthesizer converts data to prose

        3. Never call another agent
           — agent-to-agent calls create context drift
           — all routing goes through the Orchestrator

        4. Always return AgentResult — never raise to the Orchestrator
           — use AgentResult.failure() for all error cases
           — the Orchestrator handles both success and failure explicitly

        5. Always log entry and exit with structured events
           — base class handles this automatically in run()

    New agents in Phase 2/3/4:
        - Inherit BaseAgent
        - Implement _execute()
        - Register tools in bootstrap.py with correct agent_scope
        - Add a routing case in orchestrator/router.py
        - Nothing else changes
    """

    # Subclasses declare their name — used for logging and tool scoping
    agent_name: str = "base_agent"

    async def run(self, request: AgentRequest) -> AgentResult:
        """
        Public entry point called exclusively by the Orchestrator.

        Wraps _execute() with:
          - Structured entry/exit logging
          - Execution timing
          - Top-level exception guard (never raises to Orchestrator)

        Subclasses implement _execute(), not run().
        """
        logger.info(
            LogEvent.AGENT_START,
            f"{self.agent_name} started",
            agent_name=self.agent_name,
            request_id=request.request_id,
            session_id=request.session_id,
            intent=request.intent.value,
            task_preview=request.task[:100],
        )

        try:
            async with logger.timed(
                LogEvent.AGENT_END,
                f"{self.agent_name}_execution",
                agent_name=self.agent_name,
                request_id=request.request_id,
            ):
                result = await self._execute(request)

        except Exception as e:
            # Top-level guard — unhandled exceptions become AgentResult.failure
            # The Orchestrator always receives a result, never an exception
            logger.error(
                LogEvent.AGENT_END,
                f"{self.agent_name} unhandled exception",
                agent_name=self.agent_name,
                request_id=request.request_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return AgentResult.failure(
                request_id=request.request_id,
                agent_name=self.agent_name,
                error_code="UNHANDLED_EXCEPTION",
                error_detail=str(e),
                metadata={"exception_type": type(e).__name__},
            )

        logger.info(
            LogEvent.AGENT_END,
            f"{self.agent_name} completed",
            agent_name=self.agent_name,
            request_id=request.request_id,
            success=result.success,
            confidence=result.confidence,
        )

        return result

    @abstractmethod
    async def _execute(self, request: AgentRequest) -> AgentResult:
        """
        Core agent logic. Implemented by each specialist agent.

        Must:
          - Return AgentResult (success or failure)
          - Never raise exceptions (handle internally)
          - Never access session state directly
          - Only use tools scoped to this agent
        """
        ...

    def _build_success(
        self,
        request: AgentRequest,
        data: dict[str, Any],
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Convenience builder for successful results."""
        return AgentResult(
            request_id=request.request_id,
            agent_name=self.agent_name,
            success=True,
            data=data,
            confidence=confidence,
            metadata=metadata or {},
        )

    def _build_failure(
        self,
        request: AgentRequest,
        error_code: str,
        error_detail: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Convenience builder for failure results."""
        return AgentResult.failure(
            request_id=request.request_id,
            agent_name=self.agent_name,
            error_code=error_code,
            error_detail=error_detail,
            metadata=metadata or {},
        )