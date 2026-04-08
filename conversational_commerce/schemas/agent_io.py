# conversational_commerce/schemas/agent_io.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from schemas.intent import IntentType


class AgentRequest(BaseModel):
    """
    The ONLY input contract between Orchestrator and any specialist agent.

    Design rules:
      - Orchestrator constructs this. Agents never construct it themselves.
      - Agents receive ONLY what they need — never the full session object.
        This enforces separation: agents cannot leak context to each other.
      - `context` contains a curated slice of session state prepared by
        context_builder.py — not raw session dump.
      - `constraints` contains business rules the agent MUST enforce.
        Guardrails validate agent output against these constraints.

    The agent contract is intentionally narrow:
      Input  → AgentRequest
      Output → AgentResult
    Nothing else crosses the Orchestrator ↔ Agent boundary.
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this specific agent invocation. Used for tracing.",
    )
    session_id: str = Field(
        description="Parent session ID. Agents use this for logging only — "
                    "they do NOT read/write session state directly.",
    )
    intent: IntentType = Field(
        description="Classified intent that caused this agent to be dispatched.",
    )
    user_message: str = Field(
        description="The raw user message, exactly as received. "
                    "Agents must NOT modify this.",
    )
    task: str = Field(
        description="Precise task description written by the Orchestrator. "
                    "More specific than user_message — tells agent exactly what to do.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Curated context slice from session state. "
                    "Prepared by context_builder.py. "
                    "Contents vary by agent type.",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Business rules and hard limits this agent must enforce. "
                    "Examples: max_price, allowed_categories, active_promotions.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    model_config = {"frozen": True}  # Immutable — agents cannot mutate their input


class AgentResult(BaseModel):
    """
    The ONLY output contract from any specialist agent back to the Orchestrator.

    Design rules:
      - Agents return structured DATA, never prose.
        The Orchestrator's synthesizer converts data to user-facing text.
      - `success=False` is a valid, handled state — not an exception.
        Agents must always return AgentResult, never raise to the Orchestrator.
      - `data` schema varies by agent type but must always be JSON-serialisable.
      - `confidence` lets the Orchestrator decide whether to accept results,
        trigger relaxation, or ask the user for clarification.
      - `metadata` captures observability data: token counts, latencies, trace IDs.
    """

    request_id: str = Field(
        description="Echoed from AgentRequest.request_id for correlation.",
    )
    agent_name: str = Field(
        description="Name of the agent that produced this result. "
                    "Used for logging and orchestrator routing logic.",
    )
    success: bool = Field(
        description="True if the agent completed its task. "
                    "False if it hit an unrecoverable error or empty result. "
                    "Orchestrator handles both states explicitly.",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured output from the agent. "
                    "Orchestrator's synthesizer reads this to build user response.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in its own output quality. "
                    "< 0.5 triggers orchestrator fallback logic.",
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error code when success=False. "
                    "Orchestrator maps these to user-friendly messages.",
    )
    error_detail: str | None = Field(
        default=None,
        description="Internal error detail for logging. Never shown to user.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Observability data: llm_tokens_used, retrieval_latency_ms, "
                    "relaxation_rounds, candidate_count, trace_id, etc.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @classmethod
    def failure(
        cls,
        request_id: str,
        agent_name: str,
        error_code: str,
        error_detail: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Convenience constructor for failure cases.
        Agents use this for clean, consistent error returns.

        Usage:
            return AgentResult.failure(
                request_id=request.request_id,
                agent_name="discovery_agent",
                error_code="RETRIEVAL_EMPTY",
                error_detail=f"Milvus returned 0 candidates for query: {query}",
            )
        """
        return cls(
            request_id=request_id,
            agent_name=agent_name,
            success=False,
            error_code=error_code,
            error_detail=error_detail,
            confidence=0.0,
            metadata=metadata or {},
        )