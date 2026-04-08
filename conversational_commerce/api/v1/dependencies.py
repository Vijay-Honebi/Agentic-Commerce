# conversational_commerce/api/v1/dependencies.py

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from config.settings import get_settings
from observability.logger import LogEvent, get_logger, set_request_context
from orchestrator.orchestrator_agent import OrchestratorAgent, get_orchestrator

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Orchestrator dependency
# ---------------------------------------------------------------------------

def get_orchestrator_dep() -> OrchestratorAgent:
    """
    FastAPI dependency: returns the singleton OrchestratorAgent.

    The Orchestrator is compiled once at startup (main.py lifespan).
    This dependency simply retrieves that singleton — zero overhead.
    No state is created per request.
    """
    return get_orchestrator()


# ---------------------------------------------------------------------------
# Request context dependency
# ---------------------------------------------------------------------------

async def bind_request_context(
    request: Request,
    x_request_id: Annotated[str | None, Header()] = None,
    x_session_id: Annotated[str | None, Header()] = None,
    x_trace_id: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    """
    FastAPI dependency: binds request/session/trace IDs to async context.

    IDs are read from request headers if provided by the client.
    If not provided, UUIDs are generated server-side.

    These IDs propagate through every log statement in the entire
    call stack via ContextVar — no manual passing required.

    Header contract:
        X-Request-ID  → Client-provided request ID (idempotency key)
        X-Session-ID  → Session continuity across messages
        X-Trace-ID    → Distributed tracing correlation

    Returns the IDs dict so the endpoint can include them in the response.
    """
    request_id = x_request_id or str(uuid.uuid4())
    session_id = x_session_id or ""
    trace_id = x_trace_id or str(uuid.uuid4())

    set_request_context(
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
    )

    logger.info(
        LogEvent.API_REQUEST,
        "Incoming request",
        method=request.method,
        path=str(request.url.path),
        request_id=request_id,
        session_id=session_id or "new",
        user_agent=request.headers.get("user-agent", ""),
    )

    return {
        "request_id": request_id,
        "session_id": session_id,
        "trace_id": trace_id,
    }


# ---------------------------------------------------------------------------
# Store validation dependency
# ---------------------------------------------------------------------------

async def resolve_business_context(
    request: Request,
    x_business_unit_id: Annotated[str, Header()],
    x_entity_id: Annotated[str, Header()]
) -> dict[str, str]:
    """
    Resolves business_unit_id and entity_id for this request.

    Single-entity enterprise deployment:
        Both set in .env — clients never send headers.

    Multi-entity deployment:
        Client sends X-Business-Unit-ID and X-Entity-ID headers.
        Falls back to .env defaults if headers absent.
    """

    business_unit_id = x_business_unit_id.strip()
    entity_id = x_entity_id.strip()

    if not business_unit_id:
        raise HTTPException(422, "X-Business-Unit-ID cannot be empty")

    if not entity_id:
        raise HTTPException(422, "X-Entity-ID cannot be empty")

    return {
        "business_unit_id": business_unit_id,
        "entity_id": entity_id,
    }

# New type alias
BusinessContextDep = Annotated[
    dict[str, str],
    Depends(resolve_business_context)
]


# ---------------------------------------------------------------------------
# Type aliases for cleaner endpoint signatures
# ---------------------------------------------------------------------------

OrchestratorDep = Annotated[OrchestratorAgent, Depends(get_orchestrator_dep)]
RequestContextDep = Annotated[dict[str, str], Depends(bind_request_context)]
