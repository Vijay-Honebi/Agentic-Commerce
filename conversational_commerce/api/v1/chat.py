# conversational_commerce/api/v1/chat.py

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from api.v1.dependencies import BusinessContextDep, OrchestratorDep, RequestContextDep
from observability.logger import LogEvent, get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Conversational Commerce"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Incoming chat request body.

    Design decisions:
        - `message` is the only required field — minimal friction for clients
        - `session_id` is optional — server creates a new session if absent
        - `user_id` is optional — supports anonymous sessions
        - No store_id here — comes from X-Store-ID header (cleaner for SDKs)
    """

    message: str = Field(
        description="The customer's message.",
        min_length=1,
        max_length=2000,
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Session ID from a previous response. "
            "Omit or set null to start a new session. "
            "Include on subsequent messages to maintain conversation continuity."
        ),
        max_length=128,
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "Authenticated user ID. "
            "Omit for anonymous sessions. "
            "Include for personalisation and purchase history (Phase 3/4)."
        ),
        max_length=128,
    )

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message cannot be empty or whitespace only")
        return stripped


# class ProductImageResponse(BaseModel):
#     url: str
#     alt_text: str
#     is_primary: bool


class ProductCardResponse(BaseModel):
    """
    Product representation in the API response.

    Mirrors ProductCard schema but as a Pydantic output model.
    Includes only fields the frontend needs — no internal metadata.
    """
    product_id: str = Field(description="Canonical product identifier.", min_length=1)
    product_code: str = Field(description="Merchant-specific product code.", min_length=1)
    product_name: str = Field(description="Name of the product.", min_length=1)
    url_slug: str = Field(description="URL slug for the product page.", min_length=1)
    images: list[dict] = Field(description="List of product images.")
    relevance_score: float = Field(description="Relevance score for the product.", ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    """
    Outgoing chat response.

    Every field serves a specific frontend purpose:
        response_text   → display in chat bubble
        session_id      → store client-side, send on next request
        products        → render product cards in the chat UI
        intent          → optional: analytics, UI hints (e.g. show cart button)
        latency_ms      → optional: performance monitoring
        request_id      → correlation for support requests
        follow_up       → optional: display as quick-reply chips
    """

    response_text: str = Field(
        description="The assistant's conversational response. Display in chat UI."
    )
    session_id: str = Field(
        description=(
            "Session identifier. Store client-side and include in all "
            "subsequent requests to maintain conversation continuity."
        )
    )
    products: list[ProductCardResponse] = Field(
        default_factory=list,
        description=(
            "Structured product data for rendering product cards. "
            "Empty list if the response is not product-related."
        ),
    )
    intent: str = Field(
        description="Classified intent for this message. Useful for UI state management."
    )
    follow_up_question: str | None = Field(
        default=None,
        description=(
            "A specific follow-up question from the agent. "
            "Can be displayed as a quick-reply suggestion."
        ),
    )
    latency_ms: float = Field(
        description="End-to-end processing time in milliseconds."
    )
    request_id: str = Field(
        description="Unique request ID for support and debugging correlation."
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the Honebi Commerce Agent",
    description=(
        "The single conversational endpoint for all customer interactions. "
        "Handles product discovery, cart operations (Phase 2), "
        "checkout (Phase 2), and personalised promotions (Phase 3). "
        "Maintains full conversation context via session_id."
    ),
    responses={
        200: {"description": "Successful response with assistant message and products"},
        422: {"description": "Invalid request body or missing required headers"},
        500: {"description": "Internal server error — agent processing failed"},
    },
)
async def chat(
    body: ChatRequest,
    orchestrator: OrchestratorDep,
    request_context: RequestContextDep,
    business_context: BusinessContextDep,
) -> ChatResponse:
    """
    Main chat endpoint — single entry point for all user interactions.

    Flow:
        1. Validate request (FastAPI + Pydantic)
        2. Bind request context for logging
        3. Invoke OrchestratorAgent.process()
        4. Extract products from agent result for structured response
        5. Return ChatResponse

    Headers (required):
        X-Business-Unit-ID → Business unit scope
        X-Entity-ID        → Store / branch / channel scope

    Headers (optional):
        X-Session-ID
        X-Request-ID
        X-Trace-ID
    """
    request_id = request_context["request_id"]

    logger.info(
        LogEvent.API_REQUEST,
        "Chat request received",
        business_unit_id=business_context["business_unit_id"],
        entity_id=business_context["entity_id"],
        has_session=bool(body.session_id),
        has_user_id=bool(body.user_id),
        message_length=len(body.message),
    )

    import traceback # For debugging only — remove in production
    try:
        # ── Invoke Orchestrator ───────────────────────────────────────────
        result = await orchestrator.process(
            user_message=body.message,
            session_id=body.session_id,
            business_unit_id=business_context["business_unit_id"],
            entity_id=business_context["entity_id"],
            user_id=body.user_id,
        )

        # ── Extract structured products from agent result ─────────────────
        # The Orchestrator returns response_text as prose.
        # We also need to surface structured product cards separately
        # so the frontend can render them as product cards, not just text.
        products = _extract_product_cards(result)
        follow_up = _extract_follow_up(result)

        logger.info(
            LogEvent.API_RESPONSE,
            "Chat response ready",
            session_id=result["session_id"],
            intent=result["intent"],
            product_count=len(products),
            latency_ms=result["latency_ms"],
        )

        return ChatResponse(
            response_text=result["response_text"],
            session_id=result["session_id"],
            products=products,
            intent=result["intent"],
            follow_up_question=follow_up,
            latency_ms=result["latency_ms"],
            request_id=request_id,
        )

    except Exception as e:
        logger.error(
            LogEvent.API_ERROR,
            "Unhandled error in chat endpoint",
            error=str(e),
            error_type=type(e).__name__,
            request_id=request_id,
            business_unit_id=business_context["business_unit_id"],
            entity_id=business_context["entity_id"],
            traceback=traceback.format_exc(),

        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": (
                    "An unexpected error occurred. "
                    "Our team has been notified. "
                    f"Reference: {request_id}"
                ),
                "request_id": request_id,
            },
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    include_in_schema=False,  # Hidden from public docs
)
async def health() -> HealthResponse:
    """
    Lightweight health check for load balancers and monitoring.
    Does not check downstream dependencies — use /ready for that.
    """
    from config.settings import get_settings
    s = get_settings()
    return HealthResponse(
        status="ok",
        version=s.app_version,
        environment=s.environment.value,
    )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _extract_product_cards(
    orchestrator_result: dict[str, Any],
) -> list[ProductCardResponse]:
    """
    Extracts and converts product dicts from Orchestrator result
    into typed ProductCardResponse objects for the API response.

    The agent result data is stored in the Orchestrator's internal state.
    We access it via the session's last_results stored during _discovery_node.

    Note: For Phase 1, products are embedded in the agent_result.data.
    The Orchestrator.process() method needs to surface them.
    We read them from the response via a convention: if the response
    contains product data, it's been stored in the orchestrator result dict.
    """
    raw_products = orchestrator_result.get("products", [])

    cards: list[ProductCardResponse] = []
    for p in raw_products:
        try:
            cards.append(ProductCardResponse(
                product_id=p.get("product_id", ""),
                product_code=p.get("product_code", ""),
                product_name=p.get("product_name", ""),
                url_slug=p.get("url_slug", ""),
                images=p.get("images", []),
                relevance_score=float(p.get("relevance_score", 0)),
            ))
        except Exception as e:
            logger.warning(
                LogEvent.API_RESPONSE,
                "Failed to parse product card for API response",
                error=str(e),
                product_id=p.get("product_id", "unknown"),
            )
            continue

    return cards


def _extract_follow_up(
    orchestrator_result: dict[str, Any],
) -> str | None:
    """Extracts follow-up question from Orchestrator result if present."""
    return orchestrator_result.get("follow_up_question")