# conversational_commerce/orchestrator/orchestrator_agent.py

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agents.discovery_agent import get_discovery_agent
from agents.cart_agent import CartAgent
from agents.checkout_agent import CheckoutAgent
from agents.promotion_agent import PromotionAgent
from config.settings import get_settings
from memory.session_store import create_session, load_session, persist_session
from observability.logger import LogEvent, get_logger, set_request_context
from orchestrator.context_builder import get_context_builder
from orchestrator.response_synthesizer import get_response_synthesizer
from orchestrator.router import get_intent_router
from schemas.agent_io import AgentResult
from schemas.intent import IntentClassification, IntentType
from schemas.session import (
    ConversationTurn,
    MessageRole,
    SessionState,
    SessionStatus,
)

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Orchestrator State
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    """
    Complete state for one Orchestrator graph execution.
    One execution = one user message + one system response.

    Unlike agent state (which has a message list),
    Orchestrator state is flat — each field is a discrete value.
    The Orchestrator's "memory" lives in SessionState (PostgreSQL),
    not in the LangGraph message list.
    """

    # Input
    user_message: str
    session_id: str
    business_unit_id: str
    entity_id: str
    user_id: str | None
    request_id: str

    # Session (loaded at start, persisted at end)
    session_state: SessionState | None

    # Routing
    intent_classification: IntentClassification | None
    intent: IntentType | None

    # Agent execution
    agent_result: AgentResult | None

    # Output
    response_text: str
    response_ready: bool

    # Metadata
    start_time: float
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def _load_session_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """
    Node 1: Loads or creates session from PostgreSQL.

    Logic:
        - If session_id exists and is active → load it
        - If session_id doesn't exist or expired → create fresh session
        - Sets request context for logging (request_id, session_id, trace_id)
    """
    session_id = state["session_id"]
    entity_id = state["entity_id"]
    business_unit_id = state["business_unit_id"]
    user_id = state.get("user_id")

    # Bind IDs to async context for all downstream log statements
    set_request_context(
        request_id=state["request_id"],
        session_id=session_id,
    )

    logger.info(
        LogEvent.ORCHESTRATOR_START,
        "Orchestrator started",
        request_id=state["request_id"],
        session_id=session_id,
        message_preview=state["user_message"][:100],
    )

    # Attempt to load existing session
    session_state = await load_session(session_id)

    if session_state is None:
        # Create new session — first message or expired session
        session_state = await create_session(
            session_id=session_id,
            entity_id=entity_id,
            business_unit_id=business_unit_id,
            user_id=user_id,
        )
        logger.info(
            LogEvent.SESSION_CREATED,
            "New session created by Orchestrator",
            session_id=session_id,
            entity_id=entity_id,
            business_unit_id=business_unit_id,
        )

    return {"session_state": session_state}


async def _intent_router_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """
    Node 2: Classifies user intent.
    Dispatches to IntentRouter → IntentClassification.
    """
    session_state = state["session_state"]
    user_message = state["user_message"]

    router = get_intent_router()
    classification = await router.classify(
        user_message=user_message,
        session_state=session_state,
    )

    logger.info(
        LogEvent.ORCHESTRATOR_INTENT_CLASSIFIED,
        "Intent routing complete",
        intent=classification.intent.value,
        confidence=classification.confidence,
        reasoning=classification.reasoning,
    )

    return {
        "intent_classification": classification,
        "intent": classification.intent,
    }


async def _discovery_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """
    Node 3a: Dispatches to Discovery Agent.
    Handles PRODUCT_SEARCH and PRODUCT_DETAIL intents.
    """
    session_state = state["session_state"]
    intent = state["intent"]
    user_message = state["user_message"]

    logger.info(
        LogEvent.ORCHESTRATOR_AGENT_DISPATCHED,
        "Dispatching to Discovery Agent",
        intent=intent.value,
        session_id=session_state.session_id,
    )

    # Build curated context slice for this agent
    context_builder = get_context_builder()
    agent_request = context_builder.build_discovery_request(
        user_message=user_message,
        session_state=session_state,
        intent=intent,
    )

    # Execute agent
    discovery_agent = get_discovery_agent()
    agent_result = await discovery_agent.run(agent_request)

    # Update session discovery context on success
    if agent_result.success:
        data = agent_result.data
        products = data.get("products", [])

        # Track shown product IDs (for next-turn exclusion)
        shown_ids = session_state.agent_context.discovery.shown_product_ids
        for product in products:
            pid = product.get("product_id")
            if pid and pid not in shown_ids:
                shown_ids.append(pid)

        # Update discovery context for session persistence
        session_state.agent_context.discovery.last_query = (
            user_message
        )
        session_state.agent_context.discovery.total_searches += 1

        # Store last results for reference resolution in follow-up turns
        # "tell me about the third one" → references last_results[2]
        from schemas.product import ProductCard, ProductImage
        last_results = []
        for p in products[:10]:     # Keep top 10 for reference
            card = ProductCard(
                product_id=p.get("product_id", ""),
                product_code=p.get("product_code", ""),
                product_name=p.get("product_name", ""),
                url_slug=p.get("url_slug", ""),
                relevance_score=p.get("relevance_score", 0.0),
                images=p.get("images", []),
            )
            last_results.append(card)

        session_state.agent_context.discovery.last_results = last_results

        if agent_result.metadata.get("parsed_query_filters"):
            session_state.agent_context.discovery.last_filters = (
                agent_result.metadata["parsed_query_filters"]
            )

    logger.info(
        LogEvent.ORCHESTRATOR_AGENT_DISPATCHED,
        "Discovery Agent completed",
        success=agent_result.success,
        confidence=agent_result.confidence,
        product_count=len(
            agent_result.data.get("products", [])
            if agent_result.success else []
        ),
    )


    # logger.info(
    #     LogEvent.DEBUG,
    #     "DISCOVERY OUTPUT",
    #     data=agent_result
    # )

    # return {"agent_result": agent_result}
    return {
        "agent_result": agent_result,
        "response_text": agent_result.data.get("response_text", ""),
    }


async def _cart_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """Node 3b: Phase 2 — Cart Agent dispatch."""
    session_state = state["session_state"]
    intent = state["intent"]

    context_builder = get_context_builder()
    agent_request = context_builder.build_cart_request(
        user_message=state["user_message"],
        session_state=session_state,
        intent=intent,
    )

    cart_agent = CartAgent()
    agent_result = await cart_agent.run(agent_request)

    return {"agent_result": agent_result}


async def _checkout_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """Node 3c: Phase 2 — Checkout Agent dispatch."""
    session_state = state["session_state"]
    intent = state["intent"]

    context_builder = get_context_builder()
    agent_request = context_builder.build_checkout_request(
        user_message=state["user_message"],
        session_state=session_state,
        intent=intent,
    )

    checkout_agent = CheckoutAgent()
    agent_result = await checkout_agent.run(agent_request)

    # Mark session completed on successful checkout
    if agent_result.success:
        session_state.status = SessionStatus.COMPLETED

    return {"agent_result": agent_result}


async def _promotion_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """Node 3d: Phase 3 — Promotion Agent dispatch."""
    session_state = state["session_state"]
    intent = state["intent"]

    context_builder = get_context_builder()
    agent_request = context_builder.build_promotion_request(
        user_message=state["user_message"],
        session_state=session_state,
        intent=intent,
    )

    promotion_agent = PromotionAgent()
    agent_result = await promotion_agent.run(agent_request)

    return {"agent_result": agent_result}


async def _clarification_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """
    Node 3e: Handles ambiguous messages.
    No agent dispatched — Orchestrator asks a clarifying question directly.
    """
    session_state = state["session_state"]
    synthesizer = get_response_synthesizer()

    clarification_text = await synthesizer.synthesize_clarification(
        user_message=state["user_message"],
        session_state=session_state,
    )

    logger.info(
        LogEvent.ORCHESTRATOR_RESPONSE_SYNTHESIZED,
        "Clarification response generated",
        session_id=session_state.session_id,
    )

    # Build a stub AgentResult so the synthesize node stays consistent
    from schemas.agent_io import AgentResult as AR
    stub_result = AR(
        request_id=str(uuid4()),
        agent_name="orchestrator_clarification",
        success=True,
        data={"clarification_text": clarification_text},
        confidence=1.0,
    )

    return {
        "agent_result": stub_result,
        "response_text": clarification_text,
        "response_ready": True,
    }


async def _synthesize_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """
    Node 4: Converts agent result → user-facing prose.
    Skipped if response_ready=True (clarification already has text).
    """
    if state.get("response_ready"):
        return {}

    agent_result = state["agent_result"]
    intent = state["intent"]
    session_state = state["session_state"]

    synthesizer = get_response_synthesizer()

    response_text = await synthesizer.synthesize(
        agent_result=agent_result,
        intent=intent,
        session_state=session_state,
    )

    return {
        "response_text": response_text,
        "response_ready": True,
    }


async def _persist_session_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """
    Node 5: Appends conversation turns and persists session to PostgreSQL.
    Always runs — even if earlier nodes failed.
    This ensures the session is never left in a corrupted state.
    """
    session_state = state["session_state"]
    response_text = state.get("response_text")
    agent_result = state.get("agent_result")
    intent = state.get("intent")

    if not response_text and agent_result:
        response_text = agent_result.data.get("response_text", "")

    import uuid

    # Append user turn
    user_turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        role=MessageRole.USER,
        content=state["user_message"],
        intent=intent,
        metadata={
            "request_id": state["request_id"],
            "classification_confidence": (
                state["intent_classification"].confidence
                if state.get("intent_classification")
                else None
            ),
        },
    )
    session_state.add_turn(user_turn)

    # Append assistant turn
    if response_text:
        products_shown = []
        if agent_result and agent_result.success:
            products_shown = [
                p.get("product_id", "")
                for p in agent_result.data.get("products", [])
                if p.get("product_id")
            ]

        assistant_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            role=MessageRole.ASSISTANT,
            content=response_text,
            agent_used=(
                agent_result.agent_name
                if agent_result
                else "orchestrator"
            ),
            products_shown=products_shown,
            metadata={
                "agent_confidence": (
                    agent_result.confidence
                    if agent_result
                    else None
                ),
                "guardrail_violations": (
                    agent_result.metadata.get("violations_count", 0)
                    if agent_result
                    else 0
                ),
            },
        )
        session_state.add_turn(assistant_turn)

    # Compute total latency
    total_latency_ms = (time.perf_counter() - state["start_time"]) * 1000

    # Persist full session state to PostgreSQL
    await persist_session(session_state)

    logger.info(
        LogEvent.ORCHESTRATOR_END,
        "Orchestrator completed",
        session_id=session_state.session_id,
        intent=intent.value if intent else "unknown",
        total_latency_ms=round(total_latency_ms, 2),
        response_length=len(response_text),
        total_turns=session_state.total_turns,
    )

    return {"total_latency_ms": total_latency_ms}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_by_intent(state: OrchestratorState) -> str:
    """
    Conditional edge: routes to the correct agent node based on classified intent.

    This is the ONLY place intent → agent mapping exists.
    Adding a new Phase 2/3 agent = add one case here + one node above.
    """
    intent = state.get("intent", IntentType.UNKNOWN)

    routing_map = {
        # Phase 1
        IntentType.PRODUCT_SEARCH:      "discovery",
        IntentType.PRODUCT_DETAIL:      "discovery",

        # Phase 2
        IntentType.ADD_TO_CART:         "cart",
        IntentType.REMOVE_FROM_CART:    "cart",
        IntentType.VIEW_CART:           "cart",
        IntentType.UPDATE_CART:         "cart",
        IntentType.CHECKOUT:            "checkout",
        IntentType.ORDER_STATUS:        "checkout",

        # Phase 3
        IntentType.PROMOTION_INQUIRY:       "promotion",
        IntentType.RECOMMENDATION_REQUEST:  "promotion",

        # Clarification
        IntentType.CLARIFICATION:       "clarification",
        IntentType.UNKNOWN:             "clarification",
    }

    destination = routing_map.get(intent, "clarification")

    logger.info(
        LogEvent.ORCHESTRATOR_AGENT_DISPATCHED,
        "Routing to agent",
        intent=intent.value,
        destination=destination,
    )

    return destination


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_orchestrator_graph() -> StateGraph:
    """
    Constructs the master Orchestrator LangGraph StateGraph.

    Full graph topology:
        START
          │
          ▼
        load_session
          │
          ▼
        intent_router
          │
          ├──► discovery   ──┐
          ├──► cart        ──┤
          ├──► checkout    ──┤──► synthesize ──► persist_session ──► END
          ├──► promotion   ──┤
          └──► clarification┘ (skips synthesize — has response_ready=True)
    """
    graph = StateGraph(OrchestratorState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("load_session",      _load_session_node)
    graph.add_node("intent_router",     _intent_router_node)
    graph.add_node("discovery",         _discovery_node)
    graph.add_node("cart",              _cart_node)
    graph.add_node("checkout",          _checkout_node)
    graph.add_node("promotion",         _promotion_node)
    graph.add_node("clarification",     _clarification_node)
    # graph.add_node("synthesize",        _synthesize_node)
    graph.add_node("persist_session",   _persist_session_node)

    # ── Add edges ─────────────────────────────────────────────────────────
    graph.add_edge(START, "load_session")
    graph.add_edge("load_session", "intent_router")

    # Intent router → correct agent node
    graph.add_conditional_edges(
        "intent_router",
        _route_by_intent,
        {
            "discovery":    "discovery",
            "cart":         "cart",
            "checkout":     "checkout",
            "promotion":    "promotion",
            "clarification": "clarification",
        },
    )

    # # All agent nodes → synthesize
    # for agent_node in ("discovery", "cart", "checkout", "promotion", "clarification"):
    #     graph.add_edge(agent_node, "synthesize")
    for agent_node in ("discovery", "cart", "checkout", "promotion", "clarification"):
        graph.add_edge(agent_node, "persist_session")

    # synthesize → persist → END
    # graph.add_edge("synthesize", "persist_session")
    graph.add_edge("persist_session", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# OrchestratorAgent — public interface
# ---------------------------------------------------------------------------

class OrchestratorAgent:
    """
    The master agent. Single entry point for all user messages.

    Usage (from API layer):
        orchestrator = get_orchestrator()
        result = await orchestrator.process(
            user_message="Show me badminton shoes under 2000",
            session_id="sess_01J...",
            entity_id="entity_001",
            business_unit_id="bu_001",
        )
        response_text = result["response_text"]
        session_id    = result["session_id"]

    The graph is compiled once at instantiation.
    All requests share the same compiled graph — thread-safe via LangGraph.
    """

    def __init__(self) -> None:
        self._graph = _build_orchestrator_graph()

        logger.info(
            LogEvent.APP_STARTUP,
            "Orchestrator graph compiled",
            nodes=[
                "load_session", "intent_router",
                "discovery", "cart", "checkout", "promotion",
                "clarification", "synthesize", "persist_session",
            ],
        )

    async def process(
        self,
        user_message: str,
        session_id: str | None = None,
        business_unit_id: str = "default_bu",
        entity_id: str = "default_entity",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Processes a single user message through the full Orchestrator graph.

        Args:
            user_message: Raw user input.
            session_id:   Existing session ID or None (new session created).
            entity_id:     Honebi entity scope.
            business_unit_id: Honebi business unit scope.
            user_id:      Authenticated user ID (None for anonymous).

        Returns:
            {
                "response_text": str,       # User-facing response
                "session_id":    str,       # Session ID (new or existing)
                "intent":        str,       # Classified intent
                "latency_ms":    float,     # Total end-to-end latency
            }
        """
        effective_session_id = session_id or str(uuid4())
        request_id = str(uuid4())

        initial_state: OrchestratorState = {
            "user_message": user_message,
            "session_id": effective_session_id,
            "business_unit_id": business_unit_id,
            "entity_id": entity_id,
            "user_id": user_id,
            "request_id": request_id,
            "session_state": None,
            "intent_classification": None,
            "intent": None,
            "agent_result": None,
            "response_text": "",
            "response_ready": False,
            "start_time": time.perf_counter(),
            "total_latency_ms": 0.0,
        }

        final_state = await self._graph.ainvoke(initial_state)

        products: list[dict] = []
        agent_result = final_state.get("agent_result")
        if agent_result and agent_result.success:
            products = agent_result.data.get("products", [])

        follow_up = agent_result.data.get("follow_up_question") if (
            agent_result and agent_result.success
        ) else None

        return {
            "response_text": final_state.get("response_text", ""),
            "session_id": effective_session_id,
            "intent": (
                final_state["intent"].value
                if final_state.get("intent")
                else "unknown"
            ),
            "latency_ms": round(final_state.get("total_latency_ms", 0.0), 2),
            "products": products,               # ← structured products for API layer
            "follow_up_question": follow_up,    # ← follow-up for quick-reply chips
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_orchestrator: OrchestratorAgent | None = None


def get_orchestrator() -> OrchestratorAgent:
    """
    Returns the singleton OrchestratorAgent.
    Graph compiled on first call — not per request.
    Subsequent calls return the cached instance with zero overhead.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator