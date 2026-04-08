# conversational_commerce/agents/discovery_agent.py

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from schemas.query import ParsedQuery
from services.attribute_store import AttributeStore
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.base_agent import BaseAgent
from agents.prompts import (
    DISCOVERY_AGENT_SYSTEM_PROMPT,
    DISCOVERY_AGENT_PROMPT_VERSION,
    QUERY_PARSER_PROMPT_VERSION,
    QUERY_PARSER_SYSTEM_PROMPT,
)
from config.settings import get_settings
from guardrails.discovery_guard import get_discovery_guardrail
from observability.logger import LogEvent, get_logger
from schemas.agent_io import AgentRequest, AgentResult
from tools.registry import get_tool_registry

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class DiscoveryAgentState(TypedDict):
    """
    The complete state object passed between nodes in the Discovery Agent graph.

    LangGraph passes this dict through every node.
    Nodes read from it and return partial updates — LangGraph merges them.

    `messages` uses add_messages reducer:
        Each node appends messages — never replaces the full list.
        This is how LangGraph maintains multi-turn tool call history
        within a single agent invocation.

    All other fields are plain values — last-write-wins.
    """

    # LangGraph message history for this agent invocation
    # Includes: SystemMessage, HumanMessage, AIMessage (tool calls), ToolMessages
    messages: Annotated[list[BaseMessage], add_messages]

    # Request and result — set at start and end of graph
    request: AgentRequest
    result: AgentResult | None

    # Retrieval tracking — used by guardrail node
    # Populated by tool execution, consumed by guardrail node
    retrieval_product_ids: set[str]     # Ground truth from search_products tool
    tool_results_raw: list[dict[str, Any]]  # Raw tool outputs for guardrail

    # Execution metadata
    search_performed: bool
    tool_call_count: int
    prompt_version: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def _entry_node(state: DiscoveryAgentState) -> dict[str, Any]:
    request = state["request"]
    context = request.context

    context_block = _build_context_block(request)

    human_message = HumanMessage(
        content=(
            f"{context_block}\n\n"
            f"Customer request: {request.user_message}\n\n"
            f"Task: {request.task}"
        )
    )

    return {
        "messages": [
            SystemMessage(content=DISCOVERY_AGENT_SYSTEM_PROMPT),
            human_message,
        ],
        "retrieval_product_ids": set(),
        "tool_results_raw": [],
        "search_performed": False,
        "tool_call_count": 0,
        "prompt_version": DISCOVERY_AGENT_PROMPT_VERSION,
        "result": None,
        "parsed_query": None,
    }

async def _query_parser_node(
    state: DiscoveryAgentState,
) -> dict[str, Any]:
    """
    Dedicated query parsing node — now attribute-constrained.

    The attribute store is loaded at startup and injected here.
    The LLM receives the full list of valid catalog values and
    MUST choose from them — no hallucinated attribute values.
    """
    from agents.prompts import build_query_parser_prompt
    from services.attribute_store import get_attribute_store

    request = state["request"]
    context = request.context

    # ── Build parser input (unchanged) ───────────────────────────────────
    parser_input_parts = []

    last_filters = context.get("last_filters", {})
    if last_filters:
        parser_input_parts.append("[PREVIOUS FILTERS]")
        for key, value in last_filters.items():
            if value is not None:
                parser_input_parts.append(f"{key}: {value}")
        parser_input_parts.append("")

    last_query = context.get("last_query", "")
    if last_query:
        parser_input_parts.append(f"[PREVIOUS SEARCH]\n{last_query}\n")

    parser_input_parts.append(
        f"[CUSTOMER MESSAGE]\n{request.user_message}"
    )

    parser_input = "\n".join(parser_input_parts)

    # ── Build attribute-constrained prompt ────────────────────────────────
    # This is the key change — the prompt now contains the full
    # list of valid attribute values from the catalog.
    # The LLM is forced to choose from these exact values.
    attribute_store = get_attribute_store()
    attribute_block = (
        attribute_store.build_prompt_block()
        if attribute_store.is_loaded
        else ""
    )
    system_prompt = build_query_parser_prompt(attribute_block)

    logger.info(
        LogEvent.LLM_REQUEST,
        "Query parser node started",
        request_id=request.request_id,
        has_previous_filters=bool(last_filters),
        has_previous_query=bool(last_query),
        has_attribute_constraints=bool(attribute_block),
        attribute_key_count=(
            len(attribute_store.get_all())
            if attribute_store.is_loaded else 0
        ),
        message_preview=request.user_message[:100],
        prompt_version=QUERY_PARSER_PROMPT_VERSION,
    )

    # ── Parser LLM call ───────────────────────────────────────────────────
    parser_llm = ChatOpenAI(
        model=settings.openai.chat_model,
        temperature=0.0,
        max_tokens=512,
        timeout=settings.openai.request_timeout_seconds,
        api_key=settings.openai.api_key,
    )

    async with logger.timed(
        LogEvent.LLM_RESPONSE,
        "query_parser_llm_call",
        request_id=request.request_id,
    ):
        response = await parser_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=parser_input),
        ])

    # ── Parse LLM output → ParsedQuery (unchanged) ───────────────────────
    parsed_query = _build_parsed_query_from_llm(
        llm_response=response.content,
        original_message=request.user_message,
        entity_id=context.get("entity_id"),
        business_unit_id=context.get("business_unit_id"),
        result_limit=context.get("result_limit", 10),
    )

    # ── Post-parse validation against attribute store ─────────────────────
    # Even after the prompt constraint, validate the LLM's choices.
    # LLMs occasionally deviate — this catches and corrects it silently.
    if attribute_store.is_loaded:
        parsed_query = _validate_constraints_against_store(
            parsed_query, attribute_store
        )

    logger.info(
        LogEvent.LLM_RESPONSE,
        "Query parsed successfully",
        request_id=request.request_id,
        semantic_query=parsed_query.semantic_query,
        category=parsed_query.hard_constraints.category,
        sub_category=parsed_query.hard_constraints.sub_category,
        brand=parsed_query.hard_constraints.brand,
        color=parsed_query.hard_constraints.color,
        material=parsed_query.hard_constraints.material,
        price_max=(
            parsed_query.hard_constraints.price_range.max_price
            if parsed_query.hard_constraints.price_range else None
        ),
        price_min=(
            parsed_query.hard_constraints.price_range.min_price
            if parsed_query.hard_constraints.price_range else None
        ),
        sort_order=parsed_query.sort_order.value,
        inference_notes=parsed_query.inference_notes,
    )

    return {"parsed_query": parsed_query}

async def _llm_node(state: DiscoveryAgentState) -> dict[str, Any]:
    llm = _get_llm()
    tools = get_tool_registry().get_tools_for_agent("discovery_agent")
    llm_with_tools = llm.bind_tools(tools)

    messages = list(state["messages"])

    # ── Inject ParsedQuery as structured context ──────────────────────────
    # The parser node already extracted all filters.
    # We tell the LLM exactly what was parsed so it uses these values
    # directly in the tool call — no re-derivation from raw text.
    parsed_query = state.get("parsed_query")
    if parsed_query and not state.get("search_performed"):
        parsed_query_context = _build_parsed_query_context_message(parsed_query)
        messages = messages + [
            HumanMessage(content=parsed_query_context)
        ]

    logger.info(
        LogEvent.LLM_REQUEST,
        "Discovery agent LLM call",
        model=settings.openai.chat_model,
        message_count=len(messages),
        has_parsed_query=parsed_query is not None,
    )

    async with logger.timed(
        LogEvent.LLM_RESPONSE,
        "llm_inference",
        model=settings.openai.chat_model,
    ):
        response = await llm_with_tools.ainvoke(messages)

    token_usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        token_usage = {
            "input_tokens": response.usage_metadata.get("input_tokens", 0),
            "output_tokens": response.usage_metadata.get("output_tokens", 0),
        }

    logger.info(
        LogEvent.LLM_RESPONSE,
        "Discovery agent LLM response",
        has_tool_calls=bool(
            hasattr(response, "tool_calls") and response.tool_calls
        ),
        **token_usage,
    )

    return {
        "messages": [response],
        "tool_call_count": state["tool_call_count"] + 1,
    }


async def _tool_execution_node(state: DiscoveryAgentState) -> dict[str, Any]:
    """
    Tool execution node: runs the tools the LLM decided to call.

    Uses LangGraph's built-in ToolNode for execution.
    After execution, extracts product IDs from search_products results
    to populate retrieval_product_ids for the guardrail.
    """
    tools = get_tool_registry().get_tools_for_agent("discovery_agent")
    tool_node = ToolNode(tools)

    # Execute tools via LangGraph ToolNode
    tool_result_state = await tool_node.ainvoke(state)

    # Extract product IDs from search_products results for guardrail ground truth
    retrieval_ids = set(state.get("retrieval_product_ids", set()))
    raw_results = list(state.get("tool_results_raw", []))
    search_performed = state.get("search_performed", False)

    for message in tool_result_state.get("messages", []):
        if isinstance(message, ToolMessage):
            try:
                tool_output = json.loads(message.content)

                # Track all tool outputs for guardrail
                raw_results.append(tool_output)

                # Extract product IDs from search_products results
                if tool_output.get("success") and "products" in tool_output:
                    for product in tool_output["products"]:
                        pid = product.get("product_id")
                        if pid:
                            retrieval_ids.add(pid)
                    search_performed = True

                    logger.info(
                        LogEvent.AGENT_TOOL_RESULT,
                        "search_products tool result processed",
                        product_count=len(tool_output["products"]),
                        retrieval_ids_count=len(retrieval_ids),
                    )

            except (json.JSONDecodeError, TypeError):
                # Tool returned non-JSON — log and continue
                logger.warning(
                    LogEvent.AGENT_TOOL_RESULT,
                    "Tool returned non-JSON content",
                    content_preview=str(message.content)[:200],
                )

    return {
        "messages": tool_result_state.get("messages", []),
        "retrieval_product_ids": retrieval_ids,
        "tool_results_raw": raw_results,
        "search_performed": search_performed,
    }


async def _guardrail_node(state: DiscoveryAgentState) -> dict[str, Any]:
    """
    Guardrail node: validates the LLM's final response before packaging.

    Parses the LLM's JSON response, runs DiscoveryGuardrail validation,
    and builds the AgentResult.

    If the guardrail blocks the result → AgentResult.failure
    If the guardrail sanitises → AgentResult with sanitised data
    If the guardrail passes → AgentResult with full data
    """
    request = state["request"]
    messages = state["messages"]
    retrieval_product_ids = state.get("retrieval_product_ids", set())

    # Extract the LLM's last AIMessage
    last_ai_message = _get_last_ai_message(messages)

    if last_ai_message is None:
        logger.error(
            LogEvent.GUARDRAIL_VIOLATION,
            "No AI message found in Discovery agent state",
            request_id=request.request_id,
        )
        return {
            "result": AgentResult.failure(
                request_id=request.request_id,
                agent_name="discovery_agent",
                error_code="NO_LLM_RESPONSE",
                error_detail="LLM did not produce a final response message",
            )
        }

    # Parse the LLM's JSON response
    agent_output, parse_error = _parse_llm_json_response(
        last_ai_message.content
    )

    if parse_error:
        logger.error(
            LogEvent.GUARDRAIL_VIOLATION,
            "Failed to parse Discovery agent JSON response",
            error=parse_error,
            content_preview=str(last_ai_message.content)[:300],
        )
        return {
            "result": AgentResult.failure(
                request_id=request.request_id,
                agent_name="discovery_agent",
                error_code="RESPONSE_PARSE_ERROR",
                error_detail=parse_error,
            )
        }

    # Run Discovery guardrail
    guardrail = get_discovery_guardrail()
    entity_id = request.context.get("entity_id")
    business_unit_id = request.context.get("business_unit_id")
    result_limit = request.context.get("result_limit", 10)
    confidence = float(agent_output.get("confidence", 1.0))

    guardrail_result = guardrail.validate(
        agent_result_data=agent_output,
        retrieval_product_ids=retrieval_product_ids,
        requested_entity_id=entity_id,
        requested_business_unit_id=business_unit_id,
        requested_result_limit=result_limit,
        agent_confidence=confidence,
    )

    # Build AgentResult based on guardrail outcome
    if guardrail_result.should_block:
        agent_result = AgentResult.failure(
            request_id=request.request_id,
            agent_name="discovery_agent",
            error_code="GUARDRAIL_BLOCKED",
            error_detail=guardrail_result.block_reason or "Critical guardrail violation",
            metadata={
                "violations": [
                    v.to_dict() for v in guardrail_result.violations
                ],
            },
        )
    else:
        # Use sanitised product list from guardrail
        sanitised_output = {
            **agent_output,
            "products": guardrail_result.sanitised_products,
        }

        agent_result = AgentResult(
            request_id=request.request_id,
            agent_name="discovery_agent",
            success=True,
            data=sanitised_output,
            confidence=confidence,
            metadata={
                "guardrail_passed": guardrail_result.passed,
                "violations_count": len(guardrail_result.violations),
                "search_performed": state.get("search_performed", False),
                "tool_call_count": state.get("tool_call_count", 0),
                "prompt_version": state.get("prompt_version", "unknown"),
                "retrieval_product_count": len(retrieval_product_ids),
                "parsed_query_filters": (
                    state["parsed_query"].hard_constraints.model_dump(mode="json")
                    if state.get("parsed_query")
                    else {}
                ),
            },
        )

    return {"result": agent_result}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _should_continue(state: DiscoveryAgentState) -> str:
    """
    LangGraph conditional edge — determines next node after LLM response.

    Returns:
        "tools"    → LLM made tool calls, execute them
        "guardrail" → LLM produced final response, validate it
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # Safety check: too many tool calls → force to guardrail
    # Prevents infinite tool call loops
    if state.get("tool_call_count", 0) >= 5:
        logger.warning(
            LogEvent.AGENT_END,
            "Discovery agent tool call limit reached — forcing to guardrail",
            tool_call_count=state["tool_call_count"],
        )
        return "end"

    # LangGraph standard: AIMessage with tool_calls → execute tools
    if (
        last_message is not None
        and isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        return "tools"

    # return "guardrail"
    return "end"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def _build_discovery_graph() -> StateGraph:
    """
    Updated graph topology with dedicated query parser node:

        START → entry → query_parser → llm ⟺ tools → guardrail → END
                                ↑
                        NEW — runs before LLM,
                        extracts ParsedQuery from
                        user message + session context
    """
    graph = StateGraph(DiscoveryAgentState)

    graph.add_node("entry",         _entry_node)
    graph.add_node("query_parser",  _query_parser_node)
    graph.add_node("llm",           _llm_node)
    graph.add_node("tools",         _tool_execution_node)
    # graph.add_node("guardrail",     _guardrail_node)

    graph.add_edge(START,           "entry")
    graph.add_edge("entry",         "query_parser")
    graph.add_edge("query_parser",  "llm")

    graph.add_conditional_edges(
        "llm",
        _should_continue,
        {
            "tools":     "tools",
            # "guardrail": "guardrail",
            "end":       END,
        },
    )

    graph.add_edge("tools",     "llm")
    # graph.add_edge("guardrail", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Discovery Agent class
# ---------------------------------------------------------------------------

class DiscoveryAgent(BaseAgent):
    """
    Phase 1 specialist: product discovery.

    Responsibilities:
        - Search the product catalog using natural language
        - Retrieve full product details when requested
        - Return structured product data + response text to Orchestrator

    Does NOT:
        - Speak to the user (Orchestrator synthesizer does this)
        - Access session state (receives curated context via AgentRequest)
        - Make cart/checkout decisions (Phase 2 agents handle these)
        - Call other agents

    LangGraph graph:
        entry → llm ⟺ tools → guardrail → END

    The graph is compiled once at class instantiation and reused
    across all requests — compilation is expensive, execution is not.
    """

    agent_name = "discovery_agent"

    def __init__(self) -> None:
        self._graph = _build_discovery_graph()

        logger.info(
            LogEvent.APP_STARTUP,
            "Discovery agent graph compiled",
            agent_name=self.agent_name,
            prompt_version=DISCOVERY_AGENT_PROMPT_VERSION,
        )

    async def _execute(self, request: AgentRequest) -> AgentResult:
        """
        Runs the Discovery Agent LangGraph graph.

        Initialises state, invokes the compiled graph, and returns
        the AgentResult from the guardrail node.
        """
        initial_state: DiscoveryAgentState = {
            "messages": [],
            "request": request,
            "result": None,
            "retrieval_product_ids": set(),
            "tool_results_raw": [],
            "search_performed": False,
            "tool_call_count": 0,
            "prompt_version": DISCOVERY_AGENT_PROMPT_VERSION,
        }

        logger.info(
            LogEvent.AGENT_START,
            "Discovery agent graph invoked",
            request_id=request.request_id,
            session_id=request.session_id,
        )

        final_state = await self._graph.ainvoke(initial_state)

        result = final_state.get("result")

        if result is None:
            # Should never happen — guardrail node always sets result
            # Defensive fallback
            return AgentResult.failure(
                request_id=request.request_id,
                agent_name=self.agent_name,
                error_code="GRAPH_NO_RESULT",
                error_detail=(
                    "Discovery agent graph completed but result was not set. "
                    "Check guardrail node logs."
                ),
            )

        return result
    

class DiscoveryAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    request: AgentRequest
    result: AgentResult | None

    # ── ADD THIS FIELD ──────────────────────────────────────────
    parsed_query: ParsedQuery | None
    # Populated by _query_parser_node before the LLM tool-call node.
    # The tool execution node reads this instead of letting the LLM
    # reconstruct filters from scratch inside the tool call.
    # ────────────────────────────────────────────────────────────

    retrieval_product_ids: set[str]
    tool_results_raw: list[dict[str, Any]]
    search_performed: bool
    tool_call_count: int
    prompt_version: str


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_discovery_agent: DiscoveryAgent | None = None


def get_discovery_agent() -> DiscoveryAgent:
    """
    Returns the singleton DiscoveryAgent.
    The LangGraph graph is compiled on first call — not at every request.
    """
    global _discovery_agent
    if _discovery_agent is None:
        _discovery_agent = DiscoveryAgent()
    return _discovery_agent


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_llm_instance: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """Returns the singleton ChatOpenAI instance for this agent."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=settings.openai.chat_model,
            temperature=settings.openai.temperature,
            max_tokens=settings.openai.max_tokens,
            timeout=settings.openai.request_timeout_seconds,
            api_key=settings.openai.api_key,
        )
    return _llm_instance


def _build_context_block(request: AgentRequest) -> str:
    """
    Builds the dynamic context block injected into the human message.

    This is how the agent receives session-derived context:
      - Which store it's operating in
      - Which products the user has already seen (exclusion list)
      - What the user last searched for (refinement context)
      - Result limit for this request

    Kept as structured text — easier for the LLM to parse than JSON.
    """
    context = request.context
    parts = ["[SESSION CONTEXT]"]

    if entity_id := context.get("entity_id"):
        parts.append(f"entity_id: {entity_id}")

    if business_unit_id := context.get("business_unit_id"):
        parts.append(f"business_unit_id: {business_unit_id}")

    if last_query := context.get("last_query"):
        parts.append(f"previous_search: {last_query}")

    if shown_ids := context.get("shown_product_ids", []):
        parts.append(
            f"exclude_product_ids: {json.dumps(shown_ids)}"
            f"  # These products are already shown — exclude from results"
        )

    if result_limit := context.get("result_limit", 10):
        parts.append(f"result_limit: {result_limit}")

    if constraints := request.constraints:
        parts.append(f"business_constraints: {json.dumps(constraints)}")

    return "\n".join(parts)


def _get_last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """Returns the most recent AIMessage from the message list."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not (
            hasattr(message, "tool_calls") and message.tool_calls
        ):
            return message
    return None


def _parse_llm_json_response(
    content: str | list,
) -> tuple[dict[str, Any], str | None]:
    """
    Parses the LLM's JSON response content.

    The Discovery Agent prompt instructs the LLM to return a JSON object.
    This function extracts and validates that JSON.

    Returns:
        (parsed_dict, None)           on success
        ({}, error_message)           on failure
    """
    if isinstance(content, list):
        # Multi-part content — extract text parts
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        content = " ".join(text_parts)

    if not isinstance(content, str):
        return {}, f"Unexpected content type: {type(content).__name__}"

    content = content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```) and last line (```)
        content = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {}, f"Expected JSON object, got {type(parsed).__name__}"
        return parsed, None

    except json.JSONDecodeError as e:
        return {}, f"JSON parse error: {str(e)}"
    


def _build_parsed_query_from_llm(
    llm_response: str | list,
    original_message: str,
    business_unit_id: str | None,
    entity_id: str | None,
    result_limit: int,
) -> ParsedQuery:
    """
    Converts sparse LLM filter output → ParsedQuery.

    LLM now returns ONLY explicitly extracted fields.
    Missing fields default to None in HardConstraints.
    No forced nulls. No hallucinated constraints.
    """
    from schemas.query import HardConstraints, ParsedQuery, PriceRange, SortOrder

    if isinstance(llm_response, list):
        llm_response = " ".join(
            p.get("text", "") for p in llm_response
            if isinstance(p, dict)
        )

    content = str(llm_response).strip()

    # Strip markdown fences if LLM adds them despite instructions
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )

    try:
        data = json.loads(content)

        # ── Sparse filter extraction ───────────────────────────────────
        # Only fields present in the JSON are set.
        # Absent fields → None in HardConstraints (not forced null).
        filters = data.get("filters", {})

        price_range = None
        if "price_range" in filters:
            pr = filters["price_range"]
            min_p = pr.get("min_price")
            max_p = pr.get("max_price")
            if min_p is not None or max_p is not None:
                price_range = PriceRange(
                    min_price=float(min_p) if min_p is not None else None,
                    max_price=float(max_p) if max_p is not None else None,
                )

        # sort_order inside filters (optional)
        sort_order = SortOrder.RELEVANCE
        if "sort_order" in filters:
            try:
                sort_order = SortOrder(filters["sort_order"])
            except ValueError:
                sort_order = SortOrder.RELEVANCE

        constraints = HardConstraints(
            price_range=price_range,
            category=filters.get("category"),
            sub_category=filters.get("sub_category"),
            brand=filters.get("brand"),
            gender=filters.get("gender"),
            size=filters.get("size"),
            color=filters.get("color"),
            material=filters.get("material"),
            min_rating=(
                float(filters["min_rating"])
                if "min_rating" in filters else None
            ),
            in_stock_only=True,         # Always — never from LLM
            business_unit_id=business_unit_id,
            entity_id=entity_id,
            dynamic_filters=filters.get("dynamic_filters", {}),
        )

        return ParsedQuery(
            semantic_query=data.get("semantic_query", original_message),
            hard_constraints=constraints,
            sort_order=sort_order,
            result_limit=int(data.get("result_limit", result_limit)),
            inference_notes=data.get("inference_notes", ""),
            original_query=original_message,
        )

    except Exception as e:
        logger.warning(
            LogEvent.LLM_RESPONSE,
            "Query parser fallback triggered",
            error=str(e),
            original_message=original_message[:100],
        )
        return ParsedQuery(
            semantic_query=original_message,
            hard_constraints=HardConstraints(
                in_stock_only=True,
                business_unit_id=business_unit_id,
                entity_id=entity_id,
            ),
            sort_order=SortOrder.RELEVANCE,
            result_limit=result_limit,
            inference_notes=f"Fallback: parser error — {str(e)}",
            original_query=original_message,
        )


def _build_parsed_query_context_message(parsed_query: ParsedQuery) -> str:
    """
    Builds a structured context message from ParsedQuery
    to inject into the LLM node's message list.

    This tells the tool-calling LLM exactly what filters were extracted
    so it uses them directly in the search_products call
    rather than re-deriving them from the raw user message.
    """
    hc = parsed_query.hard_constraints
    lines = [
        "[PARSED QUERY — USE THESE VALUES IN YOUR TOOL CALL]",
        f"semantic_query: {parsed_query.semantic_query}",
    ]

    if hc.category:
        lines.append(f"category: {hc.category}")
    if hc.sub_category:
        lines.append(f"sub_category: {hc.sub_category}")
    if hc.brand:
        lines.append(f"brand: {hc.brand}")
    if hc.gender:
        lines.append(f"gender: {hc.gender}")
    if hc.color:
        lines.append(f"color: {hc.color}")
    if hc.material:
        lines.append(f"material: {hc.material}")
    if hc.size:
        lines.append(f"size: {hc.size}")
    if hc.min_rating is not None:
        lines.append(f"min_rating: {hc.min_rating}")
    if hc.price_range:
        if hc.price_range.min_price is not None:
            lines.append(f"min_price: {hc.price_range.min_price}")
        if hc.price_range.max_price is not None:
            lines.append(f"max_price: {hc.price_range.max_price}")
    if hc.dynamic_filters:
        lines.append(f"dynamic_filters: {hc.dynamic_filters}")

    lines.append(f"sort_order: {parsed_query.sort_order.value}")
    lines.append(f"result_limit: {parsed_query.result_limit}")
    lines.append(
        "\nInstruction: Call search_products using exactly these values. "
        "Do not re-extract filters from the customer message — "
        "the parsing has already been done above."
    )

    return "\n".join(lines)

def _validate_constraints_against_store(
    parsed_query: ParsedQuery,
    attribute_store: "AttributeStore",
) -> ParsedQuery:
    """
    Post-parse validation layer.

    Even though the prompt instructs the LLM to use only valid catalog values,
    LLMs occasionally deviate. This function checks every constraint field
    against the attribute store and nullifies any value not found.

    This is the LAST guardrail before constraints hit the SQL layer.
    Silently corrects — never raises. Logs every correction for audit.

    Design: creates a new HardConstraints rather than mutating —
    ParsedQuery fields are validated Pydantic models.
    """
    from schemas.query import HardConstraints, ParsedQuery as PQ

    hc = parsed_query.hard_constraints
    corrections: list[str] = []

    def validate_field(
        field_name: str,
        value: str | None,
        store_key: str,
    ) -> str | None:
        """
        Returns value if valid, None if not found in attribute store.
        Case-insensitive comparison — both sides lowercased.
        """
        if value is None:
            return None

        valid_values = attribute_store.get(store_key)

        if not valid_values:
            # Store has no values for this key — pass through unchanged
            return value

        # Case-insensitive membership check
        value_lower = value.strip().lower()
        valid_lower = [v.lower() for v in valid_values]

        if value_lower in valid_lower:
            # Return the canonical form from the store (correct casing)
            idx = valid_lower.index(value_lower)
            return valid_values[idx]
        else:
            corrections.append(
                f"{field_name}='{value}' not in catalog — set to null"
            )
            return None

    # Validate each constrained field against the attribute store
    validated_category = validate_field(
        "category", hc.category, "categories"
    )
    validated_sub_category = validate_field(
        "sub_category", hc.sub_category, "sub_categories"
    )
    validated_brand = validate_field(
        "brand", hc.brand, "brands"
    )
    validated_gender = validate_field(
        "gender", hc.gender, "gender"
    )

    if corrections:
        logger.warning(
            LogEvent.LLM_RESPONSE,
            "Query parser produced non-catalog values — corrected",
            corrections=corrections,
            original_category=hc.category,
            original_brand=hc.brand,
        )

    # Rebuild constraints with validated values
    validated_constraints = HardConstraints(
        price_range=hc.price_range,
        category=validated_category,
        sub_category=validated_sub_category,
        brand=validated_brand,
        gender=validated_gender,
        min_rating=hc.min_rating,
        in_stock_only=hc.in_stock_only,
        entity_id=hc.entity_id,
        business_unit_id=hc.business_unit_id,
        dynamic_filters=hc.dynamic_filters,
    )

    # Return new ParsedQuery with validated constraints
    return parsed_query.model_copy(
        update={"hard_constraints": validated_constraints}
    )
# ```

# ---

# ## What the Full Flow Looks Like Now
# ```
# Application Startup
# ──────────────────────────────────────────────────────────────
# PostgreSQL pool ready
#        │
#        ▼
# AttributeStore.load()
#   ├── SELECT label, fixed_values FROM attributes
#   │     → material: [cotton, leather, mesh, synthetic]
#   │     → color:    [black, blue, red, white, green]
#   │     → gender:   [men, women, unisex, kids]
#   │     → size:     [6, 7, 8, 9, 10, 11, XS, S, M, L, XL]
#   ├── SELECT name FROM categories
#   │     → categories: [apparel, handbags, saree, shoes]
#   ├── SELECT name FROM sub_categories
#   │     → sub_categories: [badminton, casual, ethnic, formal, sports]
#   └── SELECT DISTINCT brand FROM products
#         → brands: [Adidas, Nike, Puma, Yonex]

# Stored in memory. O(1) access forever.


# Every Query Parse
# ──────────────────────────────────────────────────────────────
# User: "lightweight badminton shoes under 2000"
#        │
#        ▼
# _query_parser_node
#   │
#   ├── attribute_store.build_prompt_block() →
#   │     [VALID CATALOG VALUES]
#   │     categories: apparel, handbags, saree, shoes
#   │     sub_categories: badminton, casual, ethnic, formal, sports
#   │     brands: Adidas, Nike, Puma, Yonex
#   │     material: cotton, leather, mesh, synthetic
#   │     color: black, blue, red, white
#   │     gender: kids, men, unisex, women
#   │     ...
#   │
#   ├── LLM (temp=0, max_tokens=512)
#   │     Input:  user message + attribute constraints
#   │     Output: {
#   │       semantic_query: "lightweight badminton sports shoes indoor court",
#   │       hard_constraints: {
#   │         category: "shoes",          ← from catalog ✅
#   │         sub_category: "badminton",  ← from catalog ✅
#   │         max_price: 2000,            ← extracted    ✅
#   │         material: null,             ← not stated   ✅
#   │         brand: null,                ← not stated   ✅
#   │       }
#   │     }
#   │
#   └── _validate_constraints_against_store()
#         category="shoes"     → in ["apparel","handbags","saree","shoes"] ✅
#         sub_category="badminton" → in sub_categories ✅
#         brand=null           → skip ✅
#         → ParsedQuery clean, no corrections needed

#        │
#        ▼
#   HybridRetriever
#     semantic_query → OpenAI embed → Milvus ANN
#     hard_constraints → SQL WHERE category='shoes'
#                               AND sub_category='badminton'
#                               AND base_price <= 2000
#                               AND in_stock = true


# AgentRequest (from Orchestrator)
#         │
#         ▼
# [1] Entry Node → inject context
#         │
#         ▼
# [2] Query Parser Node → structured query (ParsedQuery)
#         │
#         ▼
# [3] LLM Node (Tool Planner)
#         │
#         ├── calls tools (search_products)
#         ▼
# [4] Tool Execution Node → DB / Vector search
#         │
#         ▼
# (loop LLM ↔ tools until done)
#         │
#         ▼
# [5] Guardrail Node → validate & sanitize
#         │
#         ▼
# AgentResult (to Orchestrator)