# conversational_commerce/agents/discovery_agent.py
# Changes from your current file:
# 1. DiscoveryAgentState defined ONCE at top (you had it defined twice)
# 2. _tool_execution_node stores raw products in state
# 3. New _response_generator_node replaces guardrail node
# 4. Graph updated: tools → response_generator → END
# 5. _execute reads result from state["result"]

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from services.attribute_store import AttributeStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.base_agent import BaseAgent
from agents.prompts import (
    DISCOVERY_AGENT_SYSTEM_PROMPT,
    DISCOVERY_AGENT_PROMPT_VERSION,
    QUERY_PARSER_PROMPT_VERSION,
    build_query_parser_prompt,
)
from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.agent_io import AgentRequest, AgentResult
from schemas.query import ParsedQuery
from tools.registry import get_tool_registry

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# State — defined ONCE (your file had it defined twice — removed duplicate)
# ---------------------------------------------------------------------------

class DiscoveryAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    request: AgentRequest
    result: AgentResult | None
    parsed_query: ParsedQuery | None

    # Raw products from tools — facts live here, never passed to response LLM
    raw_products: list[dict[str, Any]]

    retrieval_product_ids: set[str]
    tool_results_raw: list[dict[str, Any]]
    search_performed: bool
    tool_call_count: int
    prompt_version: str


# ---------------------------------------------------------------------------
# Node 1 — Entry (unchanged from your file)
# ---------------------------------------------------------------------------

async def _entry_node(state: DiscoveryAgentState) -> dict[str, Any]:
    request = state["request"]
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
        "raw_products": [],             # ← initialise facts store
        "search_performed": False,
        "tool_call_count": 0,
        "prompt_version": DISCOVERY_AGENT_PROMPT_VERSION,
        "result": None,
        "parsed_query": None,
    }


# ---------------------------------------------------------------------------
# Node 2 — Query Parser (unchanged from your file)
# ---------------------------------------------------------------------------

async def _query_parser_node(state: DiscoveryAgentState) -> dict[str, Any]:
    from services.attribute_store import get_attribute_store

    request = state["request"]
    context = request.context

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

    parser_input_parts.append(f"[CUSTOMER MESSAGE]\n{request.user_message}")
    parser_input = "\n".join(parser_input_parts)

    attribute_store = get_attribute_store()
    attribute_block = (
        attribute_store.build_prompt_block()
        if attribute_store.is_loaded else ""
    )
    system_prompt = build_query_parser_prompt(attribute_block)

    logger.info(
        LogEvent.LLM_REQUEST,
        "Query parser node started",
        request_id=request.request_id,
        has_previous_filters=bool(last_filters),
        message_preview=request.user_message[:100],
        prompt_version=QUERY_PARSER_PROMPT_VERSION,
    )

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

    parsed_query = _build_parsed_query_from_llm(
        llm_response=response.content,
        original_message=request.user_message,
        entity_id=context.get("entity_id"),
        business_unit_id=context.get("business_unit_id"),
        result_limit=context.get("result_limit", 10),
    )

    attribute_store = get_attribute_store()
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
        inference_notes=parsed_query.inference_notes,
    )

    return {"parsed_query": parsed_query}


# ---------------------------------------------------------------------------
# Node 3 — LLM Tool Planner (unchanged from your file)
# ---------------------------------------------------------------------------

async def _llm_node(state: DiscoveryAgentState) -> dict[str, Any]:
    llm = _get_llm()
    tools = get_tool_registry().get_tools_for_agent("discovery_agent")
    llm_with_tools = llm.bind_tools(tools)

    messages = list(state["messages"])

    parsed_query = state.get("parsed_query")
    if parsed_query and not state.get("search_performed"):
        parsed_query_context = _build_parsed_query_context_message(parsed_query)
        messages = messages + [HumanMessage(content=parsed_query_context)]

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


# ---------------------------------------------------------------------------
# Node 4 — Tool Execution
# KEY CHANGE: raw products stored in state, never passed to response LLM
# ---------------------------------------------------------------------------

async def _tool_execution_node(state: DiscoveryAgentState) -> dict[str, Any]:
    tools = get_tool_registry().get_tools_for_agent("discovery_agent")
    tool_node = ToolNode(tools)

    tool_result_state = await tool_node.ainvoke(state)

    retrieval_ids = set(state.get("retrieval_product_ids", set()))
    raw_results = list(state.get("tool_results_raw", []))

    # ── KEY CHANGE: accumulate raw products in state ───────────────────────
    # Full product dicts stored here — the response generator
    # NEVER receives these. It only gets a lightweight summary.
    accumulated_products = list(state.get("raw_products", []))
    search_performed = state.get("search_performed", False)

    for message in tool_result_state.get("messages", []):
        if isinstance(message, ToolMessage):
            try:
                tool_output = json.loads(message.content)
                raw_results.append(tool_output)

                if tool_output.get("success") and "products" in tool_output:
                    products = tool_output["products"]

                    # Store full product data in state
                    accumulated_products.extend(products)

                    # Track IDs for guardrail
                    for product in products:
                        pid = product.get("product_id")
                        if pid:
                            retrieval_ids.add(pid)

                    search_performed = True

                    logger.info(
                        LogEvent.AGENT_TOOL_RESULT,
                        "search_products tool result stored in state",
                        product_count=len(products),
                        total_accumulated=len(accumulated_products),
                    )

            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    LogEvent.AGENT_TOOL_RESULT,
                    "Tool returned non-JSON content",
                    content_preview=str(message.content)[:200],
                )

    return {
        "messages": tool_result_state.get("messages", []),
        "retrieval_product_ids": retrieval_ids,
        "tool_results_raw": raw_results,
        "raw_products": accumulated_products,   # ← facts live here
        "search_performed": search_performed,
    }


# ---------------------------------------------------------------------------
# Node 5 — Response Generator (REPLACES guardrail node)
# KEY CHANGE: LLM gets lightweight summary only (~300 tokens)
# Facts are stitched by the system, not echoed by LLM
# ---------------------------------------------------------------------------

_RESPONSE_GENERATOR_SYSTEM_PROMPT = """
You are a sales specialist for a professional e-commerce platform.

You will receive:
  - The customer's original message
  - A lightweight product summary (names, prices, count only)

Your job:
  1. Write a warm, natural conversational response
  2. Reference products by name and price — nothing else
  3. End with ONE specific follow-up question that moves toward purchase

STRICT RULES:
- Never invent product details beyond what's in the summary
- Never mention product IDs, system names, or technical terms
- Never refuse — always offer something helpful
- Keep response under 150 words
- Only use information explicitly present in the product summary
- Do NOT infer or assume attributes (fabric, color, fit, usage, etc.)
- Return JSON only:

{
  "response_text": "...",
  "follow_up_question": "..."
}
""".strip()


async def _response_generator_node(
    state: DiscoveryAgentState,
) -> dict[str, Any]:
    """
    Generates language-only response from a lightweight product summary.

    The LLM receives:
      - Customer message
      - Product names + prices only (not full product dicts)

    The system stitches:
      - LLM response_text
      - Full raw_products from state (straight from DB)
      → AgentResult.data

    This means:
      - LLM cannot corrupt product facts (never sees them)
      - Token usage: ~300 instead of ~7000
      - Facts are always DB-accurate
    """
    request = state["request"]
    # logger.info(
    #     LogEvent.DEBUG,
    #     "RAW PRODUCTS DEBUG",
    #     sample=state["raw_products"][0] if state["raw_products"] else None
    # )
    raw_products = state.get("raw_products", [])

    # ── Build lightweight summary for LLM ─────────────────────────────────
    # LLM only needs enough to write natural language.
    # Full facts come from raw_products in state — not from LLM output.
    product_summary = _build_lightweight_summary(raw_products)

    prompt = (
        f"Customer message: {request.user_message}\n\n"
        f"{product_summary}"
    )

    logger.info(
        LogEvent.LLM_REQUEST,
        "Response generator started",
        request_id=request.request_id,
        product_count=len(raw_products),
        prompt_length=len(prompt),
    )

    response_llm = ChatOpenAI(
        model=settings.openai.chat_model,
        temperature=0.3,    # slight warmth for natural prose
        max_tokens=300,     # language only — never needs more
        timeout=settings.openai.request_timeout_seconds,
        api_key=settings.openai.api_key,
    )

    async with logger.timed(
        LogEvent.LLM_RESPONSE,
        "response_generator_llm_call",
        request_id=request.request_id,
    ):
        response = await response_llm.ainvoke([
            SystemMessage(content=_RESPONSE_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

    # ── Parse LLM language output ─────────────────────────────────────────
    response_text, follow_up = _parse_response_output(
        content=response.content,
        fallback_message=request.user_message,
        product_count=len(raw_products),
    )

    token_usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        token_usage = {
            "input_tokens": response.usage_metadata.get("input_tokens", 0),
            "output_tokens": response.usage_metadata.get("output_tokens", 0),
        }

    logger.info(
        LogEvent.LLM_RESPONSE,
        "Response generator complete",
        request_id=request.request_id,
        response_length=len(response_text),
        has_follow_up=bool(follow_up),
        **token_usage,
    )
    final_products = raw_products[: request.context.get("result_limit", 10)]

    # ── Build AgentResult ─────────────────────────────────────────────────
    # data contains:
    #   products      → full DB facts (from state, never from LLM)
    #   response_text → LLM language only
    #   follow_up     → LLM language only
    agent_result = AgentResult(
        request_id=request.request_id,
        agent_name="discovery_agent",
        success=True,
        data={
            "products": final_products,           # ← straight from DB
            "response_text": response_text,     # ← LLM language
            "follow_up_question": follow_up,    # ← LLM language
            "is_empty_result": len(final_products) == 0,
            "search_performed": state.get("search_performed", False),
        },
        confidence=1.0 if final_products else 0.5,
        metadata={
            "product_count": len(final_products),
            "tool_call_count": state.get("tool_call_count", 0),
            "prompt_version": state.get("prompt_version", "unknown"),
            "parsed_query_filters": (
                state["parsed_query"].hard_constraints.model_dump(mode="json")
                if state.get("parsed_query") else {}
            ),
        },
    )

    return {"result": agent_result}


# ---------------------------------------------------------------------------
# Routing (unchanged from your file)
# ---------------------------------------------------------------------------

def _should_continue(state: DiscoveryAgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if state.get("tool_call_count", 0) >= 5:
        logger.warning(
            LogEvent.AGENT_END,
            "Tool call limit reached — forcing to response generator",
            tool_call_count=state["tool_call_count"],
        )
        return "respond"

    if (
        last_message is not None
        and isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        return "tools"

    return "respond"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_discovery_graph() -> StateGraph:
    """
    Graph topology:
        START → entry → query_parser → llm ⟺ tools → response_generator → END

    The response_generator is the ONLY node that produces user-facing language.
    It receives a lightweight summary — never full product data.
    """
    graph = StateGraph(DiscoveryAgentState)

    graph.add_node("entry",              _entry_node)
    graph.add_node("query_parser",       _query_parser_node)
    graph.add_node("llm",                _llm_node)
    graph.add_node("tools",              _tool_execution_node)
    graph.add_node("response_generator", _response_generator_node)

    graph.add_edge(START,          "entry")
    graph.add_edge("entry",        "query_parser")
    graph.add_edge("query_parser", "llm")

    graph.add_conditional_edges(
        "llm",
        _should_continue,
        {
            "tools":   "tools",
            "respond": "response_generator",    # ← was "end" or "guardrail"
        },
    )

    graph.add_edge("tools",              "response_generator")
    graph.add_edge("response_generator", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Discovery Agent class (unchanged except _execute reads result correctly)
# ---------------------------------------------------------------------------

class DiscoveryAgent(BaseAgent):
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
        initial_state: DiscoveryAgentState = {
            "messages": [],
            "request": request,
            "result": None,
            "parsed_query": None,
            "raw_products": [],
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
            return AgentResult.failure(
                request_id=request.request_id,
                agent_name=self.agent_name,
                error_code="GRAPH_NO_RESULT",
                error_detail="Discovery agent graph completed but result was not set.",
            )

        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_discovery_agent: DiscoveryAgent | None = None


def get_discovery_agent() -> DiscoveryAgent:
    global _discovery_agent
    if _discovery_agent is None:
        _discovery_agent = DiscoveryAgent()
    return _discovery_agent


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_llm_instance: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
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


def _build_lightweight_summary(products: list[dict[str, Any]]) -> str:
    """
    Builds a minimal product summary for the response generator LLM.

    Contains ONLY what the LLM needs to write natural language:
      - Count
      - Name
      - Price

    NOT included (lives in raw_products in state):
      - product_id, images, attributes, ratings, variants, etc.

    Keeps response generator prompt under 400 tokens regardless of result size.
    """
    if not products:
        return "PRODUCTS FOUND: 0\nNo products matched the search."

    lines = [f"PRODUCTS FOUND: {len(products)}"]

    for i, p in enumerate(products[:10], 1):  # cap at 10 for summary
        name = p.get("name", "Unknown")
        price = p.get("base_price", 0)
        currency = p.get("currency", "INR")
        brand = p.get("brand", "")
        on_sale = p.get("is_on_sale", False)
        discount = p.get("discount_percent")

        line = f"{i}. {name}"
        if brand:
            line += f" by {brand}"
        line += f" — {currency} {price:,.0f}" if price else "Price unavailable"
        if on_sale and discount:
            line += f" ({discount}% off)"

        lines.append(line)

    if len(products) > 10:
        lines.append(f"... and {len(products) - 10} more products")

    return "\n".join(lines)


def _parse_response_output(
    content: str | list,
    fallback_message: str,
    product_count: int,
) -> tuple[str, str | None]:
    """
    Parses response generator LLM output → (response_text, follow_up).
    Falls back gracefully if JSON parse fails.
    """
    if isinstance(content, list):
        content = " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict)
        )

    content = str(content).strip()

    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )

    try:
        data = json.loads(content)
        return (
            data.get("response_text", ""),
            data.get("follow_up_question"),
        )
    except (json.JSONDecodeError, TypeError):
        # LLM returned plain text instead of JSON — use as-is
        logger.warning(
            LogEvent.LLM_RESPONSE,
            "Response generator returned plain text — using as response_text",
            content_preview=content[:100],
        )
        return content, None


def _build_context_block(request: AgentRequest) -> str:
    context = request.context
    parts = ["[SESSION CONTEXT]"]

    if entity_id := context.get("entity_id"):
        parts.append(f"entity_id: {entity_id}")
    if business_unit_id := context.get("business_unit_id"):
        parts.append(f"business_unit_id: {business_unit_id}")
    if last_query := context.get("last_query"):
        parts.append(f"previous_search: {last_query}")
    if shown_ids := context.get("shown_product_ids", []):
        parts.append(f"exclude_product_ids: {json.dumps(shown_ids)}")
    if result_limit := context.get("result_limit", 10):
        parts.append(f"result_limit: {result_limit}")
    if constraints := request.constraints:
        parts.append(f"business_constraints: {json.dumps(constraints)}")

    return "\n".join(parts)


def _get_last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not (
            hasattr(message, "tool_calls") and message.tool_calls
        ):
            return message
    return None


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

# All other helpers unchanged from your file:
# _build_parsed_query_from_llm, _build_parsed_query_context_message,
# _validate_constraints_against_store, _parse_llm_json_response