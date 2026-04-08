# conversational_commerce/agents/__init__.py

from agents.base_agent import BaseAgent
from agents.discovery_agent import DiscoveryAgent, get_discovery_agent
from agents.cart_agent import CartAgent
from agents.checkout_agent import CheckoutAgent
from agents.promotion_agent import PromotionAgent

__all__ = [
    "BaseAgent",
    "DiscoveryAgent",
    "get_discovery_agent",
    "CartAgent",
    "CheckoutAgent",
    "PromotionAgent",
]
# ```

# ---

# ## What Step 6 Gave You — Full Agent Stack
# ```
# BaseAgent (abstract)
#     └── run() → wraps _execute() with logging + timing + exception guard
#     └── _build_success() / _build_failure() helpers

# DiscoveryAgent (LangGraph StateGraph)
#     │
#     ├── Graph: START → entry → llm ⟺ tools → guardrail → END
#     │
#     ├── entry_node      builds SystemMessage + HumanMessage with context block
#     ├── llm_node        gpt-4.1-mini with bound tools, tracks token usage
#     ├── tool_exec_node  ToolNode execution, extracts retrieval_product_ids
#     ├── guardrail_node  DiscoveryGuardrail validation → AgentResult
#     │
#     └── _should_continue  routes: tool_calls → tools | no tool_calls → guardrail
#                           hard cap: 5 tool calls max (prevents infinite loops)

# CartAgent       → Phase 2 stub (correct structure, NOT_IMPLEMENTED error)
# CheckoutAgent   → Phase 2 stub
# PromotionAgent  → Phase 3 stub