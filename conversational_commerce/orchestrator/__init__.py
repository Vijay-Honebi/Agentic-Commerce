# conversational_commerce/orchestrator/__init__.py

from orchestrator.orchestrator_agent import OrchestratorAgent, get_orchestrator
from orchestrator.router import IntentRouter, get_intent_router
from orchestrator.context_builder import ContextBuilder, get_context_builder
from orchestrator.response_synthesizer import (
    ResponseSynthesizer,
    get_response_synthesizer,
)

__all__ = [
    "OrchestratorAgent", "get_orchestrator",
    "IntentRouter", "get_intent_router",
    "ContextBuilder", "get_context_builder",
    "ResponseSynthesizer", "get_response_synthesizer",
]
# ```

# ---

## What Step 7 Gave You — The Complete Brain
# ```
# User Message
#       │
#       ▼
# OrchestratorAgent.process()
#       │
#       ▼
# ┌─────────────────────────────────────────────────────────┐
# │                  MASTER LANGGRAPH                        │
# │                                                          │
# │  load_session ──► intent_router                         │
# │                        │                                 │
# │          ┌─────────────┼──────────────────┐              │
# │          ▼             ▼                  ▼              │
# │       discovery      cart/checkout    clarification      │
# │          │             │                  │              │
# │          └─────────────┴──────────────────┘              │
# │                        │                                 │
# │                   synthesize                             │
# │                        │                                 │
# │                  persist_session ──► END                 │
# └─────────────────────────────────────────────────────────┘
#       │
#       ▼
# {response_text, session_id, intent, latency_ms}