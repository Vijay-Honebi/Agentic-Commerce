# conversational_commerce/guardrails/__init__.py

from guardrails.discovery_guard import (
    DiscoveryGuardrail,
    DiscoveryGuardrailResult,
    DiscoveryViolationType,
    GuardrailViolation,
    ViolationSeverity,
    get_discovery_guardrail,
)
from guardrails.orchestrator_guard import (
    OrchestratorGuardrail,
    OrchestratorGuardrailResult,
    OrchestratorViolationType,
    OrchestratorViolation,
    PromotionLimits,
    get_orchestrator_guardrail,
)

__all__ = [
    # Discovery
    "DiscoveryGuardrail",
    "DiscoveryGuardrailResult",
    "DiscoveryViolationType",
    "GuardrailViolation",
    "ViolationSeverity",
    "get_discovery_guardrail",
    # Orchestrator
    "OrchestratorGuardrail",
    "OrchestratorGuardrailResult",
    "OrchestratorViolationType",
    "OrchestratorViolation",
    "PromotionLimits",
    "get_orchestrator_guardrail",
]
# ```

# ---

# ## What Step 5 Gave You — Complete Guardrail Stack
# ```
# Agent Output (AgentResult.data)
#          │
#          ▼
# ┌─────────────────────────────────────────┐
# │         DISCOVERY GUARDRAIL             │
# │                                         │
# │  ✓ Confidence threshold check           │
# │  ✓ Product ID integrity (no hallucination) │
# │  ✓ Store scope / tenant isolation       │
# │  ✓ Discount computation verification   │
# │  ✓ Stock status validation              │
# │  ✓ Result limit enforcement             │
# │  ✓ Empty result flagging                │
# │                                         │
# │  CRITICAL → Drop product from results   │
# │  HIGH     → Sanitise in-place           │
# │  LOW      → Log only                    │
# └────────────────┬────────────────────────┘
#                  │ sanitised product list
#                  ▼
#          Response Synthesizer
#                  │ natural language response
#                  ▼
# ┌─────────────────────────────────────────┐
# │        ORCHESTRATOR GUARDRAIL           │
# │                                         │
# │  ✓ Internal system ID leak detection   │
# │  ✓ PII pattern detection               │
# │  ✓ Illegal promotion language          │
# │  ✓ Discount % validation against data  │
# │  ✓ Merchant maximum enforcement        │
# │  ✓ Refusal language detection          │
# │  ✓ Response length sanity              │
# │                                         │
# │  CRITICAL → Block, return safe fallback │
# │  HIGH     → Sanitise text in-place     │
# │  MEDIUM   → Log warning, pass through  │
# └────────────────┬────────────────────────┘
#                  │ final clean response
#                  ▼
#               User