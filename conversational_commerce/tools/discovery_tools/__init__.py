# conversational_commerce/tools/discovery_tools/__init__.py

from tools.discovery_tools.search_products import (
    create_search_products_tool,
    SearchProductsInput,
)
from tools.discovery_tools.get_products_details import (
    create_get_product_details_tool,
    GetProductDetailsInput,
)

__all__ = [
    "create_search_products_tool",
    "SearchProductsInput",
    "create_get_product_details_tool",
    "GetProductDetailsInput",
]
# ```

# ---

# ## What Step 4 Gave You
# ```
# ToolRegistry (singleton)
# ├── search_products          [discovery_agent only] [read-only]
# │     └── SearchProductsInput (11 typed fields, LLM-written descriptions)
# │           └── _search_products_impl()
# │                 ├── Build ParsedQuery + HardConstraints
# │                 ├── Encode query → OpenAI embedding
# │                 ├── HybridRetriever.retrieve() → candidates
# │                 └── BusinessRanker.rank() → ProductCards → dict
# │
# └── get_product_details      [discovery_agent only] [read-only]
#       └── GetProductDetailsInput (3 fields)
#             └── _get_product_details_impl()
#                   ├── fetch_product_by_id() from PSQL
#                   ├── Store-level access control check
#                   ├── _fetch_product_variants() → Phase 2 ready
#                   ├── _fetch_product_images()
#                   └── ProductDetail → dict

# bootstrap_tools()            ← called once at startup
# make_instrumented_tool()     ← wraps every tool with logging + timing + error capture