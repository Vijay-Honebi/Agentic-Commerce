# conversational_commerce/tools/bootstrap.py

"""
Tool bootstrapper — registers ALL tools at application startup.

This is the single file that knows about every tool in the system.
Adding a new tool in any phase = add two lines here.
Nothing else changes.

Called once from main.py lifespan. Never called at request time.
"""

from __future__ import annotations

from tools.registry import ToolMetadata, _init_registry
from tools.discovery_tools.search_products import create_search_products_tool
from tools.discovery_tools.get_products_details import create_get_product_details_tool
from observability.logger import LogEvent, get_logger

logger = get_logger(__name__)


def bootstrap_tools() -> None:
    """
    Instantiates and registers all tools into the global ToolRegistry.

    Registration order does not matter — tools are looked up by name.
    Agent scope enforces which agents can call which tools.

    Phase 2: import and register cart_tools here
    Phase 3: import and register promotion_tools here
    """
    registry = _init_registry()

    # ── Phase 1: Discovery Tools ──────────────────────────────────────────

    registry.register(
        tool=create_search_products_tool(),
        metadata=ToolMetadata(
            name="search_products",
            description="Semantic + structured product search with business ranking",
            phase=1,
            agent_scope=["discovery_agent"],
            is_read_only=True,
            version="1.0.0",
            tags=["discovery", "search", "retrieval"],
        ),
    )

    registry.register(
        tool=create_get_product_details_tool(),
        metadata=ToolMetadata(
            name="get_product_details",
            description="Full product detail fetch including variants and images",
            phase=1,
            agent_scope=["discovery_agent"],   # Phase 2: add "cart_agent" here
            is_read_only=True,
            version="1.0.0",
            tags=["discovery", "detail", "variants"],
        ),
    )

    # ── Phase 2: Cart Tools (register here when implemented) ──────────────
    # registry.register(
    #     tool=create_add_to_cart_tool(),
    #     metadata=ToolMetadata(
    #         name="add_to_cart",
    #         phase=2,
    #         agent_scope=["cart_agent"],
    #         is_read_only=False,   # Makes backend commits
    #     ),
    # )

    logger.info(
        LogEvent.APP_STARTUP,
        "Tool bootstrap complete",
        total_tools=len(registry.all_tool_names),
        tools=registry.summary(),
    )