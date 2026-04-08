# conversational_commerce/tools/registry.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool

from observability.logger import LogEvent, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

@dataclass
class ToolMetadata:
    """
    Metadata attached to every registered tool.

    Agents use `phase` and `agent_scope` to determine which tools
    are available to them — preventing, e.g., the Discovery Agent
    from accidentally calling a cart tool.
    """

    name: str
    description: str
    phase: int                          # Which phase introduced this tool
    agent_scope: list[str]              # Which agents can use this tool
    is_read_only: bool = True           # False = makes backend commits (Phase 2+)
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool registry — central catalogue of all tools in the system
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Central registry for all LangGraph tools across all phases.

    Design rules:
      1. Tools are registered once at startup — never at request time
      2. Each agent requests only its scoped tools via get_tools_for_agent()
         This enforces the architectural boundary:
         Discovery Agent can never call a cart tool
      3. Every tool call is intercepted and logged — no silent failures
      4. Registry is a singleton — one instance for the entire process

    Phase 2/3/4 tools are registered here as they're implemented.
    The registry is append-only — registered tools are never removed.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, ToolMetadata] = {}

    def register(
        self,
        tool: BaseTool,
        metadata: ToolMetadata,
    ) -> None:
        """
        Registers a tool with its metadata.
        Called at application startup for every tool in the system.

        Raises:
            ValueError: If a tool with the same name is already registered.
                        Tools must have unique names across all phases.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Tool names must be unique across all phases."
            )

        self._tools[tool.name] = tool
        self._metadata[tool.name] = metadata

        logger.info(
            LogEvent.APP_STARTUP,
            "Tool registered",
            tool_name=tool.name,
            phase=metadata.phase,
            agent_scope=metadata.agent_scope,
            is_read_only=metadata.is_read_only,
            version=metadata.version,
        )

    def get_tools_for_agent(self, agent_name: str) -> list[BaseTool]:
        """
        Returns the tools available to a specific agent.
        Agents ONLY receive tools scoped to them — enforces SoC.

        Args:
            agent_name: e.g. "discovery_agent", "cart_agent"

        Returns:
            List of BaseTool instances the agent may call.
        """
        available = [
            self._tools[name]
            for name, meta in self._metadata.items()
            if agent_name in meta.agent_scope
        ]

        logger.debug(
            LogEvent.AGENT_START,
            "Tools resolved for agent",
            agent_name=agent_name,
            tool_names=[t.name for t in available],
            tool_count=len(available),
        )

        return available

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """Direct tool lookup by name. Returns None if not found."""
        return self._tools.get(tool_name)

    def get_metadata(self, tool_name: str) -> ToolMetadata | None:
        """Returns tool metadata. Used by observability and guardrails."""
        return self._metadata.get(tool_name)

    @property
    def all_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def summary(self) -> list[dict[str, Any]]:
        """Returns a summary of all registered tools for logging at startup."""
        return [
            {
                "name": meta.name,
                "phase": meta.phase,
                "agent_scope": meta.agent_scope,
                "is_read_only": meta.is_read_only,
                "version": meta.version,
            }
            for meta in self._metadata.values()
        ]


# ---------------------------------------------------------------------------
# Instrumented tool wrapper
# ---------------------------------------------------------------------------

def make_instrumented_tool(
    func: Callable,
    name: str,
    description: str,
    args_schema: Any,
) -> StructuredTool:
    """
    Wraps any async tool function with:
      - Structured logging on every call (entry + exit)
      - Execution timing
      - Error capture with full context
      - Consistent return shape on failure (never raises to LangGraph)

    All discovery tools are created via this factory.
    Phase 2/3 tools will use the same factory — observability is free.

    Args:
        func:         The async tool implementation function.
        name:         Tool name (must match ToolMetadata.name).
        description:  LangGraph uses this to decide when to call the tool.
                      Write it for the LLM, not for engineers.
        args_schema:  Pydantic model defining the tool's input schema.

    Returns:
        StructuredTool ready for LangGraph ToolNode.
    """

    async def instrumented(*args: Any, **kwargs: Any) -> Any:
        logger.info(
            LogEvent.AGENT_TOOL_CALL,
            f"Tool called: {name}",
            tool_name=name,
            kwargs_keys=list(kwargs.keys()),
        )

        try:
            async with logger.timed(
                LogEvent.AGENT_TOOL_RESULT,
                f"tool_execution_{name}",
                tool_name=name,
            ):
                result = await func(*args, **kwargs)

            logger.info(
                LogEvent.AGENT_TOOL_RESULT,
                f"Tool succeeded: {name}",
                tool_name=name,
                result_type=type(result).__name__,
            )
            return result

        except Exception as e:
            logger.error(
                LogEvent.AGENT_TOOL_RESULT,
                f"Tool failed: {name}",
                tool_name=name,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return structured error — never raise to LangGraph
            # LangGraph treats raised exceptions as graph failures
            return {
                "success": False,
                "error_code": "TOOL_EXECUTION_ERROR",
                "error_detail": str(e),
                "tool_name": name,
            }

    return StructuredTool(
        name=name,
        description=description,
        args_schema=args_schema,
        coroutine=instrumented,
    )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """
    Returns the global ToolRegistry singleton.
    Initialised by bootstrap_tools() at startup.
    """
    if _registry is None:
        raise RuntimeError(
            "ToolRegistry not initialised. "
            "Call bootstrap_tools() in the application lifespan."
        )
    return _registry


def _init_registry() -> ToolRegistry:
    """Creates the singleton. Called only by bootstrap_tools()."""
    global _registry
    _registry = ToolRegistry()
    return _registry