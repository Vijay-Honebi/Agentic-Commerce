from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, AsyncGenerator

from config.settings import get_settings

# ── Context variables ────────────────────────────────────────────────────────
# These propagate through async call chains automatically.
# Set once per request at the API boundary; read anywhere in the call stack.
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


class LogEvent(str, Enum):
    """
    Structured event taxonomy.
    Every log line in this system must use one of these events.
    This makes log querying deterministic — no free-text event matching.
    """
    DEBUG = "debug"
    # Lifecycle
    APP_STARTUP = "app.startup"
    APP_SHUTDOWN = "app.shutdown"

    # API layer
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"

    # Orchestrator
    ORCHESTRATOR_START = "orchestrator.start"
    ORCHESTRATOR_INTENT_CLASSIFIED = "orchestrator.intent_classified"
    ORCHESTRATOR_AGENT_DISPATCHED = "orchestrator.agent_dispatched"
    ORCHESTRATOR_RESPONSE_SYNTHESIZED = "orchestrator.response_synthesized"
    ORCHESTRATOR_END = "orchestrator.end"

    # Agents
    AGENT_START = "agent.start"
    AGENT_TOOL_CALL = "agent.tool_call"
    AGENT_TOOL_RESULT = "agent.tool_result"
    AGENT_END = "agent.end"

    # LLM
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"

    # Retrieval
    RETRIEVAL_MILVUS_START = "retrieval.milvus.start"
    RETRIEVAL_MILVUS_END = "retrieval.milvus.end"
    RETRIEVAL_PSQL_START = "retrieval.psql.start"
    RETRIEVAL_PSQL_END = "retrieval.psql.end"
    RETRIEVAL_RELAXATION_TRIGGERED = "retrieval.relaxation.triggered"
    RETRIEVAL_RELAXATION_ROUND = "retrieval.relaxation.round"
    RETRIEVAL_HYBRID_END = "retrieval.hybrid.end"

    # PSQL enrichment
    PSQL_QUERY_BUILT = "psql.query_built"
    PSQL_QUERY_EXECUTED = "psql.query_executed"

    # Ranking
    RANKER_START = "ranker.start"
    RANKER_END = "ranker.end"

    # Session
    SESSION_CREATED = "session.created"
    SESSION_LOADED = "session.loaded"
    SESSION_PERSISTED = "session.persisted"
    SESSION_EXPIRED = "session.expired"

    # Guardrails
    GUARDRAIL_PASS = "guardrail.pass"
    GUARDRAIL_VIOLATION = "guardrail.violation"

    # Performance
    SLOW_QUERY = "performance.slow_query"


class StructuredFormatter(logging.Formatter):
    """
    Emits every log record as a single-line JSON object.
    Used in staging and production environments.
    Fields are ordered for human readability in log viewers.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "event": getattr(record, "event", ""),
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": _request_id_var.get(""),
            "session_id": _session_id_var.get(""),
            "trace_id": _trace_id_var.get(""),
        }

        # Merge any extra fields attached at call site
        extra_fields = getattr(record, "extra_fields", {})
        if extra_fields:
            log_obj.update(extra_fields)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            log_obj["traceback"] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_obj, default=str)


class PrettyFormatter(logging.Formatter):
    """
    Human-readable formatter for development.
    Includes colour-coded levels for terminal output.
    """

    COLOURS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        event = getattr(record, "event", "")
        extra_fields = getattr(record, "extra_fields", {})

        parts = [
            f"{colour}[{record.levelname}]{self.RESET}",
            f"[{event}]" if event else "",
            record.getMessage(),
        ]

        if extra_fields:
            parts.append(f"| {extra_fields}")

        session_id = _session_id_var.get("")
        if session_id:
            parts.append(f"| session={session_id[:8]}...")

        return " ".join(p for p in parts if p)


class HonebiLogger:
    """
    Thin wrapper around stdlib logger that:
      1. Enforces LogEvent usage — no free-text events in production
      2. Attaches request/session/trace context automatically
      3. Provides timing helpers for performance observability
      4. Never raises — logging failures must not crash the application
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        event: LogEvent,
        message: str,
        **kwargs: Any,
    ) -> None:
        try:
            extra = {
                "event": event.value,
                "extra_fields": kwargs,
            }
            self._logger.log(level, message, extra=extra)
        except Exception:
            # Logging must never crash the application
            pass

    def debug(self, event: LogEvent, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, message, **kwargs)

    def info(self, event: LogEvent, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, message, **kwargs)

    def warning(self, event: LogEvent, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, message, **kwargs)

    def error(self, event: LogEvent, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, message, **kwargs)

    def critical(self, event: LogEvent, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, message, **kwargs)

    @asynccontextmanager
    async def timed(
        self,
        event: LogEvent,
        operation: str,
        **kwargs: Any,
    ) -> AsyncGenerator[None, None, None]:
        """
        Context manager that logs start, end, and elapsed_ms of any operation.
        Emits SLOW_QUERY warning if elapsed exceeds configured threshold.

        Usage:
            async with logger.timed(LogEvent.RETRIEVAL_MILVUS_END, "milvus_ann_search",
                                    collection="product_embeddings", top_k=50):
                results = await milvus.search(...)
        """
        settings = get_settings()
        start = time.perf_counter()
        self.debug(event, f"{operation} started", **kwargs)
        try:
            yield
        except Exception as e:
            self.error(event, f"{operation} failed", error=str(e), **kwargs)
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            self.info(
                event,
                f"{operation} completed",
                elapsed_ms=elapsed_ms,
                **kwargs,
            )
            if elapsed_ms > settings.observability.slow_query_threshold_ms:
                self.warning(
                    LogEvent.SLOW_QUERY,
                    f"{operation} exceeded slow query threshold",
                    elapsed_ms=elapsed_ms,
                    threshold_ms=settings.observability.slow_query_threshold_ms,
                    **kwargs,
                )


def configure_logging() -> None:
    """
    Configures the root logger for the entire application.
    Called once at application startup (main.py).
    After this call, every `get_logger()` in any module is correctly configured.
    """
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.observability.log_level.upper()))

    # Remove any handlers added by third-party libraries
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if settings.observability.json_logs:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(PrettyFormatter())

    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> HonebiLogger:
    """
    Factory for HonebiLogger instances.

    Usage:
        from observability.logger import get_logger, LogEvent
        logger = get_logger(__name__)
        logger.info(LogEvent.AGENT_START, "Discovery agent started", session_id=sid)
    """
    return HonebiLogger(name)


# ── Request context helpers ──────────────────────────────────────────────────
# Called at the API boundary (dependencies.py) to bind IDs to the async context.

def set_request_context(
    request_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    _request_id_var.set(request_id or str(uuid.uuid4()))
    _session_id_var.set(session_id or "")
    _trace_id_var.set(trace_id or str(uuid.uuid4()))


def get_request_id() -> str:
    return _request_id_var.get("")


def get_session_id() -> str:
    return _session_id_var.get("")


def get_trace_id() -> str:
    return _trace_id_var.get("")