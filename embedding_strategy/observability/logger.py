import logging
import json
import time
from typing import Any
from functools import wraps

class StructuredLogger:
    """
    Emits structured JSON logs for every pipeline stage.
    Each log line is a complete, parseable JSON object —
    ready for ingestion into Datadog, CloudWatch, or ELK.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self._JsonFormatter())
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
        self.component = name

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "component": record.name,
                "message": record.getMessage(),
            }
            if hasattr(record, "extra"):
                payload.update(record.extra)
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    def _log(self, level: str, message: str, **kwargs):
        extra = {"extra": kwargs} if kwargs else {}
        getattr(self._logger, level)(message, extra=extra)

    def info(self, message: str, **kwargs):
        self._log("info", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("debug", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log("critical", message, **kwargs)


def trace_stage(stage_name: str):
    """
    Decorator that wraps any pipeline function with:
    - Entry/exit structured logs
    - Wall-clock timing
    - Exception capture with context
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            logger = StructuredLogger(f"trace.{stage_name}")
            start = time.perf_counter()
            logger.info(f"{stage_name} started")
            try:
                result = fn(*args, **kwargs)
                elapsed = round((time.perf_counter() - start) * 1000, 2)
                logger.info(f"{stage_name} completed", duration_ms=elapsed)
                return result
            except Exception as e:
                elapsed = round((time.perf_counter() - start) * 1000, 2)
                logger.error(
                    f"{stage_name} failed",
                    duration_ms=elapsed,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
        return wrapper
    return decorator