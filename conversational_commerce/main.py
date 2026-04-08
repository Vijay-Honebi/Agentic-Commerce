# conversational_commerce/main.py

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.v1.chat import router as chat_router
from config.settings import get_settings
from memory.session_store import close_session_store, init_session_store
from observability.logger import LogEvent, configure_logging, get_logger
from retrieval.milvus_client import connect_milvus, disconnect_milvus
from retrieval.psql_client import close_psql_retrieval_pool, init_psql_retrieval_pool, _get_pool as get_retrieval_pool
from tools.bootstrap import bootstrap_tools
from services.attribute_store import get_attribute_store
from memory.migration_sql import create_if_not_exists
# from embedding_strategy.db.postgres import PostgresConnectionPool as Pool


# Configure logging FIRST — before any other imports use the logger
configure_logging()

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan handler.
    Manages startup and shutdown of ALL system components.

    Startup order matters — dependencies must initialise before dependents:
        1. Logging          (everything uses this)
        2. PostgreSQL pools (session store + retrieval)
        3. Milvus           (retrieval depends on this)
        4. Tool registry    (agents depend on this)
        5. Orchestrator     (depends on everything above)

    Shutdown order is reverse startup — dependents close before dependencies.
    """

    # ── STARTUP ───────────────────────────────────────────────────────────
    logger.info(
        LogEvent.APP_STARTUP,
        "Honebi Conversational Commerce starting up",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value,
    )

    # PostgreSQL — session store pool
    logger.info(LogEvent.APP_STARTUP, "Initialising session store...")
    await init_session_store()

    # PostgreSQL — retrieval pool (separate pool, separate sizing)
    logger.info(LogEvent.APP_STARTUP, "Initialising retrieval pool...")
    await init_psql_retrieval_pool()

    logger.info(LogEvent.APP_STARTUP, "Loading attribute store...")
    attribute_store = get_attribute_store()
    await attribute_store.load(await get_retrieval_pool())

    # Milvus — vector search
    logger.info(LogEvent.APP_STARTUP, "Connecting to Milvus...")
    connect_milvus()

    # Tool registry — register all Phase 1 tools
    logger.info(LogEvent.APP_STARTUP, "Bootstrapping tool registry...")
    bootstrap_tools()

    # create or replace agent_sessions table with new schema
    await create_if_not_exists()
    logger.info(LogEvent.APP_STARTUP, "Database migrations complete")

    # Orchestrator + Discovery Agent — compile LangGraph graphs
    # Importing triggers singleton instantiation and graph compilation.
    # This is intentional — we want compilation cost at startup, not first request.
    logger.info(LogEvent.APP_STARTUP, "Compiling agent graphs...")
    from orchestrator.orchestrator_agent import get_orchestrator
    from agents.discovery_agent import get_discovery_agent
    get_discovery_agent()   # Compiles Discovery Agent graph
    get_orchestrator()      # Compiles Orchestrator graph

    logger.info(
        LogEvent.APP_STARTUP,
        "All systems ready — Honebi Conversational Commerce is live",
        app_name=settings.app_name,
        version=settings.app_version,
    )

    # ── Application runs here ─────────────────────────────────────────────
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info(
        LogEvent.APP_SHUTDOWN,
        "Honebi Conversational Commerce shutting down",
    )

    # Reverse order: dependents first, then dependencies
    disconnect_milvus()
    await close_psql_retrieval_pool()
    await close_session_store()

    logger.info(LogEvent.APP_SHUTDOWN, "Clean shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Application factory.
    Returns a configured FastAPI instance.

    Using a factory (not a module-level app) makes testing cleaner —
    tests can call create_app() with overridden settings.
    """
    app = FastAPI(
        title="Honebi Conversational Commerce API",
        description=(
            "AI-powered agentic commerce engine. "
            "Phase 1: Product Discovery. "
            "Phase 2: Cart & Checkout (coming soon). "
            "Phase 3: Promotions & Recommendations (coming soon)."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
    )

    # ── Middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "X-Store-ID",
            "X-Session-ID",
            "X-Request-ID",
            "X-Trace-ID",
        ],
    )

    # ── Exception handlers ────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """
        Catches any unhandled exception that escapes the endpoint handler.
        Returns a clean JSON error — never exposes stack traces to clients.
        """
        logger.error(
            LogEvent.API_ERROR,
            "Unhandled exception at application level",
            error=str(exc),
            error_type=type(exc).__name__,
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again.",
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(chat_router, prefix="/api/v1")

    return app


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.observability.log_level.lower(),
        access_log=False,       # We handle request logging ourselves
    )