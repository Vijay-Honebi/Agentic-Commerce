from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from embedding_strategy.api.routers import embedding
from embedding_strategy.config.settings import get_settings
from embedding_strategy.db.postgres import PostgresConnectionPool
from embedding_strategy.observability.logger import StructuredLogger
from conversational_commerce.api.v1.search_endpoints import router as serach_router
from dotenv import load_dotenv
load_dotenv()

logger = StructuredLogger("server.app")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle.

    Startup  : Initialize Postgres connection pool once.
               Pool is reused across all requests and
               background tasks for the process lifetime.

    Shutdown : Drain and close all Postgres connections cleanly.
               Prevents connection leaks on pod restarts.
    """
    logger.info("Honebi API starting up")
    PostgresConnectionPool.initialize()
    logger.info("Postgres connection pool ready")

    yield

    logger.info("Honebi API shutting down")
    PostgresConnectionPool.close_all()
    logger.info("Postgres connection pool closed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Honebi Agentic Commerce API",
        description=(
            "Backend AI infrastructure for Honebi agentic commerce. "
            "Phase 1: Product embedding and AI-powered discovery."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(embedding.router)
    app.include_router(serach_router, prefix="/api/v1")

    logger.info("FastAPI app created", routes=[r.path for r in app.routes])

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=os.getenv("HOST"), port=int(os.getenv("PORT", 8000)), reload=True)


# # Install
# pip install fastapi uvicorn pydantic-settings

# # Dev
# uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# # Production
# uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4