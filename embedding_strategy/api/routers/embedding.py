from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import threading

from embedding_strategy.db.postgres import PostgresConnectionPool, SyncWatermarkRepository
from embedding_strategy.embedding.schema import ProductCollectionManager
from embedding_strategy.embedding.sync import SyncOrchestrator, SyncRunMode, SyncRunResult
from embedding_strategy.config import settings
from embedding_strategy.observability.logger import StructuredLogger
from pymilvus import connections, utility

logger = StructuredLogger("api.routers.embedding")
router = APIRouter(prefix="/embedding", tags=["Embedding"])
_orchestrator = SyncOrchestrator()
_orchestrator_lock = threading.Lock()

# Response Schemas

class SyncTriggerResponse(BaseModel):
    """
    Returned immediately when a sync is triggered.

    Sync runs are async (BackgroundTasks) — the API
    does not block waiting for ingestion to complete.
    The DevOps cron job fires and forgets. Status is
    checked separately via GET /embedding/sync/status.
    """
    message: str
    mode: str
    status: str = "accepted"


class WatermarkInfo(BaseModel):
    last_synced_at: int | None
    total_synced: int | None
    exists: bool


class CollectionStats(BaseModel):
    collection: str
    row_count: int
    index_state: str


class SyncStatusResponse(BaseModel):
    watermark: WatermarkInfo
    collection: CollectionStats | None
    milvus_reachable: bool

class StopResponse(BaseModel):
    message: str
    was_running: bool

# Background Sync Runner

def _run_sync_in_background(mode: SyncRunMode):
    """
    Executed by FastAPI BackgroundTasks after the HTTP response
    is already returned to the caller.

    Why BackgroundTasks over Celery for now:
    - Sync is a single long-running job, not a distributed queue
    - No retry queue needed — DevOps cron handles re-scheduling
    - Zero infrastructure overhead — no Redis, no worker process
    - When scale demands it, swap this for a Celery task with
      one line change here

    Failures are logged with full structured context.
    The cron job can inspect /embedding/sync/status to verify
    the watermark was updated after the background task completes.
    """

    with _orchestrator_lock:
        if _orchestrator.is_running():
            logger.warning(
                "Sync already running — ignoring duplicate trigger",
                mode=mode.name,
            )
            return
    try:
        result: SyncRunResult = _orchestrator.run(mode=mode)
        if result.is_successful:
            logger.info(
                "Background sync completed successfully",
                mode=mode.name,
                total_upserted=result.pipeline_report.total_upserted,
                duration_seconds=result.pipeline_report.duration_seconds,
                new_watermark=result.new_watermark_timestamp,
            )
        else:
            logger.error(
                "Background sync completed with failure",
                mode=mode.name,
                error=result.error,
                watermark_saved=result.watermark_saved,
            )

    except Exception as e:
        logger.error(
            "Background sync raised unhandled exception",
            mode=mode.name,
            error=str(e),
            error_type=type(e).__name__,
        )


# Endpoints

@router.post(
    "/sync/full",
    response_model=SyncTriggerResponse,
    summary="Trigger full catalog re-embedding",
)
async def trigger_full_sync(background_tasks: BackgroundTasks):
    if _orchestrator.is_running():
        raise HTTPException(
            status_code=409,
            detail="A sync is already running. Stop it first or wait for completion.",
        )
    logger.info("Full sync trigger received")
    background_tasks.add_task(_run_sync_in_background, SyncRunMode.FULL)
    return SyncTriggerResponse(
        message="Full catalog embedding sync accepted and running in background.",
        mode="full",
    )


@router.post(
    "/sync/incremental",
    response_model=SyncTriggerResponse,
    summary="Trigger incremental delta sync",
)
async def trigger_incremental_sync(background_tasks: BackgroundTasks):
    if _orchestrator.is_running():
        raise HTTPException(
            status_code=409,
            detail="A sync is already running. Stop it first or wait for completion.",
        )
    logger.info("Incremental sync trigger received")
    background_tasks.add_task(_run_sync_in_background, SyncRunMode.INCREMENTAL)
    return SyncTriggerResponse(
        message="Incremental embedding sync accepted and running in background.",
        mode="incremental",
    )

@router.post(
    "/sync/stop",
    response_model=StopResponse,
    summary="Gracefully stop the running sync",
    description=(
        "Signals the active sync pipeline to stop after the current batch completes. "
        "Already embedded products are preserved in Milvus. "
        "Watermark is NOT saved — next run will retry from the last safe checkpoint. "
        "Returns immediately — does not wait for the pipeline to actually stop."
    ),
)
async def stop_sync():
    was_running = _orchestrator.is_running()
    _orchestrator.request_stop()

    if was_running:
        logger.info("Stop signal sent to active pipeline")
        return StopResponse(
            message=(
                "Stop signal sent. Pipeline will halt after current batch completes. "
                "Already embedded products are preserved. "
                "Watermark not saved — next sync retries from last checkpoint."
            ),
            was_running=True,
        )
    else:
        return StopResponse(
            message="No active sync running.",
            was_running=False,
        )

@router.get(
    "/sync/status",
    response_model=SyncStatusResponse,
    summary="Get current embedding sync status",
    description=(
        "Returns the last sync watermark and Milvus collection stats. "
        "DevOps uses this to verify cron runs completed successfully."
    ),
)
async def get_sync_status():
    """
    Reads watermark from PSQL and collection stats from Milvus.
    Both are read-only — no side effects.

    Milvus connectivity is checked independently —
    if Milvus is down, watermark is still returned
    with milvus_reachable: false.
    """
    # ── Watermark ────────────────────────────────────────────────────
    watermark_repo = SyncWatermarkRepository()
    watermark = watermark_repo.get_watermark()

    watermark_info = WatermarkInfo(
        last_synced_at=watermark.last_synced_at if watermark else None,
        total_synced=watermark.total_synced if watermark else None,
        exists=watermark is not None,
    )

    # ── Milvus Stats ─────────────────────────────────────────────────
    collection_stats: CollectionStats | None = None
    milvus_reachable = False

    try:
        connections.connect(
            alias="status_check",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )

        if utility.has_collection(settings.milvus_collection_name):
            manager = ProductCollectionManager()
            raw_stats = manager.get_collection_stats()
            collection_stats = CollectionStats(
                collection=raw_stats["collection"],
                row_count=raw_stats["row_count"],
                index_state=raw_stats["index_state"],
            )

        milvus_reachable = True

    except Exception as e:
        logger.warning(
            "Milvus unreachable during status check",
            error=str(e),
        )

    finally:
        try:
            connections.disconnect(alias="status_check")
        except Exception:
            pass

    return SyncStatusResponse(
        watermark=watermark_info,
        collection=collection_stats,
        milvus_reachable=milvus_reachable,
    )

@router.get("/health")
async def health():
    return {"status": "ok"}