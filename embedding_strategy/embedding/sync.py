from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto

from embedding_strategy.config.settings import get_settings
from embedding_strategy.db.postgres import SyncWatermarkRepository, SyncWatermark
from embedding_strategy.embedding.pipeline import ProductEmbeddingPipeline, PipelineRunReport
from embedding_strategy.embedding.repository import ProductVectorRepository
from embedding_strategy.embedding.schema import ProductCollectionManager
from embedding_strategy.observability.logger import StructuredLogger, trace_stage
from pymilvus import connections

logger = StructuredLogger("embedding.sync")
settings = get_settings()


# Run Mode

class SyncRunMode(Enum):
    """
    Determines which pipeline entry point is invoked.

    FULL        : Embed entire catalog. No watermark dependency.
                  Used for: first run, forced re-index, schema migration.

    INCREMENTAL : Embed only products updated since last watermark.
                  Used for: daily scheduled sync.

    AUTO        : Inspect watermark — if none exists, run FULL.
                  If watermark exists, run INCREMENTAL.
                  This is the default mode for the daily cron job.
                  Zero manual intervention required.
    """
    FULL = auto()
    INCREMENTAL = auto()
    AUTO = auto()


# Sync Decision

@dataclass(frozen=True)
class SyncDecision:
    """
    The resolved decision made by SyncOrchestrator before execution.

    Logged before pipeline starts — gives full audit trail of
    why a particular run mode was chosen.

    resolved_mode   : The actual mode that will execute (never AUTO).
    since_timestamp : Unix epoch used for incremental fetch.
                      None if resolved_mode is FULL.
    watermark       : The loaded watermark, if any.
    reason          : Human-readable explanation for the decision.
    """
    resolved_mode: SyncRunMode
    since_timestamp: int | None
    watermark: SyncWatermark | None
    reason: str


# Sync Run Result

@dataclass(frozen=True)
class SyncRunResult:
    """
    Complete result of a sync orchestration run.

    Wraps PipelineRunReport with sync-level metadata:
    - What decision was made and why
    - Whether the watermark was saved
    - Whether the run is considered successful

    is_successful:
        True  → pipeline completed + watermark saved
        False → pipeline failed OR watermark save failed

    A run where embedding_failures > 0 is still considered
    successful if the overall rejection rate is within threshold.
    Partial success is expected and acceptable for large catalogs.
    """
    decision: SyncDecision
    pipeline_report: PipelineRunReport | None
    watermark_saved: bool
    new_watermark_timestamp: int | None
    error: str | None

    @property
    def is_successful(self) -> bool:
        return (
            self.pipeline_report is not None
            and self.watermark_saved
            and self.error is None
        )

    def to_log_dict(self) -> dict:
        base = {
            "resolved_mode": self.decision.resolved_mode.name,
            "decision_reason": self.decision.reason,
            "watermark_saved": self.watermark_saved,
            "new_watermark_timestamp": self.new_watermark_timestamp,
            "is_successful": self.is_successful,
            "error": self.error,
        }
        if self.pipeline_report:
            base.update(self.pipeline_report.to_log_dict())
        return base


# Milvus Connection Manager

class MilvusConnectionManager:
    """
    Manages Milvus connection lifecycle for the sync process.

    Isolated here so:
    - Pipeline and repository never manage connections
    - Connection setup/teardown is always symmetric
    - Easy to swap connection strategy (e.g. TLS, auth) in one place

    Why not a context manager on every operation:
    - Milvus connections are expensive to establish
    - One connection per sync run is the correct granularity
    - Connection is shared across collection manager + repository
    """

    @staticmethod
    def connect():
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        logger.info(
            "Milvus connected",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )

    @staticmethod
    def disconnect():
        connections.disconnect(alias="default")
        logger.info("Milvus disconnected")


# Sync Orchestrator

class SyncOrchestrator:
    """
    Top-level orchestrator for the product embedding sync process.

    Responsibilities (and nothing more):
    1. Establish Milvus connection
    2. Ensure collection exists (idempotent)
    3. Decide run mode (full / incremental)
    4. Execute pipeline
    5. Save watermark — ONLY on confirmed success
    6. Disconnect Milvus
    7. Return SyncRunResult

    This is the only class that knows about:
    - Milvus connection lifecycle
    - Watermark read/write
    - Run mode decision logic

    Everything else (fetching, embedding, upserting) is
    delegated to ProductEmbeddingPipeline.

    Calling convention:
        orchestrator = SyncOrchestrator()
        result = orchestrator.run(mode=SyncRunMode.AUTO)
    """

    def __init__(self):
        self._watermark_repo = SyncWatermarkRepository()
        self._collection_manager = ProductCollectionManager()
        self._active_pipeline: ProductEmbeddingPipeline | None = None
        self._lock = threading.Lock()

    def request_stop(self):
        """
        Requests graceful stop of the currently running pipeline.
        Safe to call even if no pipeline is running — no-op in that case.
        """
        with self._lock:
            if self._active_pipeline is None:
                logger.info("Stop requested but no active pipeline running")
                return
            self._active_pipeline.request_stop()

    def is_running(self) -> bool:
        with self._lock:
            return self._active_pipeline is not None

    @trace_stage("sync_orchestrator_run")
    def run(self, mode: SyncRunMode = SyncRunMode.AUTO) -> SyncRunResult:
        """
        Execute a full sync orchestration run.

        Steps:
        1. Connect to Milvus
        2. Ensure collection + index exist
        3. Resolve run mode decision
        4. Execute pipeline
        5. Save watermark on success
        6. Disconnect
        7. Return result

        Milvus is always disconnected in the finally block —
        even if an unhandled exception escapes the pipeline.
        """
        MilvusConnectionManager.connect()

        try:
            collection = self._collection_manager.ensure_collection()
            repository = ProductVectorRepository(collection)
            pipeline = ProductEmbeddingPipeline(repository)

            with self._lock:
                self._active_pipeline = pipeline

            decision = self._resolve_decision(mode)

            logger.info(
                "Sync decision resolved",
                resolved_mode=decision.resolved_mode.name,
                since_timestamp=decision.since_timestamp,
                reason=decision.reason,
            )

            return self._execute(pipeline, decision)

        finally:
            with self._lock:
                self._active_pipeline = None
            MilvusConnectionManager.disconnect()

    # Decision Resolution

    def _resolve_decision(self, mode: SyncRunMode) -> SyncDecision:
        """
        Resolves the actual run mode from the requested mode.

        AUTO resolution logic:
        - No watermark found → FULL
          Reason: First run. No checkpoint to resume from.

        - Watermark found → INCREMENTAL with lookback overlap
          Reason: Daily delta. Lookback guards against clock skew.

        FULL and INCREMENTAL bypass watermark entirely:
        - FULL: Always re-embed everything regardless of state
        - INCREMENTAL: Uses watermark but caller explicitly requested delta
          If no watermark exists for explicit INCREMENTAL, we fallback
          to FULL with a warning — never fail silently.
        """
        if mode == SyncRunMode.FULL:
            return SyncDecision(
                resolved_mode=SyncRunMode.FULL,
                since_timestamp=None,
                watermark=None,
                reason="Explicitly requested full ingestion.",
            )

        watermark = self._watermark_repo.get_watermark()

        if mode == SyncRunMode.INCREMENTAL:
            if watermark is None:
                logger.warning(
                    "Incremental sync requested but no watermark found. "
                    "Falling back to full ingestion.",
                )
                return SyncDecision(
                    resolved_mode=SyncRunMode.FULL,
                    since_timestamp=None,
                    watermark=None,
                    reason=(
                        "Incremental requested but no watermark exists. "
                        "Fallback to full ingestion."
                    ),
                )

            since = self._apply_lookback(watermark.last_synced_at)
            return SyncDecision(
                resolved_mode=SyncRunMode.INCREMENTAL,
                since_timestamp=since,
                watermark=watermark,
                reason=(
                    f"Explicit incremental. "
                    f"Watermark: {watermark.last_synced_at}. "
                    f"With {settings.incremental_sync_lookback_hours}h "
                    f"lookback → since: {since}."
                ),
            )

        # AUTO mode
        if watermark is None:
            return SyncDecision(
                resolved_mode=SyncRunMode.FULL,
                since_timestamp=None,
                watermark=None,
                reason="AUTO: No watermark found. Running full ingestion.",
            )

        since = self._apply_lookback(watermark.last_synced_at)
        return SyncDecision(
            resolved_mode=SyncRunMode.INCREMENTAL,
            since_timestamp=since,
            watermark=watermark,
            reason=(
                f"AUTO: Watermark found at {watermark.last_synced_at}. "
                f"Running incremental with "
                f"{settings.incremental_sync_lookback_hours}h "
                f"lookback → since: {since}."
            ),
        )

    @staticmethod
    def _apply_lookback(last_synced_at: int) -> int:
        """
        Subtracts the configured lookback window from the watermark.

        Why lookback overlap:
        - Clock skew between app servers and PSQL can be 1-5 seconds
        - Distributed writes near the watermark boundary may not be
          visible immediately due to PSQL MVCC snapshot isolation
        - A 25-hour lookback (default) on a daily sync guarantees
          no product updated in the last day is ever missed
        - Re-processing already-indexed products is harmless —
          Milvus upsert is idempotent on primary key

        The slight cost of re-embedding a few extra products on each
        run is vastly preferable to silently missing updates.
        """
        lookback_seconds = settings.incremental_sync_lookback_hours * 3600
        since = last_synced_at - lookback_seconds

        logger.debug(
            "Lookback applied",
            last_synced_at=last_synced_at,
            lookback_hours=settings.incremental_sync_lookback_hours,
            lookback_seconds=lookback_seconds,
            effective_since=since,
        )

        return since

    # Execution

    def _execute(
        self,
        pipeline: ProductEmbeddingPipeline,
        decision: SyncDecision,
    ) -> SyncRunResult:
        """
        Executes the pipeline based on the resolved decision.
        Saves watermark only if pipeline completes without raising.

        Watermark timestamp is set to run start time — not end time.

        Why run start time:
        - Products updated DURING the run (between start and end)
          will have updated_at > run_start_timestamp
        - Next incremental sync will catch them via lookback
        - If we used end_time, products updated during a long run
          could be missed in the next incremental window
        """
        run_start_timestamp = int(time.time())
        pipeline_report: PipelineRunReport | None = None
        error: str | None = None
        watermark_saved = False
        new_watermark_timestamp: int | None = None

        try:
            if decision.resolved_mode == SyncRunMode.FULL:
                pipeline_report = pipeline.run_full_ingestion()
            else:
                pipeline_report = pipeline.run_incremental_sync(
                    since_timestamp=decision.since_timestamp
                )

            # Save watermark ONLY after confirmed success
            new_watermark_timestamp = run_start_timestamp
            self._watermark_repo.save_watermark(
                last_synced_at=new_watermark_timestamp,
                total_synced=pipeline_report.total_upserted,
            )
            watermark_saved = True

            logger.info(
                "Watermark saved after successful run",
                new_watermark_timestamp=new_watermark_timestamp,
                total_upserted=pipeline_report.total_upserted,
            )

        except Exception as e:
            error = str(e)
            logger.error(
                "Sync run failed — watermark NOT saved",
                error=error,
                error_type=type(e).__name__,
                resolved_mode=decision.resolved_mode.name,
                # Watermark is intentionally not saved.
                # Next run will re-attempt from the previous safe checkpoint.
            )

        result = SyncRunResult(
            decision=decision,
            pipeline_report=pipeline_report,
            watermark_saved=watermark_saved,
            new_watermark_timestamp=new_watermark_timestamp,
            error=error,
        )

        logger.info("Sync orchestration complete", **result.to_log_dict())

        return result