from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Generator

from embedding_strategy.config.settings import get_settings
from embedding_strategy.db.postgres import GlobalProductFetcher
from embedding_strategy.embedding.client import EmbeddingClient
from embedding_strategy.embedding.composer import EmbeddingComposer, GlobalProduct
from embedding_strategy.embedding.repository import (
    ProductVector,
    ProductVectorRepository,
    UpsertResult,
)
from embedding_strategy.observability.logger import StructuredLogger, trace_stage

logger = StructuredLogger("embedding.pipeline")
settings = get_settings()


# Pipeline Run Report

@dataclass
class PipelineRunReport:
    """
    Complete observability record for a single pipeline run.

    Emitted at the end of every run — full ingestion or incremental.
    Structured for direct ingestion into Datadog / CloudWatch / ELK.

    Alert thresholds (recommended):
    - overall_rejection_rate > 0.05  → systematic data quality issue
    - embedding_failures > 0         → OpenAI API degradation
    - duration_seconds > 3600        → ingestion taking too long
    """
    run_type: str                           # "full" | "incremental"
    started_at: int                         # Unix timestamp
    completed_at: int = 0
    duration_seconds: float = 0.0

    total_fetched: int = 0
    total_embedded: int = 0
    total_upserted: int = 0
    total_rejected: int = 0
    total_skipped: int = 0                  # composed to empty string

    embedding_batches: int = 0
    embedding_failures: int = 0

    psql_batches: int = 0
    milvus_upsert_batches: int = 0

    rejected_ids: list[str] = field(default_factory=list)

    @property
    def overall_rejection_rate(self) -> float:
        if self.total_fetched == 0:
            return 0.0
        return round(self.total_rejected / self.total_fetched, 4)

    @property
    def success_rate(self) -> float:
        if self.total_fetched == 0:
            return 0.0
        return round(self.total_upserted / self.total_fetched, 4)

    def finalize(self):
        self.completed_at = int(time.time())
        self.duration_seconds = round(
            self.completed_at - self.started_at, 2
        )

    def to_log_dict(self) -> dict:
        return {
            "run_type": self.run_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "total_fetched": self.total_fetched,
            "total_embedded": self.total_embedded,
            "total_upserted": self.total_upserted,
            "total_rejected": self.total_rejected,
            "total_skipped": self.total_skipped,
            "embedding_batches": self.embedding_batches,
            "embedding_failures": self.embedding_failures,
            "psql_batches": self.psql_batches,
            "milvus_upsert_batches": self.milvus_upsert_batches,
            "overall_rejection_rate": self.overall_rejection_rate,
            "success_rate": self.success_rate,
        }


# Sub-Batch Splitter

class BatchSplitter:
    """
    Splits a PSQL batch (size 500) into OpenAI embedding
    sub-batches (size 100).

    Why two different batch sizes:
    - PSQL batch (500): Larger batches reduce round trips to DB.
      500 rows per query is efficient for network + DB cursor overhead.

    - OpenAI batch (100): OpenAI recommends ≤ 100 texts per embedding
      call for optimal throughput. Beyond 100, token limits per request
      can be hit on long product descriptions.

    This splitter is the only place that knows about this boundary.
    Pipeline and fetcher remain decoupled from each other's batch sizes.
    """

    @staticmethod
    def split(
        items: list, sub_batch_size: int
    ) -> Generator[list, None, None]:
        for i in range(0, len(items), sub_batch_size):
            yield items[i: i + sub_batch_size]


# Product Vector Assembler

class ProductVectorAssembler:
    """
    Assembles ProductVector instances from GlobalProduct records
    and their corresponding embedding vectors.

    Responsibilities:
    - Pair each product with its embedding by position
    - Build the category_path scalar field
    - Produce the final ProductVector write contract

    This is deliberately separated from EmbeddingComposer:
    - Composer owns: text composition (what to embed)
    - Assembler owns: vector assembly (how to pair and package)
    """

    def __init__(self):
        self._composer = EmbeddingComposer()

    def assemble(
        self,
        products: list[GlobalProduct],
        embeddings: list[list[float]],
    ) -> list[ProductVector]:
        """
        Pairs products with embeddings positionally.
        len(products) must equal len(embeddings) — caller guarantees this.
        """
        assert len(products) == len(embeddings), (
            f"Product count ({len(products)}) != "
            f"embedding count ({len(embeddings)}). "
            f"Positional pairing would produce corrupted vectors."
        )

        vectors = []
        for product, embedding in zip(products, embeddings):
            category_path = self._composer.build_category_path_for_filter(product)
            vectors.append(
                ProductVector(
                    global_product_id=product.global_product_id,
                    embedding=embedding,
                    category_path=category_path,
                    bu_ids=product.bu_ids,
                    updated_at=product.updated_at,
                )
            )

        return vectors


# Core Pipeline

class ProductEmbeddingPipeline:
    """
    Orchestrates the full product embedding ingestion pipeline.

    Execution flow per PSQL batch:

        PSQL batch (500 products)
            │
            ├─► Filter empty compositions (skip, log)
            │
            └─► Split into OpenAI sub-batches (100 each)
                    │
                    ├─► Compose embedding text per product
                    ├─► Call OpenAI text-embedding-3-small
                    ├─► Assemble ProductVector list
                    └─► Upsert into Milvus
                            │
                            └─► Accumulate into PipelineRunReport

    Two public entry points:
    - run_full_ingestion()      : Full catalog. First run or forced re-index.
    - run_incremental_sync()    : Delta since last watermark. Daily cron.

    Both return a PipelineRunReport for observability.
    """

    def __init__(self, repository: ProductVectorRepository):
        self._repository = repository
        self._fetcher = GlobalProductFetcher()
        self._composer = EmbeddingComposer()
        self._embedding_client = EmbeddingClient()
        self._assembler = ProductVectorAssembler()
        self._splitter = BatchSplitter()
        self._stop_event = threading.Event()

    # Public Entry Points

    @trace_stage("pipeline_full_ingestion")
    def run_full_ingestion(self) -> PipelineRunReport:
        """
        Embeds and upserts the entire global products catalog.

        When to use:
        - First time setup
        - Schema migration requiring full re-index
        - Data quality remediation requiring full re-embed

        This is idempotent — safe to re-run. Milvus upsert
        on existing primary keys updates in place.
        """
        logger.info("Full ingestion started")

        report = PipelineRunReport(
            run_type="full",
            started_at=int(time.time()),
        )

        batch_generator = self._fetcher.fetch_all_batched()
        self._execute_pipeline(batch_generator, report)
        self._finalize_run(report)

        return report

    @trace_stage("pipeline_incremental_sync")
    def run_incremental_sync(self, since_timestamp: int) -> PipelineRunReport:
        """
        Embeds and upserts only products updated after since_timestamp.

        since_timestamp : Unix epoch. Provided by SyncWatermarkRepository.
                          The caller (sync.py) owns watermark read/write.
                          Pipeline only processes — it never manages state.

        When to use:
        - Daily scheduled sync (cron / Airflow / Celery beat)
        - Triggered sync after bulk product updates
        """
        logger.info(
            "Incremental sync started",
            since_timestamp=since_timestamp,
        )

        report = PipelineRunReport(
            run_type="incremental",
            started_at=int(time.time()),
        )

        batch_generator = self._fetcher.fetch_updated_batched(since_timestamp)
        self._execute_pipeline(batch_generator, report)
        self._finalize_run(report)

        return report

    # Core Execution Loop

    def request_stop(self):
        """
        Signals the pipeline to stop after the current batch completes.
        Does NOT kill mid-batch — current OpenAI call and Milvus upsert
        finish cleanly before the pipeline checks and exits.
        """
        self._stop_event.set()
        logger.info("Stop requested — pipeline will halt after current batch")

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _execute_pipeline(
        self,
        batch_generator: Generator[list[GlobalProduct], None, None],
        report: PipelineRunReport,
    ):
        """
        Shared execution loop for both full and incremental runs.

        Processes each PSQL batch through:
        1. Composition filter
        2. OpenAI sub-batching
        3. Milvus upsert
        4. Report accumulation

        A failure in one PSQL batch is logged and skipped —
        it does not abort the entire run. The pipeline is
        designed for resilience over atomicity at batch level.
        """
        for psql_batch in batch_generator:

            if self._stop_event.is_set():
                logger.info(
                    "Stop signal detected — halting pipeline",
                    batches_completed=report.psql_batches,
                    products_upserted=report.total_upserted,
                )
                report.run_type = f"{report.run_type}_stopped"
                break
            report.psql_batches += 1
            report.total_fetched += len(psql_batch)

            try:
                self._process_psql_batch(psql_batch, report)
            except Exception as e:
                logger.error(
                    "PSQL batch processing failed — skipping batch",
                    batch_number=report.psql_batches,
                    batch_size=len(psql_batch),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Accumulate as skipped — not silent loss
                report.total_skipped += len(psql_batch)

    def _process_psql_batch(
        self,
        psql_batch: list[GlobalProduct],
        report: PipelineRunReport,
    ):
        """
        Processes one PSQL batch (500 products) end-to-end.

        Step 1: Compose embedding texts
        Step 2: Filter out products with empty composition
        Step 3: Split into OpenAI sub-batches (100 each)
        Step 4: Embed each sub-batch
        Step 5: Assemble ProductVectors
        Step 6: Upsert into Milvus
        """

        # ── Step 1 & 2: Compose and filter ──────────────────────────
        composed_pairs = self._compose_and_filter(psql_batch, report)
        if not composed_pairs:
            logger.warning(
                "Entire PSQL batch produced empty compositions",
                batch_size=len(psql_batch),
            )
            return

        products_to_embed = [pair[0] for pair in composed_pairs]
        texts_to_embed = [pair[1] for pair in composed_pairs]

        # ── Step 3: Split into OpenAI sub-batches ───────────────────
        product_sub_batches = list(
            self._splitter.split(products_to_embed, settings.embedding_batch_size)
        )
        text_sub_batches = list(
            self._splitter.split(texts_to_embed, settings.embedding_batch_size)
        )

        # ── Steps 4, 5, 6: Embed → Assemble → Upsert ────────────────
        for products_sub, texts_sub in zip(product_sub_batches, text_sub_batches):
            self._process_embedding_sub_batch(products_sub, texts_sub, report)

    def _compose_and_filter(
        self,
        products: list[GlobalProduct],
        report: PipelineRunReport,
    ) -> list[tuple[GlobalProduct, str]]:
        """
        Composes embedding text for each product.
        Filters out products whose composition resolves to an empty string.

        An empty composition means the product has no meaningful
        semantic content to embed — no name, no category, no attributes.
        Embedding an empty string would produce a meaningless vector
        that pollutes the search space.
        """
        composed_pairs = []

        for product in products:
            try:
                text = self._composer.compose(product)
                if not text or not text.strip():
                    logger.warning(
                        "Empty composition — product skipped",
                        global_product_id=product.global_product_id,
                    )
                    report.total_skipped += 1
                    continue
                composed_pairs.append((product, text))

            except Exception as e:
                logger.error(
                    "Composition failed — product skipped",
                    global_product_id=product.global_product_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                report.total_skipped += 1

        return composed_pairs

    def _process_embedding_sub_batch(
        self,
        products: list[GlobalProduct],
        texts: list[str],
        report: PipelineRunReport,
    ):
        """
        Embeds one OpenAI sub-batch (≤100 texts), assembles vectors,
        and upserts into Milvus.

        Embedding failure on a sub-batch:
        - Logged with full context
        - Counted in report.embedding_failures
        - Sub-batch is skipped — not re-queued (sync will catch on next run)
        - Does NOT abort the parent PSQL batch
        """

        if self._stop_event.is_set():
            logger.info("Stop signal detected mid-batch — skipping sub-batch")
            report.total_skipped += len(products)
            return
        
        report.embedding_batches += 1

        # Embed
        try:
            embeddings = self._embedding_client.embed_batch(texts)
        except Exception as e:
            logger.error(
                "OpenAI embedding sub-batch failed",
                sub_batch_size=len(texts),
                embedding_batch_number=report.embedding_batches,
                error=str(e),
                error_type=type(e).__name__,
            )
            report.embedding_failures += 1
            report.total_skipped += len(products)
            return

        report.total_embedded += len(embeddings)

        # Assemble
        vectors = self._assembler.assemble(products, embeddings)

        # Upsert
        upsert_result: UpsertResult = self._repository.upsert_batch(vectors)
        report.milvus_upsert_batches += 1
        report.total_upserted += upsert_result.inserted
        report.total_rejected += upsert_result.rejected
        report.rejected_ids.extend(upsert_result.rejected_ids)

        # Rejection rate alert threshold
        if upsert_result.rejected_rate > 0.05:
            logger.warning(
                "High rejection rate on sub-batch",
                rejected_rate=upsert_result.rejected_rate,
                rejected_count=upsert_result.rejected,
                sub_batch_size=len(vectors),
            )

    # Finalization

    def _finalize_run(self, report: PipelineRunReport):
        """
        Post-run operations:
        1. Flush Milvus — seal segments, make data fully queryable
        2. Finalize report — set completed_at and duration
        3. Emit structured run summary log
        """
        logger.info("Flushing Milvus — sealing segments")
        self._repository.flush()

        report.finalize()

        logger.info(
            "Pipeline run complete",
            **report.to_log_dict(),
        )

        # Alert on systematic issues
        if report.overall_rejection_rate > 0.05:
            logger.warning(
                "Run-level rejection rate exceeded 5% threshold",
                overall_rejection_rate=report.overall_rejection_rate,
                total_rejected=report.total_rejected,
                total_fetched=report.total_fetched,
            )

        if report.embedding_failures > 0:
            logger.warning(
                "Embedding failures detected in this run",
                embedding_failures=report.embedding_failures,
                affected_products=report.total_skipped,
            )
# ```

# ---

# ## What This Step Establishes

# Four clean, separated responsibilities:
# ```
# PipelineRunReport           → Complete observability record for every run
# BatchSplitter               → Owns the PSQL(500) → OpenAI(100) boundary
# ProductVectorAssembler      → Pairs products with embeddings by position
# ProductEmbeddingPipeline    → Orchestrator — connects all components


# Three architectural decisions worth highlighting:
# The two-level batch boundary is explicit and owned — BatchSplitter is the only place that knows PSQL batch size is 500 and OpenAI sub-batch size is 100. Neither the fetcher nor the embedding client knows about each other's constraints. Change either size in settings.py and the split adjusts automatically.
# Pipeline never manages watermark state — run_incremental_sync() accepts since_timestamp as a parameter. It does not read or write watermarks itself. That responsibility belongs entirely to sync.py. This separation means the pipeline is purely a processing engine — stateless, testable, and reusable.
# Failure isolation is tiered — an OpenAI failure on a 100-product sub-batch skips that sub-batch but continues the parent 500-product PSQL batch. A PSQL batch failure skips that batch but continues the full run. Nothing short of a catastrophic unhandled exception aborts the entire ingestion. Skipped products are always counted — never silently dropped.