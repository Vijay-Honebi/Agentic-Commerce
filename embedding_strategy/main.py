# main.py

import sys
import argparse

from db.postgres import PostgresConnectionPool
from embedding.sync import SyncOrchestrator, SyncRunMode
from observability.logger import StructuredLogger

logger = StructuredLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Honebi Product Embedding Sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run modes:
  auto          Inspect watermark → full if none, incremental if exists (default)
  full          Re-embed entire catalog regardless of watermark state
  incremental   Embed only products updated since last watermark

Examples:
  python main.py                    # AUTO mode (daily cron default)
  python main.py --mode full        # Force full re-index
  python main.py --mode incremental # Force delta sync
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "incremental"],
        default="auto",
        help="Sync run mode (default: auto)",
    )
    return parser.parse_args()


def resolve_run_mode(mode_str: str) -> SyncRunMode:
    return {
        "auto": SyncRunMode.AUTO,
        "full": SyncRunMode.FULL,
        "incremental": SyncRunMode.INCREMENTAL,
    }[mode_str]


def main():
    args = parse_args()
    mode = resolve_run_mode(args.mode)

    logger.info(
        "Honebi embedding sync starting",
        requested_mode=args.mode,
    )

    # Initialize connection pool once at startup
    PostgresConnectionPool.initialize()

    try:
        orchestrator = SyncOrchestrator()
        result = orchestrator.run(mode=mode)

        if result.is_successful:
            logger.info(
                "Sync completed successfully",
                total_upserted=result.pipeline_report.total_upserted,
                duration_seconds=result.pipeline_report.duration_seconds,
                new_watermark=result.new_watermark_timestamp,
            )
            sys.exit(0)
        else:
            logger.error(
                "Sync completed with failure",
                error=result.error,
                watermark_saved=result.watermark_saved,
            )
            sys.exit(1)

    finally:
        PostgresConnectionPool.close_all()


if __name__ == "__main__":
    main()
# ```

# ---

# ## Complete Project Map

# Every file, every responsibility, one final view:
# ```
# honebi/
# │
# ├── main.py                         # CLI entrypoint. Args → SyncOrchestrator.
# │
# ├── config/
# │   └── settings.py                 # All config. Env-driven. Cached singleton.
# │
# ├── observability/
# │   └── logger.py                   # Structured JSON logger + @trace_stage decorator.
# │
# ├── db/
# │   └── postgres.py                 # Connection pool, batch fetchers, watermark repo.
# │
# └── embedding/
#     ├── composer.py                 # Text composition. Semantic meaning decisions.
#     ├── client.py                   # OpenAI wrapper. Retry, batching, observability.
#     ├── schema.py                   # Milvus schema, HNSW index, collection lifecycle.
#     ├── repository.py               # Milvus read/write. Validation. Filter builder.
#     ├── pipeline.py                 # Orchestrates fetch → embed → upsert per run.
#     └── sync.py                     # Decision logic. Watermark. Milvus connection.
# ```

# ---

# ## The Dependency Flow
# ```
# main.py
#   └── SyncOrchestrator (sync.py)
#         ├── MilvusConnectionManager       ← connect / disconnect
#         ├── ProductCollectionManager      ← ensure collection + index
#         ├── SyncWatermarkRepository       ← read / write watermark
#         └── ProductEmbeddingPipeline (pipeline.py)
#               ├── GlobalProductFetcher    ← PSQL batch generator
#               ├── EmbeddingComposer       ← text composition
#               ├── EmbeddingClient         ← OpenAI API
#               ├── ProductVectorAssembler  ← pair products + embeddings
#               └── ProductVectorRepository ← Milvus upsert + search



# # First run — AUTO detects no watermark → full ingestion
# python main.py

# # Daily cron (07:00 UTC)
# 0 7 * * * cd /app && python main.py --mode auto

# # Force full re-index after schema migration
# python main.py --mode full

# # Manual delta catch-up
# python main.py --mode incremental