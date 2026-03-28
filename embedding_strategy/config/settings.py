# config/settings.py

from typing import Optional

from pydantic import ConfigDict, computed_field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512          # Matryoshka truncation

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "products_collection"

    # PSQL
    postgres_host: str
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_statement_timeout_ms: int = 300000  # 5 minutes. Some batch queries can be slow, especially on large datasets. This prevents them from hanging indefinitely.

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:"
            f"{self.postgres_port}/"
            f"{self.postgres_db}"
        )

    # Pipeline
    ingestion_batch_size: int = 500          # records per PSQL fetch
    embedding_batch_size: int = 100          # records per OpenAI call
    milvus_insert_batch_size: int = 1000
    incremental_sync_lookback_hours: int = 25  # slight overlap for safety. applied against p.modified_at

    # HNSW Index
    hnsw_m: int = 16
    hnsw_ef_construction: int = 256

    # Server
    host: str | None = None
    port: int | None = None

    # Test
    test_product_limit: Optional[int] = None
    # When set, pipeline processes at most this many products total.
    # Leave unset (None) in production — no cap applied.
    # Example: TEST_PRODUCT_LIMIT=100 in .env for local testing.

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()