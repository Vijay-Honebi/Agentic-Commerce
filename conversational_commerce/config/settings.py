from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class MilvusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MILVUS_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=19530)
    collection_name: str = Field(default="product_embeddings")
    metric_type: str = Field(default="COSINE")
    # Candidates pulled from Milvus before PSQL hard-filter reranking.
    # 50 gives enough headroom for relaxation without over-fetching.
    top_k: int = Field(default=50)
    # If enriched candidates fall below this, relaxation engine triggers.
    min_candidates_threshold: int = Field(default=10)
    search_ef: int = Field(default=128)          # HNSW ef parameter


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    # dsn: str = Field(
    #     default="postgresql+asyncpg://postgres:password@localhost:5432/honebi"
    # )

    host: str
    port: int = 5432
    db: str
    user: str
    password: str
    statement_timeout_ms: int = 300000
    @computed_field
    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:"
            f"{self.password}@"
            f"{self.host}:"
            f"{self.port}/"
            f"{self.db}"
        )

    pool_min_size: int = Field(default=5)
    pool_max_size: int = Field(default=20)
    command_timeout_seconds: int = Field(default=30)
    # Session table — owned exclusively by session_store.py
    session_table: str = Field(default="agent_sessions")


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")

    api_key: str = Field(default="")

    # ── Model: gpt-4.1-mini ─────────────────────────────────────────────────
    # Chosen over gpt-4o-mini for three reasons specific to this system:
    #   1. Superior agentic tool-calling fidelity (multi-step, multi-turn)
    #   2. 1M token context window — full Phase 2/3/4 sessions fit without trimming
    #   3. 75% cached-input discount — system prompt + tool schemas are cached
    #      on every call after the first, making production cost negligible
    # Full justification documented in architecture decision records.
    # ────────────────────────────────────────────────────────────────────────
    chat_model: str = Field(default="gpt-4.1-mini")

    # temperature=0 → deterministic structured outputs from LLM parser.
    # Non-zero would risk inconsistent filter extraction across identical queries.
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=1024)
    request_timeout_seconds: int = Field(default=30)


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    # Must match the model used in embedding_strategy/ pipeline.
    # Changing this invalidates all vectors in Milvus — coordinate with ML team.
    model_name: str = Field(default="text-embedding-3-small")
    dimensions: int = Field(default=1536)
    batch_size: int = Field(default=64)


class SessionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SESSION_", extra="ignore")

    ttl_seconds: int = Field(default=3600)          # 1-hour inactivity expiry
    # Sliding window: only last N turns sent to LLM context.
    # Prevents prompt overflow on long sessions. 20 turns ≈ ~4K tokens typical.
    max_turns_in_memory: int = Field(default=20)
    max_context_tokens: int = Field(default=8000)   # hard cap before truncation


class RelaxationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RELAXATION_", extra="ignore")

    # Hard cap at 2 rounds — guardrail against infinite relaxation loops.
    # Round 1: relax price/size/color by 10%.
    # Round 2: relax category boundaries by 10%.
    # If still < threshold after round 2 → return best-effort, flag in response.
    max_rounds: int = Field(default=2)
    step_percent: float = Field(default=0.10)


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OBS_", extra="ignore")

    log_level: str = Field(default="INFO")
    # JSON logs in staging/prod for log aggregators (Datadog, CloudWatch, etc.)
    # Pretty logs in development for human readability.
    json_logs: bool = Field(default=True)
    # When True: every LLM call, tool call, retrieval step emits a trace event.
    trace_enabled: bool = Field(default=True)
    # Queries exceeding this threshold emit a WARN log with full timing breakdown.
    slow_query_threshold_ms: int = Field(default=500)


class Settings(BaseSettings):
    """
    Master settings singleton for conversational_commerce.

    Usage (everywhere in the codebase):
        from config.settings import get_settings
        settings = get_settings()

    Never instantiate Settings() directly outside this module.
    """

    # model_config = SettingsConfigDict(
    #     env_file=".env",
    #     env_file_encoding="utf-8",
    #     extra="ignore",
    #     case_sensitive=False,
    # )
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=ENV_PATH,
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

    environment: Environment = Field(default=Environment.DEVELOPMENT)
    app_name: str = Field(default="honebi-conversational-commerce")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    # ── Nested settings (each reads its own env prefix) ──────────────────
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    relaxation: RelaxationSettings = Field(default_factory=RelaxationSettings)
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings
    )

    business_unit_id: str = Field(
        default="default_bu",
        description=(
            "Business unit identifier for this deployment. "
            "e.g. 'south_zone', 'online', 'retail_north'. "
            "Set once in .env for single-BU deployments. "
            "Sent via X-Business-Unit-ID header for multi-BU."
        ),
    )

    entity_id: str = Field(
        default="default_entity",
        description=(
            "Entity identifier within the business unit. "
            "e.g. 'bangalore_store', 'mumbai_store', 'website'. "
            "Set once in .env for single-entity deployments. "
            "Sent via X-Entity-ID header for multi-entity."
        ),
    )

    @field_validator("environment", mode="before")
    @classmethod
    def normalise_env(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached Settings singleton.
    First call reads from environment + .env file.
    All subsequent calls return the cached instance — zero I/O overhead.
    """
    return Settings()