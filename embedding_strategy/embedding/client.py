import time
from typing import List
from openai import OpenAI
from embedding_strategy.config.settings import get_settings
from embedding_strategy.observability.logger import StructuredLogger, trace_stage

logger = StructuredLogger("embedding.client")
settings = get_settings()


class EmbeddingClient:
    """
    Wraps OpenAI text-embedding-3-small with:
    - Matryoshka dimension truncation (512)
    - Batch processing
    - Exponential backoff on rate limits
    - Per-call observability
    """

    _MAX_RETRIES = 5
    _BASE_BACKOFF_SECONDS = 2

    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        self._dimensions = settings.embedding_dimensions

    @trace_stage("openai_embedding_batch")
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of texts. Returns list of 512-dim vectors.
        Raises on unrecoverable errors after max retries.
        """
        if not texts:
            return []

        self._validate_texts(texts)

        for attempt in range(self._MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dimensions,     # Matryoshka truncation
                )
                vectors = [item.embedding for item in response.data]

                logger.info(
                    "Batch embedded successfully",
                    batch_size=len(texts),
                    dimensions=self._dimensions,
                    total_tokens=response.usage.total_tokens,
                    attempt=attempt + 1,
                )
                return vectors

            except Exception as e:
                is_rate_limit = "rate_limit" in str(e).lower()
                is_last_attempt = attempt == self._MAX_RETRIES - 1

                if is_last_attempt:
                    logger.error(
                        "Embedding batch failed after all retries",
                        error=str(e),
                        batch_size=len(texts),
                    )
                    raise

                backoff = self._BASE_BACKOFF_SECONDS ** (attempt + 1)
                logger.warning(
                    "Embedding attempt failed, retrying",
                    attempt=attempt + 1,
                    backoff_seconds=backoff,
                    is_rate_limit=is_rate_limit,
                    error=str(e),
                )
                time.sleep(backoff)

    def _validate_texts(self, texts: List[str]):
        empty = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if empty:
            logger.warning(
                "Empty texts found in batch",
                empty_indices=empty,
                count=len(empty),
            )