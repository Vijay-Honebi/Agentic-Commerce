from pymilvus import (
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
    connections,
    utility,
)
from embedding_strategy.config.settings import get_settings
from embedding_strategy.observability.logger import StructuredLogger, trace_stage

logger = StructuredLogger("embedding.schema")
settings = get_settings()


# Field Definitions

class ProductCollectionFields:
    """
    Single source of truth for every field in products_collection.
    Any change to the schema must go through here — nowhere else.

    Design decisions:
    - global_product_id: VARCHAR(64) — UUIDs are max 36 chars, 64 gives headroom
    - embedding: 512-dim via Matryoshka truncation of text-embedding-3-small
    - category_path: VARCHAR(256) — "Shoes > Sports > Badminton" style, 3 levels max
    - bu_id: VARCHAR(64) — business unit scoping on every query
    - updated_at: INT64 Unix timestamp — used ONLY for incremental sync watermark
    """

    GLOBAL_PRODUCT_ID = FieldSchema(
        name="global_product_id",
        dtype=DataType.VARCHAR,
        max_length=64,
        is_primary=True,
        auto_id=False,           # We own the ID — never let Milvus generate it
        description="Primary key. Bridge to PSQL global_products table.",
    )

    EMBEDDING = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=512,
        description="512-dim Matryoshka vector from text-embedding-3-small.",
    )

    CATEGORY_PATH = FieldSchema(
        name="category_path",
        dtype=DataType.VARCHAR,
        max_length=256,
        description="Structural category path. Pre-filter to scope ANN search.",
    )

    BU_IDS = FieldSchema(
        name="bu_ids",
        dtype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=50,        # max BUs a single product can belong to
                                # 50 is generous — adjust to the domain reality
        max_length=64,          # max chars per individual bu_id element
        description=(
            "All business unit IDs this product belongs to. "
            "One product can belong to multiple BUs. "
            "Use array_contains(bu_ids, 'X') to filter."
        ),
    )

    UPDATED_AT = FieldSchema(
        name="updated_at",
        dtype=DataType.INT64,
        description="Unix timestamp. Used for incremental sync watermark only.",
    )


# Index Configuration

class HNSWIndexConfig:
    """
    HNSW (Hierarchical Navigable Small World) index parameters.

    Why HNSW over IVF_FLAT or ANNOY:
    - Best recall/latency tradeoff for e-commerce scale (millions of products)
    - No training phase required (unlike IVF — critical for daily incremental sync)
    - Consistent low latency even as collection grows
    - Supports incremental inserts without index rebuild

    Parameter reasoning:
    - M=16: Connections per node. 16 is the proven sweet spot for
            high-dimensional text vectors. Higher M = better recall
            but more memory. 32+ gives diminishing returns.
    - ef_construction=256: Build-time search depth. Higher = better
            index quality but slower build. 256 is production standard
            for recall-sensitive workloads.
    - metric_type=COSINE: text-embedding-3-small vectors are optimized
            for cosine similarity. DO NOT use L2 for OpenAI embeddings.
    """

    INDEX_TYPE = "HNSW"
    METRIC_TYPE = "COSINE"

    PARAMS = {
        "M": settings.hnsw_m,                          # from settings: default 16
        "efConstruction": settings.hnsw_ef_construction,  # from settings: default 256
    }

    SEARCH_PARAMS = {
        "ef": 64,     # Query-time search depth.
                      # ef >= top_k always. 64 gives good recall for top 10-20 results.
                      # Can be tuned per query for recall vs latency tradeoff.
    }


# Schema Builder

class ProductCollectionSchema:
    """
    Builds and validates the Milvus CollectionSchema for products_collection.
    Stateless — only constructs, never connects to Milvus.
    """

    DESCRIPTION = (
        "Honebi global products semantic index. "
        "Stores stable product facts + embeddings for AI-powered discovery. "
        "Transactional data (price, stock, active status) lives in PSQL only."
    )

    @staticmethod
    def build() -> CollectionSchema:
        fields = [
            ProductCollectionFields.GLOBAL_PRODUCT_ID,
            ProductCollectionFields.EMBEDDING,
            ProductCollectionFields.CATEGORY_PATH,
            ProductCollectionFields.BU_IDS,
            ProductCollectionFields.UPDATED_AT,
        ]

        schema = CollectionSchema(
            fields=fields,
            description=ProductCollectionSchema.DESCRIPTION,
            enable_dynamic_field=False,   # Strict schema — no surprise fields
        )

        logger.debug(
            "Schema built",
            fields=[f.name for f in fields],
            primary_key="global_product_id",
            vector_dim=512,
        )

        return schema


# Collection Manager

class ProductCollectionManager:
    """
    Owns the full lifecycle of products_collection in Milvus:
    - Create (idempotent)
    - Index creation
    - Load into memory
    - Drop (only for dev/test)

    This is the ONLY place in the codebase that calls Milvus
    collection management APIs. All other modules receive a
    Collection instance — they never manage it.
    """

    def __init__(self):
        self._collection_name = settings.milvus_collection_name
        self._collection: Collection | None = None

    @trace_stage("milvus_collection_ensure")
    def ensure_collection(self) -> Collection:
        """
        Idempotent. Creates collection + index if not exists.
        Safe to call on every application startup.
        Returns a loaded Collection ready for insert and search.
        """
        if utility.has_collection(self._collection_name):
            logger.info(
                "Collection already exists, loading",
                collection=self._collection_name,
            )
            self._collection = Collection(name=self._collection_name)
        else:
            logger.info(
                "Collection not found, creating",
                collection=self._collection_name,
            )
            schema = ProductCollectionSchema.build()
            self._collection = Collection(
                name=self._collection_name,
                schema=schema,
                consistency_level="Bounded",
                # Bounded consistency: reads may lag writes by a small window.
                # Chosen deliberately — we do not need Strong consistency for
                # product search. Bounded gives significantly better throughput.
            )
            self._create_index()

        self._load_collection()
        return self._collection

    def _create_index(self):
        """Creates HNSW index on the embedding field."""
        logger.info(
            "Creating HNSW index",
            collection=self._collection_name,
            M=HNSWIndexConfig.PARAMS["M"],
            ef_construction=HNSWIndexConfig.PARAMS["efConstruction"],
        )

        self._collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": HNSWIndexConfig.INDEX_TYPE,
                "metric_type": HNSWIndexConfig.METRIC_TYPE,
                "params": HNSWIndexConfig.PARAMS,
            },
            index_name="embedding_hnsw_idx",
        )

        logger.info("HNSW index created", collection=self._collection_name)

    def _load_collection(self):
        """
        Loads collection into Milvus query node memory.
        Required before any search or query operation.
        """
        self._collection.load()
        logger.info("Collection loaded into memory", collection=self._collection_name)

    def get_collection_stats(self) -> dict:
        """Returns row count and index state for observability."""
        col = Collection(name=self._collection_name)
        stats = {
            "collection": self._collection_name,
            "row_count": col.num_entities,
            "index_state": str(col.index().params),
        }
        logger.info("Collection stats fetched", **stats)
        return stats

    def drop_collection(self):
        """
        DANGER: Irreversible. Only for dev/test environments.
        Guarded by an explicit confirmation flag.
        """
        raise NotImplementedError(
            "drop_collection is intentionally unimplemented in this module. "
            "If you need to drop the collection for dev/test, do it directly "
            "from the Milvus console or a one-off script with explicit confirmation."
        )
# ```

# ---

# ## What This Step Establishes

# Three clean, separated responsibilities:
# ```
# ProductCollectionFields    → What fields exist and why (documented)
# ProductCollectionSchema    → How fields compose into a schema
# ProductCollectionManager   → How the collection is managed in Milvus




# Key decisions:

# auto_id=False — we own the primary key. Milvus never generates IDs for us. This is critical for idempotent upserts during daily sync
# enable_dynamic_field=False — strict schema. No accidental fields bleeding in
# consistency_level="Bounded" — not Strong. Product search does not need read-your-write consistency. This gives significantly better query throughput
# drop_collection raises NotImplementedError — a guard against accidental data loss in production. Dropping must be a deliberate out-of-band action
# COSINE not L2 — OpenAI embeddings are unit-normalized and explicitly optimized for cosine. Using L2 here would be a silent quality regression