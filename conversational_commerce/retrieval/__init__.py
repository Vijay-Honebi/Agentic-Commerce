# conversational_commerce/retrieval/__init__.py

from retrieval.hybrid_retrieval import HybridRetriever, HybridRetrievalResult
from retrieval.milvus_client import (
    connect_milvus,
    disconnect_milvus,
    vector_search,
    MilvusSearchResult,
    MilvusFilterBuilder,
)
from retrieval.psql_client import (
    init_psql_retrieval_pool,
    close_psql_retrieval_pool,
    enrich_candidates,
    fetch_product_by_id,
    EnrichedProduct,
)
from retrieval.ranker import BusinessRanker, RankingWeights
from retrieval.relaxation_engine import RelaxationEngine

__all__ = [
    # Milvus
    "connect_milvus", "disconnect_milvus",
    "vector_search", "MilvusSearchResult", "MilvusFilterBuilder",
    # PSQL
    "init_psql_retrieval_pool", "close_psql_retrieval_pool",
    "enrich_candidates", "fetch_product_by_id", "EnrichedProduct",
    # Hybrid
    "HybridRetriever", "HybridRetrievalResult",
    # Ranker
    "BusinessRanker", "RankingWeights",
    # Relaxation
    "RelaxationEngine",
]
# ```

# ---

# ## What Step 3 Gave You — Full Picture
# ```
# User Query Vector
#       │
#       ▼
# MilvusClient.vector_search()
#   └── Milvus pre-filter (store_id, in_stock) ← high selectivity only
#   └── ANN search → top 50 candidate_ids + cosine scores
#       │
#       ▼
# PSQLClient.enrich_candidates()
#   └── Parameterised SQL with ALL hard constraints
#   └── LEFT JOIN product_images for primary image
#   └── Returns only products passing ALL filters
#       │
#       ▼
# RelaxationEngine (if candidates < 10)
#   └── Round 1: widen price ±10%
#   └── Round 2: drop one attribute (color → material → size)
#   └── Max 2 rounds, full audit trail logged
#       │
#       ▼
# BusinessRanker.rank()
#   └── Weighted scoring: 50% semantic + 20% rating + 20% popularity + 10% boost
#   └── Sort by ParsedQuery.sort_order
#   └── Exclude already-shown products (session dedup)
#   └── Attach query-relevant key_attributes to each card
#   └── Truncate to result_limit
#       │
#       ▼
# list[ProductCard]  ← ready for tools layer

# milvus_client.py  ←─────────────────────────────┐
# psql_client.py    ←──────────────────────────┐   │
#                                               │   │
# relaxation_engine.py  (uses schemas only)     │   │
#                                               │   │
# hybrid_retriever.py ──────── imports all four─┘───┘
#                                               │
# ranker.py  ←──────────────────────────────────┘