import os
import sys

# ── Make sure imports resolve from embedding_strategy root ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pymilvus import connections, Collection
from embedding_strategy.db.postgres import PostgresConnectionPool as Pool
from embedding_strategy.config.settings import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TOP_K        = 10       # how many candidates to fetch from Milvus


# ─────────────────────────────────────────────
# Step 1 — Embed the query
# ─────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=[query],
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────
# Step 2 — Search Milvus
# ─────────────────────────────────────────────

def search_milvus(query_vector: list[float]) -> list[dict]:
    collection = Collection(name=settings.milvus_collection_name)
    collection.load()

    # Build filter
    expressions = []

    expr = " && ".join(expressions) if expressions else None

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={
            "metric_type": "COSINE",
            "params": {"ef": 64},
        },
        limit=TOP_K,
        expr=expr,
        output_fields=["global_product_id"],
    )

    candidates = []
    for hit in results[0]:
        candidates.append({
            "global_product_id": hit.fields.get("global_product_id"),
            "score":             round(float(hit.score), 4),
        })

    return candidates


# ─────────────────────────────────────────────
# Step 3 — Fetch product names from PSQL
# ─────────────────────────────────────────────

def fetch_product_images(product_ids: list[str]) -> list[dict]:
    """
    Fetch product images for given product IDs.

    Args:
        product_ids: List of product IDs

    Returns:
        List of dicts:
        [
          {"id": "...", "images": [...]},
          ...
        ]
    """
    if not product_ids:
        return []

    id_placeholders = ",".join(["%s"] * len(product_ids))

    query = f"""
        SELECT p.id, p.images, p.code, p.name, s.url_slug
        FROM products p
        JOIN seo s ON p.id = s.object_id
        WHERE p.id IN ({id_placeholders})
    """

    with Pool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, product_ids)

            rows = cur.fetchall()

    return list(rows)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def test_query(query: str):
    print(f"\n{'─' * 60}")
    print(f"  Query : {query}")
    print(f"{'─' * 60}\n")

    # Embed
    vector = embed_query(query)

    # Search Milvus
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    candidates = search_milvus(vector)
    connections.disconnect("default")

    if not candidates:
        return []

    # Extract IDs
    ids = [c["global_product_id"] for c in candidates]

    # Build score lookup
    score_map = {
        c["global_product_id"]: c["score"]
        for c in candidates
    }

    # Fetch product metadata
    products = fetch_product_images(ids)

    # Sort by score (highest first)
    products.sort(
            key=lambda p: score_map.get(p["id"], 0),
            reverse=True
        )

    # Assign sequence numbers (1 = highest score)
    for i, p in enumerate(products, start=1):
        p["sequence"] = i

    return products

