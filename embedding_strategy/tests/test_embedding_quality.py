import os
import sys

# ── Make sure imports resolve from embedding_strategy root ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pymilvus import connections, Collection
from db.postgres import PostgresConnectionPool
from config.settings import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TOP_K        = 10       # how many candidates to fetch from Milvus
BU_ID        = None     # set to a specific BU ID to scope, or None to skip filter
CATEGORY     = None     # set to e.g. "Shoes > Sports" to narrow, or None to skip


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
    if BU_ID:
        expressions.append(f'array_contains(bu_ids, "{BU_ID}")')
    if CATEGORY:
        safe = CATEGORY.replace('"', '\\"')
        expressions.append(f'category_path like "{safe}%"')

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
        output_fields=["global_product_id", "category_path"],
    )

    candidates = []
    for hit in results[0]:
        candidates.append({
            "global_product_id": hit.fields.get("global_product_id"),
            "category_path":     hit.fields.get("category_path", ""),
            "score":             round(float(hit.score), 4),
        })

    return candidates


# ─────────────────────────────────────────────
# Step 3 — Fetch product names from PSQL
# ─────────────────────────────────────────────

def fetch_product_names(product_ids: list[str]) -> dict[str, str]:
    if not product_ids:
        return {}

    PostgresConnectionPool.initialize()

    from db.postgres import PostgresConnectionPool as Pool
    import psycopg2.extras

    id_placeholders = ",".join(["%s"] * len(product_ids))
    query = f"""
        SELECT id, name
        FROM products
        WHERE id IN ({id_placeholders})
    """

    with Pool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, product_ids)
            rows = cur.fetchall()

    return {str(row["id"]): row["name"] for row in rows}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def test_query(query: str):
    print(f"\n{'─' * 60}")
    print(f"  Query : {query}")
    if BU_ID:
        print(f"  BU    : {BU_ID}")
    if CATEGORY:
        print(f"  Cat   : {CATEGORY}")
    print(f"{'─' * 60}\n")

    # Embed
    print("Embedding query...")
    vector = embed_query(query)
    print(f"Vector dim: {len(vector)}\n")

    # Search
    print("Searching Milvus...")
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    candidates = search_milvus(vector)
    connections.disconnect("default")

    if not candidates:
        print("No results found.")
        return

    # Fetch names
    ids = [c["global_product_id"] for c in candidates]
    names = fetch_product_names(ids)

    # Print results
    print(f"{'Rank':<5} {'Score':<8} {'Product Name':<45} {'Category':<30} {'ID'}")
    print(f"{'─'*5} {'─'*8} {'─'*45} {'─'*30} {'─'*36}")

    for i, candidate in enumerate(candidates, start=1):
        pid   = candidate["global_product_id"]
        score = candidate["score"]
        cat   = candidate["category_path"] or "—"
        name  = names.get(pid, "— name not found —")
        print(f"{i:<5} {score:<8} {name[:44]:<45} {cat[:29]:<30} {pid}")

    print()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Run from command line:
    #   python test_embedding_quality.py "red cotton shirt for men"
    #
    # Or edit the queries list below and run without args

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        test_query(query)
    else:
        # Batch test — edit these to match your catalog
        queries = [
            "red cotton shirt for men",
            "running shoes for women",
            "casual summer dress",
            "leather wallet black",
            "wireless bluetooth headphones",
        ]
        for q in queries:
            test_query(q)

    PostgresConnectionPool.close_all()