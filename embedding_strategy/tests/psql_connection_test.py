from db.postgres import PostgresConnectionPool

try:
    with PostgresConnectionPool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok;")
            result = cur.fetchone()

    print("✅ Connection pool working!")
    print("Result:", result)

except Exception as e:
    print("❌ Pool test failed:")
    print(e)