from pymilvus import connections, Collection

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)


collection = Collection("products_collection")
print(collection.num_entities)