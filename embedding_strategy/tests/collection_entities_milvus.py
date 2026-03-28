from pymilvus import Collection, utility
from embedding.sync import MilvusConnectionManager

MilvusConnectionManager.connect()
print("Connected to Milvus")

collections = utility.list_collections()

collection_name = collections[0]
collection = Collection(collection_name)

print("Number of records:", collection.num_entities)