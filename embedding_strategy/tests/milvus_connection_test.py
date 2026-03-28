from pymilvus import utility
from embedding.sync import MilvusConnectionManager

MilvusConnectionManager.connect()
print("Connected to Milvus")

collections = utility.list_collections()
print("Collections:", collections)

drop_collection_name = collections[0]
drop_flag = True

if drop_flag:
    if utility.has_collection(drop_collection_name):
        utility.drop_collection(drop_collection_name)
        print(f"Dropped collection: {drop_collection_name}")
    else:
        print(f"Collection '{drop_collection_name}' does not exist")

after_drop_collections = utility.list_collections()
print("After Drop Collections:", after_drop_collections)