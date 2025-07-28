from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")

collection = Collection("bajaj")
collection.load()

print(f"📦 Total documents: {collection.num_entities}\n")
print("📌 Field Schema:")
for field in collection.schema.fields:
    print(f" - {field.name}: {field.dtype}")

results = collection.query(
    expr="", 
    output_fields=["id", "text", "source"], 
    limit=10
)

for i, doc in enumerate(results):
    print(f"🔹 Document {i+1}")
    print(f"   ID     : {doc['id']}")
    print(f"   Source : {doc['source']}")
    print(f"   Text   : {doc['text'][:300]}{'...' if len(doc['text']) > 300 else ''}")
    print("-" * 50)
