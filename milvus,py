from pymilvus import connections, utility, Collection
from pprint import pprint

# Step 1: Connect to Milvus
print("🔌 Connecting to Milvus at http://localhost:19530")
connections.connect("default", host="localhost", port="19530")

# Step 2: List all collections
collections = utility.list_collections()
if not collections:
    print("❌ No collections found.")
    exit()

print(f"✅ Found {len(collections)} collection(s): {collections}\n")

# Step 3: Inspect each collection
for name in collections:
    print(f"🔍 Inspecting collection: {name}")
    try:
        collection = Collection(name)
        print("📐 Schema:")
        for field in collection.schema.fields:
            print(f" - {field.name} ({field.dtype})")

        # Load and sample a few documents
        print("📥 Loading collection into memory...")
        collection.load()

        print("📄 Sample documents:")
        sample_docs = collection.query(
            expr="", output_fields=[field.name for field in collection.schema.fields], limit=5
        )
        if sample_docs:
            pprint(sample_docs)
        else:
            print("⚠️ No documents found.")

        print("📊 Total documents:", collection.num_entities)
    except Exception as e:
        print(f"❌ Error loading collection '{name}': {e}")

    print("-" * 80)

# Done
print("✅ Done inspecting Milvus.")
