from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
query = "What are the supported calculator operations?"
vector = model.encode(query, normalize_embeddings=True).tolist()

print(f"Dimensions: {len(vector)}")
print("Vector:", vector)