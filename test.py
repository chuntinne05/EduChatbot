from services.embedding_service import TextEmbeddingService
import numpy as np

DB_URL = "postgresql://trinhtrantrungtin@localhost:5432/admission_chatbot"
service = TextEmbeddingService(DB_URL)
queries = [
    "thông tin tuyển sinh đại học bách khoa",
    "ký túc xá sinh viên",
    "chương trình đào tạo kỹ thuật"
]

for query in queries:
    embedding = service.create_embedding_for_query(query)
    print(f"Query: {query}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print("---")

# emb = service.create_embedding_for_query("điều kiện tuyển sinh")
# print("Shape:", emb.shape)
# print("Norm:", np.linalg.norm(emb))