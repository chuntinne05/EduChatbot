from services.embedding_service import TextEmbeddingService
import numpy as np

DB_URL = "postgresql://trinhtrantrungtin@localhost:5432/admission_chatbot"
service = TextEmbeddingService(DB_URL)

emb = service.create_embedding_for_query("điều kiện tuyển sinh")
print("Shape:", emb.shape)
print("Norm:", np.linalg.norm(emb))