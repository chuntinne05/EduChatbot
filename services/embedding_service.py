"""
Text Embedding Service - Tạo embedding cho text chunks
Sử dụng PhoBERT và multilingual-e5 models
"""

import os
import logging
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from models.database_models import TextChunk, ProcessingLog
import time
from datetime import datetime, timezone
from tqdm import tqdm
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEmbeddingService:
    """Service để tạo và lưu text embeddings"""
    
    def __init__(self, 
                 db_url: str,
                 model_name: str = "intfloat/multilingual-e5-base",
                 batch_size: int = 32):
        """
        Initialize text embedding service
        
        Args:
            db_url: PostgreSQL connection URL
            model_name: Hugging Face model name
            batch_size: Batch size for processing
        """
        self.db_url = db_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.engine = create_engine(db_url)
        self._load_model()

        # Qdrant Client for vector storage
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = "text_chunks"

        self._create_qdrant_collection()
        
        logger.info(f"Text Embedding Service initialized with {model_name} on {self.device}. Qdrant connected at {settings.QDRANT_URL}")

    def _create_qdrant_collection(self):
        """Create Qdrant collection if not exists"""
        if not self.qdrant_client.has_collection(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def _load_model(self):
        """Load embedding model"""
        try:
            # Option 1: SentenceTransformers (recommended for multilingual-e5)
            if "e5" in self.model_name.lower():
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.tokenizer = self.model.tokenizer
                self.model_type = "sentence_transformer"
                
            # Option 2: Direct transformers (cho PhoBERT)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.model_type = "transformer"
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text: str, category: Optional[str] = None, chunk: Optional[TextChunk] = None) -> str:
        """
        Preprocess text trước khi tạo embedding
        
        Args:
            text: Raw text content
            category: Text category for context
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Clean text
        text = text.strip().replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split()) 
        
        # Add category context for better embedding
        if category and "e5" in self.model_name.lower():
            # E5 models work better with instruction prefix
            category_prefix = {
                'tuyen_sinh': 'passage: Tuyển sinh: ',
                'chuong_trinh': 'passage: Chương trình đào tạo: ',
                'co_so_vat_chat': 'passage: Cơ sở vật chất: ',
                'sinh_vien': 'passage: Hoạt động sinh viên: ',
                'thong_tin_truong': 'passage: Thông tin trường: ',
                'nganh_hoc': 'passage: Ngành học: ',
                'hoc_phi': 'passage: Học phí: ',
                'chinh_sach': 'passage: Chính sách: ',
                'khac': 'passage: Giáo dục khác: ',
                'general': 'passage: Chung: '
            }
            prefix = category_prefix.get(category.lower(), category_prefix['general'])
            if chunk and chunk.school:
                prefix = f"{prefix}{chunk.school}: "
            text = prefix + text
        
        # Truncate if too long (most models have 512 token limit)
        if len(text) > 2000:  # Rough estimate for tokens
            text = text[:2000] + "..."
        
        return text
    
    def create_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """
        Tạo embeddings cho batch of texts
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Numpy array of embeddings (batch_size, embedding_dim)
        """
        try:
            if self.model_type == "sentence_transformer":
                # SentenceTransformer approach
                embeddings = self.model.encode(
                    texts, 
                    batch_size=min(len(texts), self.batch_size),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                
            else:
                # Manual transformer approach
                embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i+self.batch_size]
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Mean pooling
                        batch_embeddings = self._mean_pooling(
                            outputs.last_hidden_state,
                            inputs['attention_mask']
                        )
                        
                        # Normalize
                        batch_embeddings = torch.nn.functional.normalize(
                            batch_embeddings, p=2, dim=1
                        )
                        
                        embeddings.append(batch_embeddings.cpu().numpy())
                
                embeddings = np.vstack(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def _mean_pooling(self, last_hidden_states, attention_mask):
        """Mean pooling for transformer outputs"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_unprocessed_chunks(self, limit: int = 1000) -> List[TextChunk]:
        """
        Lấy text chunks chưa có embedding
        
        Args:
            limit: Maximum number of chunks to process
            
        Returns:
            List of TextChunk objects
        """
        with Session(self.engine) as session:
            chunks = session.query(TextChunk).filter(
                TextChunk.embedding.is_(None)
            ).limit(limit).all()
            
            return chunks
    
    def update_chunk_embedding(self, chunk_id: str, embedding: np.ndarray):
        """
        Cập nhật embedding cho một text chunk
        
        Args:
            chunk_id: Chunk ID
            embedding: Embedding vector
        """
        try:
            with Session(self.engine) as session:
                chunk = session.query(TextChunk).filter(
                    TextChunk.chunk_id == chunk_id
                ).first()
                
                if chunk:
                    # Convert numpy array to list for PostgreSQL
                    chunk.embedding = embedding.tolist()
                    chunk.updated_at = datetime.now(timezone.utc)
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {e}")
            raise
    
    def process_text_chunks(self, limit: int = 1000) -> Dict[str, int]:
        """
        Xử lý text chunks để tạo embeddings
        
        Args:
            limit: Maximum chunks to process
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        processed_count = 0
        error_count = 0
        
        # Create processing log
        log_id = self._create_processing_log("text_embedding", "processing")
        
        try:
            # Get unprocessed chunks
            chunks = self.get_unprocessed_chunks(limit)
            
            if not chunks:
                logger.info("No chunks to process")
                return {"processed": 0, "errors": 0}
            
            logger.info(f"Processing {len(chunks)} text chunks...")
            
            # Process in batches
            for i in tqdm(range(0, len(chunks), self.batch_size), desc="Processing batches"):
                batch_chunks = chunks[i:i+self.batch_size]
                
                try:
                    # Prepare texts
                    texts = []
                    for chunk in batch_chunks:
                        preprocessed_text = self.preprocess_text(
                            chunk.content, 
                            chunk.category,
                            chunk
                        )
                        texts.append(preprocessed_text)
                    
                    # Create embeddings
                    embeddings = self.create_embedding_batch(texts)
                    
                    # Update database
                    for chunk, embedding in zip(batch_chunks, embeddings):
                        self.update_chunk_embedding(chunk.chunk_id, embedding)
                        processed_count += 1
                    
                    # Log progress every 10 batches
                    if i % (self.batch_size * 10) == 0:
                        logger.info(f"Processed {processed_count} chunks...")
                        
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {e}")
                    error_count += len(batch_chunks)
                    continue
            
            # Update processing log
            processing_time = time.time() - start_time
            self._update_processing_log(
                log_id, 
                "success" if error_count == 0 else "partial_success",
                processing_time,
                processed_count,
                f"Processed: {processed_count}, Errors: {error_count}"
            )
            
            logger.info(f"Text embedding completed: {processed_count} processed, {error_count} errors")
            
            return {
                "processed": processed_count,
                "errors": error_count,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self._update_processing_log(log_id, "failed", 0, 0, str(e))
            logger.error(f"Text embedding failed: {e}")
            raise
    
    def create_embedding_for_query(self, query: str, query_type: str = "query") -> np.ndarray:
        """
        Tạo embedding cho search query
        
        Args:
            query: Search query
            query_type: Type of query (query, passage)
            
        Returns:
            Query embedding vector
        """
        # Preprocess query
        if "e5" in self.model_name.lower():
            # E5 models need query prefix
            query = f"query: {query}"
        
        # Create embedding
        if self.model_type == "sentence_transformer":
            embedding = self.model.encode(
                [query], 
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
        else:
            # Manual approach for single query
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embedding = embedding.cpu().numpy()[0]
        
        return embedding
    
    def _create_processing_log(self, process_type: str, status: str) -> str:
        """Create processing log entry"""
        try:
            with Session(self.engine) as session:
                log = ProcessingLog(
                    process_type=process_type,
                    status=status,
                    source_file="text_chunks"
                )
                session.add(log)
                session.commit()
                return log.log_id
        except Exception as e:
            logger.error(f"Error creating processing log: {e}")
            return ""
    
    def _update_processing_log(self, log_id: str, status: str, 
                             processing_time: float, items_processed: int, 
                             error_message: str = None):
        """Update processing log"""
        try:
            with Session(self.engine) as session:
                log = session.query(ProcessingLog).filter(
                    ProcessingLog.log_id == log_id
                ).first()
                
                if log:
                    log.status = status
                    log.processing_time = processing_time
                    log.items_processed = items_processed
                    if error_message:
                        log.error_message = error_message
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Error updating processing log: {e}")

# Usage example
if __name__ == "__main__":
    # Database configuration
    DB_URL = "postgresql://username:password@localhost/education_db"
    
    # Initialize service
    embedding_service = TextEmbeddingService(
        db_url=DB_URL,
        model_name="intfloat/multilingual-e5-base",  # Hoặc "vinai/phobert-base"
        batch_size=32
    )
    
    # Process text chunks
    results = embedding_service.process_text_chunks(limit=1000)
    print(f"Processing results: {results}")