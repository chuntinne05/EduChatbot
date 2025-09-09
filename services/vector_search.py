import numpy as np
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, and_, or_
from pgvector.sqlalchemy import Vector
import json
import numpy as np
from config.database import get_async_session
from models.database_models import TextChunk, Image, text_image_association
from models.retrieval_models import SearchResult, ContentType
from services.embedding_service import TextEmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """Vector search engine sử dụng pgvector"""
    
    def __init__(self):
        self.embedding_service = TextEmbeddingService(
            db_url=settings.DATABASE_URL,
            model_name="intfloat/multilingual-e5-base",
            batch_size=32
        )
        
    async def semantic_search_text(
        self, 
        query_embedding: List[float], 
        top_k: int = 100,
        filters: Dict[str, Any] = None,
        similarity_threshold: float = 0.5
    ) -> List[SearchResult]:
        """Tìm kiếm semantic cho text chunks"""
        
        async with get_async_session() as session:
            # Build base query
            query_parts = [
                "SELECT ",
                "tc.chunk_id, tc.content, tc.source_doc, tc.category, tc.school, tc.major, tc.location, tc.year, tc.entities, ",
                "tc.embedding <=> :embedding as distance ",
                "FROM text_chunks tc "
            ]
            
            params = {"embedding": f"[{','.join(str(float(x)) for x in query_embedding)}]"}
            
            # Add filters
            where_conditions = []
            if filters:
                if filters.get("category"):
                    where_conditions.append("tc.category = :category")
                    params["category"] = filters["category"]
                
                if filters.get("location"):
                    where_conditions.append("tc.location ILIKE :location")
                    params["location"] = f"%{filters['location']}%"
                
                if filters.get("source_type"):
                    where_conditions.append("tc.source_type = :source_type")
                    params["source_type"] = filters["source_type"]
            
            # Add similarity threshold
            where_conditions.append("tc.embedding <=> :embedding < :threshold")
            params["threshold"] = 1 - similarity_threshold
            
            if where_conditions:
                query_parts.append("WHERE " + " AND ".join(where_conditions))
            
            query_parts.extend([
                f"ORDER BY distance ASC ",
                f"LIMIT {top_k}"
            ])
            
            query_sql = " ".join(query_parts)
            
            try:
                result = await session.execute(text(query_sql), params)
                rows = result.fetchall()
                
                search_results = []
                for row in rows:
                    similarity_score = 1 - float(row.distance)  # Convert distance to similarity
                    
                    metadata = {
                        "category": row.category,
                        "school": row.school,
                        "major": row.major,
                        "location": row.location,
                        "year": row.year,
                        "entities": row.entities,
                    }
                    
                    search_result = SearchResult(
                        id=row.chunk_id,
                        content_type=ContentType.TEXT,
                        title=row.source_doc or "",
                        content=row.content,
                        metadata=metadata,
                        similarity_score=similarity_score,
                        source_type="text"
                    )
                    search_results.append(search_result)
                        
                return search_results
                
            except Exception as e:
                logger.error(f"Error in semantic search: {e}")
                return []
    
    async def semantic_search_images(
        self, 
        query_embedding: List[float], 
        top_k: int = 50,
        filters: Dict[str, Any] = None,
        similarity_threshold: float = 0.5
    ) -> List[SearchResult]:
        """Tìm kiếm semantic cho images"""
        
        async with get_async_session() as session:
            query_parts = [
                "SELECT ",
                "ie.image_id, ie.file_path, ie.s3_url, ie.caption, ie.category, ",
                "ie.embedding <=> :embedding as distance ",
                "FROM images ie "
            ]
            
            params = {"embedding": f"[{','.join(str(float(x)) for x in query_embedding)}]"}
            
            # Add filters
            where_conditions = []
            if filters:
                if filters.get("category"):
                    where_conditions.append("ie.category = :category")
                    params["category"] = filters["category"]
                
                if filters.get("location"):
                    where_conditions.append("ie.metadata->>'location' ILIKE :location")
                    params["location"] = f"%{filters['location']}%"
            
            # Add similarity threshold
            where_conditions.append("ie.embedding <=> :embedding < :threshold")
            params["threshold"] = 1 - similarity_threshold
            
            if where_conditions:
                query_parts.append("WHERE " + " AND ".join(where_conditions))
            
            query_parts.extend([
                f"ORDER BY distance ASC ",
                f"LIMIT {top_k}"
            ])
            
            query_sql = " ".join(query_parts)
            
            try:
                result = await session.execute(text(query_sql), params)
                rows = result.fetchall()
                
                search_results = []
                for row in rows:
                    similarity_score = 1 - float(row.distance)
                    
                    metadata = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata or {}
                    
                    search_result = SearchResult(
                        id=row.image_id,
                        content_type=ContentType.IMAGE,
                        title=row.caption or "",
                        content=row.caption or row.ocr_text or "",
                        metadata={
                            "category": row.category,
                            "school": row.school,
                            "location": row.location,
                            "file_format": row.file_format
                        },
                        similarity_score=similarity_score,
                        image_url=row.s3_url or row.file_path,
                        source_type="image"
                    )
                    search_results.append(search_result)
                
                return search_results
                
            except Exception as e:
                logger.error(f"Error in image search: {e}")
                return []
    
    async def cross_modal_search(
        self,
        text_query: str,
        query_embedding: List[float],
        top_k: int = 30,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> List[SearchResult]:
        """Cross-modal search: text query tìm cả text và image"""
        
        # Tạo embedding cho image search từ text
        # image_embedding = self.embedding_service.get_clip_text_embedding(text_query)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        image_embedding_np = await asyncio.to_thread(self.embedding_service.create_embedding_for_query, text_query)
        image_embedding = image_embedding_np.tolist() if isinstance(image_embedding_np, np.ndarray) else image_embedding_np    
        # Parallel search
        text_results_task = self.semantic_search_text(query_embedding, top_k=int(top_k * 0.7))
        image_results_task = self.semantic_search_images(image_embedding, top_k=int(top_k * 0.5))
        
        text_results, image_results = await asyncio.gather(
            text_results_task, 
            image_results_task
        )
        
        # Combine và reweight scores
        all_results = []
        
        for result in text_results:
            result.similarity_score = result.similarity_score * text_weight
            all_results.append(result)
        
        for result in image_results:
            result.similarity_score = result.similarity_score * image_weight
            all_results.append(result)
        
        # Sort by weighted score
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return all_results[:top_k]
    
    async def find_related_images_for_text(
        self,
        text_chunk_ids: List[str]
    ) -> Dict[str, List[SearchResult]]:
        """Tìm images liên quan đến text chunks thông qua cross_modal_links"""
        
        if not text_chunk_ids:
            return {}
        
        async with get_async_session() as session:
            query_sql = """
                SELECT 
                    tip.text_chunk_id,
                    ie.image_id, ie.file_path, ie.s3_url, ie.caption, ie.category,
                    tip.relation_score
                FROM text_image_pairs tip
                JOIN images ie ON tip.image_id = ie.image_id
                WHERE tip.text_chunk_id = ANY(:text_ids)
                AND tip.relation_score > 0.6
                ORDER BY tip.text_chunk_id, tip.relation_score DESC
            """
            
            try:
                result = await session.execute(text(query_sql), {"text_ids": text_chunk_ids})
                rows = result.fetchall()
                
                related_images = {}
                for row in rows:
                    text_id = row.text_chunk_id
                    
                    if text_id not in related_images:
                        related_images[text_id] = []
                    
                    metadata = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata or {}
                    
                    image_result = SearchResult(
                        id=row.image_id,
                        content_type=ContentType.IMAGE,
                        title=row.caption or "",
                        content=row.caption or row.ocr_text or "",
                        metadata={"category": row.category},
                        similarity_score=float(row.relation_score),
                        image_url=row.s3_url or row.file_path,
                        source_type="linked_image"
                    )
                    
                    related_images[text_id].append(image_result)
                
                return related_images
                
            except Exception as e:
                logger.error(f"Error finding related images: {e}")
                return {}
    
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        include_images: bool = True,
        top_k: int = 100,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Hybrid search kết hợp semantic + cross-modal"""
        
        if include_images:
            # Cross-modal search
            results = await self.cross_modal_search(
                query, 
                query_embedding, 
                top_k=top_k
            )
        else:
            # Text-only search
            results = await self.semantic_search_text(
                query_embedding, 
                top_k=top_k,
                filters=filters
            )
        
        # Tìm related images cho text results
        if include_images:
            text_results = [r for r in results if r.content_type == ContentType.TEXT]
            if text_results:
                text_ids = [r.id for r in text_results[:10]]  # Chỉ lấy top 10 để tìm related images
                related_images = await self.find_related_images_for_text(text_ids)
                
                # Thêm related images vào metadata
                for result in text_results:
                    if result.id in related_images:
                        result.metadata["related_images"] = [
                            {
                                "id": img.id,
                                "url": img.image_url,
                                "caption": img.title,
                                "similarity": img.similarity_score
                            }
                            for img in related_images[result.id][:2]  # Chỉ lấy 2 ảnh liên quan nhất
                        ]
        
        return results