"""
Cross-modal Linking Service - Tạo liên kết giữa text chunks và images
Sử dụng semantic similarity và metadata matching
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text, and_
from models.database_models import TextChunk, Image, ProcessingLog, text_image_association
import time
from datetime import datetime, timezone
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossModalLinkingService:
    """Service để tạo liên kết giữa text và images"""
    
    def __init__(self, 
                 db_url: str,
                 similarity_threshold: float = 0.3,
                 metadata_weight: float = 0.4,
                 semantic_weight: float = 0.6,
                 max_links_per_text: int = 5,
                 max_links_per_image: int = 10):
        """
        Initialize cross-modal linking service
        
        Args:
            db_url: PostgreSQL connection URL
            similarity_threshold: Minimum similarity score for linking
            metadata_weight: Weight for metadata similarity
            semantic_weight: Weight for semantic similarity
            max_links_per_text: Maximum images linked to one text chunk
            max_links_per_image: Maximum text chunks linked to one image
        """
        self.db_url = db_url
        self.similarity_threshold = similarity_threshold
        self.metadata_weight = metadata_weight
        self.semantic_weight = semantic_weight
        self.max_links_per_text = max_links_per_text
        self.max_links_per_image = max_links_per_image
        
        # Initialize database engine
        self.engine = create_engine(db_url)
        
        logger.info("Cross-modal Linking Service initialized")
    
    def calculate_metadata_similarity(self, text_chunk: TextChunk, image: Image) -> float:
        """
        Tính similarity dựa trên metadata (school, category, location)
        
        Args:
            text_chunk: TextChunk object
            image: Image object
            
        Returns:
            Metadata similarity score (0-1)
        """
        score = 0.0
        factors = 0
        
        # School matching (highest weight)
        if text_chunk.school and image.school:
            if text_chunk.school.lower() == image.school.lower():
                score += 0.5
            factors += 0.5
        
        # Category matching
        if text_chunk.category and image.category:
            # Define category relationships
            category_relations = {
                'co_so_vat_chat': ['campus', 'classroom', 'dormitory', 'facility'],
                'sinh_vien': ['activity', 'student_life', 'dormitory'],
                'tuyen_sinh': ['campus', 'ceremony', 'activity'],
                'chuong_trinh': ['classroom', 'laboratory', 'library']
            }
            
            text_cat = text_chunk.category.lower()
            img_cat = image.category.lower()
            
            if text_cat == img_cat:
                score += 0.3
            elif text_cat in category_relations and img_cat in category_relations.get(text_cat, []):
                score += 0.2
            elif img_cat in category_relations and text_cat in category_relations.get(img_cat, []):
                score += 0.2
                
            factors += 0.3
        
        # Location matching
        if text_chunk.location and image.location:
            if text_chunk.location.lower() == image.location.lower():
                score += 0.2
            factors += 0.2
        
        # Normalize score
        if factors > 0:
            score = score / factors
        
        return min(score, 1.0)
    
    def calculate_semantic_similarity(self, text_embedding: np.ndarray, image_embedding: np.ndarray) -> float:
        """
        Tính semantic similarity giữa text và image embeddings
        
        Args:
            text_embedding: Text embedding vector
            image_embedding: Image embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Ensure embeddings are same dimension
            if len(text_embedding) != len(image_embedding):
                # Resize to match (take minimum dimension)
                min_dim = min(len(text_embedding), len(image_embedding))
                text_embedding = text_embedding[:min_dim]
                image_embedding = image_embedding[:min_dim]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([text_embedding], [image_embedding])[0][0]
            
            # Convert to 0-1 range (cosine similarity can be -1 to 1)
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_combined_similarity(self, text_chunk: TextChunk, image: Image) -> float:
        """
        Tính combined similarity score
        
        Args:
            text_chunk: TextChunk object  
            image: Image object
            
        Returns:
            Combined similarity score (0-1)
        """
        # Metadata similarity
        metadata_sim = self.calculate_metadata_similarity(text_chunk, image)
        
        # Semantic similarity
        semantic_sim = 0.0
        if text_chunk.embedding is not None and image.embedding is not None:
            text_emb = np.array(text_chunk.embedding)
            img_emb = np.array(image.embedding)
            if text_emb.size > 0 and img_emb.size > 0:
                semantic_sim = self.calculate_semantic_similarity(text_emb, img_emb)
        
        # Combined score
        combined_score = (
            self.metadata_weight * metadata_sim + 
            self.semantic_weight * semantic_sim
        )
        
        return combined_score
    
    def find_text_image_pairs(self, limit: int = 1000) -> List[Tuple[str, str, float]]:
        """
        Tìm các cặp text-image có similarity cao
        
        Args:
            limit: Maximum number of text chunks to process
            
        Returns:
            List of (text_chunk_id, image_id, similarity_score)
        """
        pairs = []
        
        with Session(self.engine) as session:
            # Get text chunks with embeddings
            text_chunks = session.query(TextChunk).filter(
                TextChunk.embedding.isnot(None)
            ).limit(limit).all()
            
            # Get images with embeddings  
            images = session.query(Image).filter(
                Image.embedding.isnot(None)
            ).all()
            
            logger.info(f"Finding pairs for {len(text_chunks)} texts and {len(images)} images")
            
            # Calculate similarities
            for text_chunk in tqdm(text_chunks, desc="Processing text chunks"):
                chunk_pairs = []
                
                for image in images:
                    similarity = self.calculate_combined_similarity(text_chunk, image)
                    
                    if similarity >= self.similarity_threshold:
                        chunk_pairs.append((text_chunk.chunk_id, image.image_id, similarity))
                
                # Sort by similarity and take top matches
                chunk_pairs.sort(key=lambda x: x[2], reverse=True)
                pairs.extend(chunk_pairs[:self.max_links_per_text])
            
        return pairs
    
    def save_text_image_links(self, pairs: List[Tuple[str, str, float]]) -> int:
        """
        Lưu text-image links vào database
        
        Args:
            pairs: List of (text_chunk_id, image_id, similarity_score)
            
        Returns:
            Number of links saved
        """
        saved_count = 0
        
        with Session(self.engine) as session:
            try:
                # Clear existing links (optional - có thể comment nếu muốn giữ old links)
                session.execute(text("DELETE FROM text_image_pairs"))
                
                # Insert new links
                for text_id, image_id, score in tqdm(pairs, desc="Saving links"):
                    # Check if link already exists
                    existing = session.execute(
                        text("SELECT 1 FROM text_image_pairs WHERE text_chunk_id = :text_id AND image_id = :image_id"),
                        {"text_id": text_id, "image_id": image_id}
                    ).fetchone()
                    
                    if not existing:
                        session.execute(
                            text("""
                                INSERT INTO text_image_pairs (text_chunk_id, image_id, relation_score, created_at)
                                VALUES (:text_id, :image_id, :score, :created_at)
                            """),
                            {
                                "text_id": text_id,
                                "image_id": image_id,
                                "score": score,
                                "created_at": datetime.now(timezone.utc)
                            }
                        )
                        saved_count += 1
                
                session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving links: {e}")
                raise
        
        return saved_count
    
    def process_cross_modal_linking(self, limit: int = 1000) -> Dict[str, int]:
        """
        Xử lý toàn bộ quá trình linking
        
        Args:
            limit: Maximum text chunks to process
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        
        # Create processing log
        log_id = self._create_processing_log("cross_modal_linking", "processing")
        
        try:
            logger.info("Starting cross-modal linking process...")
            
            # Find text-image pairs
            pairs = self.find_text_image_pairs(limit)
            logger.info(f"Found {len(pairs)} potential links")
            
            # Save links to database
            saved_count = self.save_text_image_links(pairs)
            logger.info(f"Saved {saved_count} links")
            
            # Update processing log
            processing_time = time.time() - start_time
            self._update_processing_log(
                log_id,
                "success",
                processing_time,
                saved_count,
                f"Created {saved_count} cross-modal links"
            )
            
            return {
                "pairs_found": len(pairs),
                "links_saved": saved_count,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self._update_processing_log(log_id, "failed", 0, 0, str(e))
            logger.error(f"Cross-modal linking failed: {e}")
            raise
    
    def get_related_images_for_text(self, text_chunk_id: str, limit: int = 5) -> List[Dict]:
        """
        Lấy images liên quan đến một text chunk
        
        Args:
            text_chunk_id: Text chunk ID
            limit: Maximum number of images to return
            
        Returns:
            List of image info with similarity scores
        """
        with Session(self.engine) as session:
            query = """
                SELECT i.image_id, i.file_path, i.s3_url, i.caption, i.category, 
                       tip.relation_score
                FROM images i
                JOIN text_image_pairs tip ON i.image_id = tip.image_id
                WHERE tip.text_chunk_id = :text_id
                ORDER BY tip.relation_score DESC
                LIMIT :limit
            """
            
            results = session.execute(
                text(query),
                {"text_id": text_chunk_id, "limit": limit}
            ).fetchall()
            
            return [
                {
                    "image_id": row.image_id,
                    "file_path": row.file_path,
                    "s3_url": row.s3_url,
                    "caption": row.caption,
                    "category": row.category,
                    "similarity_score": row.relation_score
                }
                for row in results
            ]
    
    def get_related_texts_for_image(self, image_id: str, limit: int = 10) -> List[Dict]:
        """
        Lấy text chunks liên quan đến một image
        
        Args:
            image_id: Image ID
            limit: Maximum number of text chunks to return
            
        Returns:
            List of text info with similarity scores
        """
        with Session(self.engine) as session:
            query = """
                SELECT tc.chunk_id, tc.content, tc.category, tc.school, tc.major,
                       tip.relation_score
                FROM text_chunks tc
                JOIN text_image_pairs tip ON tc.chunk_id = tip.text_chunk_id
                WHERE tip.image_id = :image_id
                ORDER BY tip.relation_score DESC
                LIMIT :limit
            """
            
            results = session.execute(
                text(query),
                {"image_id": image_id, "limit": limit}
            ).fetchall()
            
            return [
                {
                    "chunk_id": row.chunk_id,
                    "content": row.content[:200] + "..." if len(row.content) > 200 else row.content,
                    "category": row.category,
                    "school": row.school,
                    "major": row.major,
                    "similarity_score": row.relation_score
                }
                for row in results
            ]
    
    def update_existing_links(self, recalculate: bool = False) -> Dict[str, int]:
        """
        Cập nhật các links hiện có (recompute similarity scores)
        
        Args:
            recalculate: Whether to recalculate all similarity scores
            
        Returns:
            Update statistics
        """
        start_time = time.time()
        updated_count = 0
        
        with Session(self.engine) as session:
            if recalculate:
                # Get all existing links
                query = """
                    SELECT tip.text_chunk_id, tip.image_id, tip.relation_score
                    FROM text_image_pairs tip
                    JOIN text_chunks tc ON tip.text_chunk_id = tc.chunk_id
                    JOIN images i ON tip.image_id = i.image_id
                    WHERE tc.embedding IS NOT NULL AND i.embedding IS NOT NULL
                """
                
                existing_links = session.execute(text(query)).fetchall()
                
                logger.info(f"Recalculating {len(existing_links)} existing links...")
                
                for link in tqdm(existing_links, desc="Updating links"):
                    # Get full objects
                    text_chunk = session.query(TextChunk).filter(
                        TextChunk.chunk_id == link.text_chunk_id
                    ).first()
                    
                    image = session.query(Image).filter(
                        Image.image_id == link.image_id
                    ).first()
                    
                    if text_chunk and image:
                        # Recalculate similarity
                        new_score = self.calculate_combined_similarity(text_chunk, image)
                        
                        # Update if score changed significantly or below threshold
                        if abs(new_score - link.relation_score) > 0.05 or new_score < self.similarity_threshold:
                            if new_score >= self.similarity_threshold:
                                # Update score
                                session.execute(
                                    text("""
                                        UPDATE text_image_pairs 
                                        SET relation_score = :score 
                                        WHERE text_chunk_id = :text_id AND image_id = :image_id
                                    """),
                                    {
                                        "score": new_score,
                                        "text_id": text_chunk.chunk_id,
                                        "image_id": image.image_id
                                    }
                                )
                                updated_count += 1
                            else:
                                # Remove low-quality link
                                session.execute(
                                    text("""
                                        DELETE FROM text_image_pairs 
                                        WHERE text_chunk_id = :text_id AND image_id = :image_id
                                    """),
                                    {
                                        "text_id": text_chunk.chunk_id,
                                        "image_id": image.image_id
                                    }
                                )
                
                session.commit()
        
        processing_time = time.time() - start_time
        
        return {
            "updated_count": updated_count,
            "processing_time": processing_time
        }
    
    def get_linking_statistics(self) -> Dict[str, int]:
        """
        Lấy thống kê về cross-modal links
        
        Returns:
            Statistics dictionary
        """
        with Session(self.engine) as session:
            stats = {}
            
            # Total links
            stats['total_links'] = session.execute(
                text("SELECT COUNT(*) FROM text_image_pairs")
            ).scalar()
            
            # Text chunks with images
            stats['texts_with_images'] = session.execute(
                text("SELECT COUNT(DISTINCT text_chunk_id) FROM text_image_pairs")
            ).scalar()
            
            # Images with texts
            stats['images_with_texts'] = session.execute(
                text("SELECT COUNT(DISTINCT image_id) FROM text_image_pairs")
            ).scalar()
            
            # Average similarity score
            avg_score = session.execute(
                text("SELECT AVG(relation_score) FROM text_image_pairs")
            ).scalar()
            stats['average_similarity'] = float(avg_score) if avg_score else 0.0
            
            # Links by category
            category_stats = session.execute(
                text("""
                    SELECT tc.category, COUNT(*) as count
                    FROM text_image_pairs tip
                    JOIN text_chunks tc ON tip.text_chunk_id = tc.chunk_id
                    GROUP BY tc.category
                    ORDER BY count DESC
                """)
            ).fetchall()
            
            stats['links_by_category'] = {
                row.category or 'unknown': row.count 
                for row in category_stats
            }
            
            return stats
    
    def _create_processing_log(self, process_type: str, status: str) -> str:
        """Create processing log entry"""
        try:
            with Session(self.engine) as session:
                log = ProcessingLog(
                    process_type=process_type,
                    status=status,
                    source_file="cross_modal_linking"
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
    linking_service = CrossModalLinkingService(
        db_url=DB_URL,
        similarity_threshold=0.3,
        metadata_weight=0.4,
        semantic_weight=0.6
    )
    
    # Process cross-modal linking
    results = linking_service.process_cross_modal_linking(limit=1000)
    print(f"Linking results: {results}")
    
    # Get statistics
    stats = linking_service.get_linking_statistics()
    print(f"Linking statistics: {stats}")