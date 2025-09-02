"""
Image Embedding Service - Tạo embedding cho images
Sử dụng CLIP model và combine với caption/OCR text
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image as PILImage
import requests
from io import BytesIO
import base64
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from models.database_models import Image, ProcessingLog
import time
from datetime import datetime, timezone
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEmbeddingService:
    """Service để tạo và lưu image embeddings"""
    
    def __init__(self, 
                 db_url: str,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 text_model_name: str = "intfloat/multilingual-e5-base",
                 batch_size: int = 16,
                 aws_access_key: str = None,
                 aws_secret_key: str = None,
                 s3_bucket: str = None):
        """
        Initialize image embedding service
        
        Args:
            db_url: PostgreSQL connection URL
            clip_model_name: CLIP model name
            text_model_name: Text model for captions
            batch_size: Batch size for processing
            aws_access_key: AWS access key for S3
            aws_secret_key: AWS secret key for S3
            s3_bucket: S3 bucket name
        """
        self.db_url = db_url
        self.clip_model_name = clip_model_name
        self.text_model_name = text_model_name
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # S3 configuration
        self.s3_bucket = s3_bucket
        if aws_access_key and aws_secret_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            self.s3_client = None
        
        # Initialize database engine
        self.engine = create_engine(db_url)
        
        # Load models
        self._load_models()
        
        logger.info(f"Image Embedding Service initialized on {self.device}")
    
    def _load_models(self):
        """Load CLIP and text models"""
        try:
            # Load CLIP model
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            
            # Load text model for caption embeddings
            self.text_model = SentenceTransformer(self.text_model_name, device=self.device)
            
            logger.info(f"Models loaded: {self.clip_model_name}, {self.text_model_name}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def load_image(self, image_path: str) -> Optional[PILImage.Image]:
        """
        Load image from various sources (local, S3, URL)
        
        Args:
            image_path: Path to image (local, S3, or URL)
            
        Returns:
            PIL Image object or None
        """
        try:
            if image_path.startswith('http'):
                # URL image
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = PILImage.open(BytesIO(response.content))
                
            elif image_path.startswith('s3://'):
                # S3 image
                if not self.s3_client:
                    logger.error("S3 client not configured")
                    return None
                
                # Parse S3 path
                path_parts = image_path[5:].split('/', 1)
                bucket = path_parts[0]
                key = path_parts[1] if len(path_parts) > 1 else ''
                
                # Download from S3
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                image = PILImage.open(BytesIO(response['Body'].read()))
                
            else:
                # Local file
                if not os.path.exists(image_path):
                    logger.error(f"Image file not found: {image_path}")
                    return None
                image = PILImage.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def create_image_embedding(self, image: PILImage.Image) -> np.ndarray:
        """
        Tạo embedding cho một image bằng CLIP
        
        Args:
            image: PIL Image object
            
        Returns:
            Image embedding vector
        """
        try:
            # Preprocess image
            inputs = self.clip_processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize for cosine similarity
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
                embedding = image_features.cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating image embedding: {e}")
            raise
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """
        Tạo embedding cho text (caption/OCR) để combine với image
        
        Args:
            text: Caption or OCR text
            
        Returns:
            Text embedding vector
        """
        try:
            if not text or text.strip() == "":
                return np.zeros(768)  # Return zero vector for empty text
            
            # Preprocess text
            text = f"passage: {text.strip()}"  # E5 prefix
            
            # Generate embedding
            embedding = self.text_model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating text embedding: {e}")
            return np.zeros(768)
    
    def create_multimodal_embedding(self, 
                                  image: PILImage.Image, 
                                  caption: str = None, 
                                  ocr_text: str = None,
                                  fusion_method: str = "weighted") -> np.ndarray:
        """
        Tạo multimodal embedding combine image + text
        
        Args:
            image: PIL Image object
            caption: Image caption
            ocr_text: OCR extracted text
            fusion_method: Method to combine embeddings (weighted, concat, attention)
            
        Returns:
            Combined embedding vector
        """
        try:
            # Get image embedding (512-dim from CLIP)
            image_emb = self.create_image_embedding(image)
            
            # Combine caption and OCR text
            combined_text = ""
            if caption:
                combined_text += caption
            if ocr_text:
                if combined_text:
                    combined_text += " " + ocr_text
                else:
                    combined_text = ocr_text
            
            # Get text embedding (768-dim from e5)
            text_emb = self.create_text_embedding(combined_text)
            
            if fusion_method == "weighted":
                # Weighted average (adjust weights based on text availability)
                if combined_text.strip():
                    # Resize embeddings to same dimension (768)
                    image_emb_resized = np.pad(image_emb, (0, 768 - len(image_emb)), 'constant')
                    
                    # Weight: 0.7 for image, 0.3 for text
                    combined_emb = 0.7 * image_emb_resized + 0.3 * text_emb
                else:
                    # Only image, resize to 768
                    combined_emb = np.pad(image_emb, (0, 768 - len(image_emb)), 'constant')
                
            elif fusion_method == "concat":
                # Simple concatenation
                combined_emb = np.concatenate([image_emb, text_emb[:256]])  # Keep 768 total
                
            else:  # Default to weighted
                image_emb_resized = np.pad(image_emb, (0, 768 - len(image_emb)), 'constant')
                combined_emb = 0.7 * image_emb_resized + 0.3 * text_emb if combined_text.strip() else image_emb_resized
            
            # Normalize final embedding
            combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-8)
            
            return combined_emb
            
        except Exception as e:
            logger.error(f"Error creating multimodal embedding: {e}")
            raise
    
    def get_unprocessed_images(self, limit: int = 500) -> List[Image]:
        """
        Lấy images chưa có embedding
        
        Args:
            limit: Maximum number of images to process
            
        Returns:
            List of Image objects
        """
        with Session(self.engine) as session:
            images = session.query(Image).filter(
                Image.embedding.is_(None)
            ).limit(limit).all()
            
            return images
    
    def update_image_embedding(self, image_id: str, embedding: np.ndarray):
        """
        Cập nhật embedding cho một image
        
        Args:
            image_id: Image ID
            embedding: Embedding vector
        """
        try:
            with Session(self.engine) as session:
                image = session.query(Image).filter(
                    Image.image_id == image_id
                ).first()
                
                if image:
                    # Convert numpy array to list for PostgreSQL
                    image.embedding = embedding.tolist()
                    image.updated_at = datetime.now(timezone.utc)
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Error updating image {image_id}: {e}")
            raise
    
    def process_image_batch(self, images: List[Image]) -> Tuple[int, int]:
        """
        Xử lý một batch images
        
        Args:
            images: List of Image objects
            
        Returns:
            (processed_count, error_count)
        """
        processed_count = 0
        error_count = 0
        
        for image in images:
            try:
                # Load image
                pil_image = self.load_image(image.file_path)
                if pil_image is None:
                    error_count += 1
                    continue
                
                # Create multimodal embedding
                embedding = self.create_multimodal_embedding(
                    pil_image, 
                    image.caption, 
                    image.ocr_text
                )
                
                # Update database
                self.update_image_embedding(image.image_id, embedding)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing image {image.image_id}: {e}")
                error_count += 1
                continue
        
        return processed_count, error_count
    
    def process_image_embeddings(self, limit: int = 500) -> Dict[str, int]:
        """
        Xử lý images để tạo embeddings
        
        Args:
            limit: Maximum images to process
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        total_processed = 0
        total_errors = 0
        
        # Create processing log
        log_id = self._create_processing_log("image_embedding", "processing")
        
        try:
            # Get unprocessed images
            images = self.get_unprocessed_images(limit)
            
            if not images:
                logger.info("No images to process")
                return {"processed": 0, "errors": 0}
            
            logger.info(f"Processing {len(images)} images...")
            
            # Process in batches
            for i in tqdm(range(0, len(images), self.batch_size), desc="Processing batches"):
                batch_images = images[i:i+self.batch_size]
                
                processed, errors = self.process_image_batch(batch_images)
                total_processed += processed
                total_errors += errors
                
                # Log progress
                if i % (self.batch_size * 5) == 0:
                    logger.info(f"Processed {total_processed} images...")
            
            # Update processing log
            processing_time = time.time() - start_time
            self._update_processing_log(
                log_id,
                "success" if total_errors == 0 else "partial_success",
                processing_time,
                total_processed,
                f"Processed: {total_processed}, Errors: {total_errors}"
            )
            
            logger.info(f"Image embedding completed: {total_processed} processed, {total_errors} errors")
            
            return {
                "processed": total_processed,
                "errors": total_errors,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self._update_processing_log(log_id, "failed", 0, 0, str(e))
            logger.error(f"Image embedding failed: {e}")
            raise
    
    def create_embedding_for_image_query(self, query: str) -> np.ndarray:
        """
        Tạo embedding cho image search query
        
        Args:
            query: Text query to search images
            
        Returns:
            Query embedding vector (768-dim để match với stored embeddings)
        """
        # Use text model to create query embedding
        query_emb = self.create_text_embedding(f"query: {query}")
        
        return query_emb
    
    def _create_processing_log(self, process_type: str, status: str) -> str:
        """Create processing log entry"""
        try:
            with Session(self.engine) as session:
                log = ProcessingLog(
                    process_type=process_type,
                    status=status,
                    source_file="images"
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
    embedding_service = ImageEmbeddingService(
        db_url=DB_URL,
        clip_model_name="openai/clip-vit-base-patch32",
        text_model_name="intfloat/multilingual-e5-base",
        batch_size=16
    )
    
    # Process image embeddings
    results = embedding_service.process_image_embeddings(limit=500)
    print(f"Processing results: {results}")