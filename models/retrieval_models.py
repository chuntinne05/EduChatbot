from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np

class QueryType(str, Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    MULTIMODAL = "multimodal"

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    BOTH = "both"

class SearchIntent(BaseModel):
    """Phân tích intent của query"""
    needs_text: bool = True
    needs_images: bool = False
    query_type: QueryType = QueryType.TEXT_ONLY
    entities: List[str] = Field(default_factory=list)
    normalized_entities: Dict[str, str] = Field(default_factory=dict)
    expanded_query: str = ""

class SearchResult(BaseModel):
    """Kết quả tìm kiếm cho một item"""
    id: str
    content_type: ContentType
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    fusion_score: float = 0.0
    image_url: Optional[str] = None
    source_type: str = ""  # text_chunk, image, etc.

class MultimodalQuery(BaseModel):
    """Query đầu vào có thể là text hoặc image"""
    text_query: Optional[str] = None
    image_query: Optional[str] = None  # base64 hoặc file path
    max_results: int = 10
    include_images: bool = True
    include_text: bool = True

class RetrievalResponse(BaseModel):
    """Response từ retrieval system"""
    query_intent: SearchIntent
    results: List[SearchResult]
    total_found: int
    processing_time: float
    search_stages: Dict[str, int] = Field(default_factory=dict)  # stage1: 100, stage2: 20, etc.

class RerankInput(BaseModel):
    """Input cho reranking model"""
    query: str
    candidates: List[Dict[str, Any]]
    top_k: int = 20

class DiversityConfig(BaseModel):
    """Cấu hình cho diversity filtering"""
    max_same_source: int = 3
    max_same_category: int = 4
    similarity_threshold: float = 0.85
    diversity_weight: float = 0.3