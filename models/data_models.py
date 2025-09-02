"""
Pydantic models cho data validation và API responses.
Định nghĩa structure cho input/output data.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import uuid

class EntitiesModel(BaseModel):
    """Model cho entities extracted từ text"""
    schools: List[str] = Field(default_factory=list)
    majors: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)

class TextChunkMetadata(BaseModel):
    """Model cho metadata của text chunk"""
    category: Optional[str] = None
    school: Optional[str] = None
    major: Optional[str] = None
    location: Optional[str] = None
    year: Optional[int] = None
    
    @validator('year')
    def validate_year(cls, v):
        if v and (v < 2020 or v > 2030):
            raise ValueError('Year must be between 2020 and 2030')
        return v

class TextChunkCreate(BaseModel):
    """Model cho creating text chunk"""
    content: str = Field(..., min_length=10)
    source_doc: str
    page_range: Optional[List[int]] = Field(None, min_items=2, max_items=2)
    metadata: TextChunkMetadata = Field(default_factory=TextChunkMetadata)
    entities: EntitiesModel = Field(default_factory=EntitiesModel)
    
    @validator('page_range')
    def validate_page_range(cls, v):
        if v and len(v) == 2 and v[0] > v[1]:
            raise ValueError('Start page must be less than or equal to end page')
        return v

class TextChunkResponse(BaseModel):
    """Model cho text chunk response"""
    chunk_id: str
    content: str
    source_doc: str
    page_range: Optional[List[int]]
    metadata: TextChunkMetadata
    entities: EntitiesModel
    related_images: List[str] = Field(default_factory=list)
    embedding_exists: bool = Field(default=False)
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ImageMetadata(BaseModel):
    """Model cho metadata của image"""
    category: Optional[str] = None  # campus, classroom, dormitory, activity
    school: Optional[str] = None
    location: Optional[str] = None
    
    @validator('category')
    def validate_category(cls, v):
        valid_categories = ['campus', 'classroom', 'dormitory', 'activity', 'building', 'lab', 'library', 'other']
        if v and v not in valid_categories:
            raise ValueError(f'Category must be one of: {valid_categories}')
        return v

class ImageCreate(BaseModel):
    """Model cho creating image"""
    file_path: str
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    metadata: ImageMetadata = Field(default_factory=ImageMetadata)
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    file_format: Optional[str] = None

class ImageResponse(BaseModel):
    """Model cho image response"""
    image_id: str
    file_path: str
    s3_url: Optional[str]
    caption: Optional[str]
    ocr_text: Optional[str]
    metadata: ImageMetadata
    width: Optional[int]
    height: Optional[int]
    file_size: Optional[int]
    file_format: Optional[str]
    related_text_chunks: List[str] = Field(default_factory=list)
    embedding_exists: bool = Field(default=False)
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class TextImagePair(BaseModel):
    """Model cho text-image relationship"""
    text_chunk_id: str
    image_id: str
    relation_score: float = Field(ge=0.0, le=1.0)
    
    @validator('relation_score')
    def validate_score(cls, v):
        return round(v, 3)  # Round to 3 decimal places

class ProcessingStatus(BaseModel):
    """Model cho processing status"""
    process_type: str
    source_file: str
    status: str  # success, failed, processing
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    items_processed: int = 0
    created_at: datetime

class BatchProcessingRequest(BaseModel):
    """Model cho batch processing request"""
    documents_path: Optional[str] = None
    images_path: Optional[str] = None
    process_text: bool = True
    process_images: bool = True
    create_embeddings: bool = True
    create_links: bool = True
    batch_size: int = Field(default=32, ge=1, le=100)

class BatchProcessingResponse(BaseModel):
    """Model cho batch processing response"""
    job_id: str
    status: str
    total_documents: int = 0
    total_images: int = 0
    processed_text_chunks: int = 0
    processed_images: int = 0
    created_links: int = 0
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    errors: List[str] = Field(default_factory=list)

class SearchQuery(BaseModel):
    """Model cho search query"""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    categories: Optional[List[str]] = None
    schools: Optional[List[str]] = None
    majors: Optional[List[str]] = None
    
class SearchResult(BaseModel):
    """Model cho search result"""
    text_chunks: List[TextChunkResponse] = Field(default_factory=list)
    images: List[ImageResponse] = Field(default_factory=list)
    total_results: int
    query_time: float  # in seconds

class SchoolModel(BaseModel):
    """Model cho School"""
    school_id: Optional[str] = None
    name: str
    short_name: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    
    class Config:
        from_attributes = True

class MajorModel(BaseModel):
    """Model cho Major"""
    major_id: Optional[str] = None
    name: str
    field: Optional[str] = None
    
    class Config:
        from_attributes = True