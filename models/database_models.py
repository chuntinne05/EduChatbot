"""
SQLAlchemy models cho PostgreSQL database.
Định nghĩa schema cho text chunks, images, và relationships.
"""

from sqlalchemy import Column, String, Integer, Text, JSON, ARRAY, Float, DateTime, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime, timezone

Base = declarative_base()

# Association table cho many-to-many relationship giữa text_chunks và images
text_image_association = Table(
    'text_image_pairs',
    Base.metadata,
    Column('text_chunk_id', String, ForeignKey('text_chunks.chunk_id'), primary_key=True),
    Column('image_id', String, ForeignKey('images.image_id'), primary_key=True),
    Column('relation_score', Float, default=0.0),
    Column('created_at', DateTime, default=datetime.now(timezone.utc))
)

class TextChunk(Base):
    """Model cho text chunks từ documents"""
    __tablename__ = 'text_chunks'
    
    chunk_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=True)
    source_doc = Column(String, nullable=True)
    page_range = Column(ARRAY(Integer), nullable=True)  # [start_page, end_page]
    
    # Metadata as JSON
    category = Column(String, nullable=True, index=True)
    school = Column(String, nullable=True, index=True)
    major = Column(String, nullable=True, index=True)
    location = Column(String, nullable=True, index=True)
    year = Column(Integer, nullable=True, index=True)
    
    # Entities as JSON arrays
    entities = Column(JSON, nullable=True)  # {"schools": [...], "majors": [...], "locations": [...]}
    
    # Embedding vector (sử dụng pgvector extension)
    embedding = Column(Vector(768), nullable=True)  # 768 cho sentence-transformers
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    # Relationships
    related_images = relationship(
        "Image", 
        secondary=text_image_association, 
        back_populates="related_text_chunks"
    )
    
    def __repr__(self):
        return f"<TextChunk(id={self.chunk_id}, school={self.school}, category={self.category})>"

class Image(Base):
    """Model cho images"""
    __tablename__ = 'images'
    
    image_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(String, nullable=False)  # S3 path hoặc local path
    s3_url = Column(String, nullable=True)      # S3 public URL
    caption = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    
    # Metadata
    category = Column(String, nullable=True, index=True)  # campus, classroom, dormitory, activity
    school = Column(String, nullable=True, index=True)
    location = Column(String, nullable=True, index=True)
    
    # Image properties
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)  # in bytes
    file_format = Column(String, nullable=True)  # jpg, png, etc.
    
    # Embedding vector
    embedding = Column(Vector(768), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    # Relationships
    related_text_chunks = relationship(
        "TextChunk", 
        secondary=text_image_association, 
        back_populates="related_images"
    )
    
    def __repr__(self):
        return f"<Image(id={self.image_id}, school={self.school}, category={self.category})>"

class ProcessingLog(Base):
    """Log table để track processing history"""
    __tablename__ = 'processing_logs'
    
    log_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    process_type = Column(String, nullable=True)  # text_processing, image_processing, linking
    source_file = Column(String, nullable=True)
    status = Column(String, nullable=True)  # success, failed, processing
    error_message = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  # in seconds
    items_processed = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<ProcessingLog(type={self.process_type}, status={self.status})>"

class School(Base):
    """Master table cho schools để normalize data"""
    __tablename__ = 'schools'
    
    school_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True, index=True)
    short_name = Column(String, nullable=True)
    location = Column(String, nullable=True, index=True)
    website = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<School(name={self.name}, location={self.location})>"

class Major(Base):
    """Master table cho majors để normalize data"""
    __tablename__ = 'majors'
    
    major_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True, index=True)
    field = Column(String, nullable=True, index=True)  # STEM, Business, Arts, etc.
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<Major(name={self.name}, field={self.field})>"