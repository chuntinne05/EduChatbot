from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config.settings import settings  
from contextlib import asynccontextmanager

# Tạo engine
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

# Tạo session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

@asynccontextmanager
async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session

async def semantic_search_text(
    db: AsyncSession,
    query_embedding: List[float],
    similarity_threshold: float = 0.8,
    category: Optional[str] = None,
    location: Optional[str] = None,
    source_type: Optional[str] = None,
    limit: int = 10,
):
    params = {"embedding": query_embedding}
    filters = []
    if category:
        filters.append("tc.category = :category")
        params["category"] = category
    if location:
        filters.append("tc.location = :location")
        params["location"] = location
    if source_type:
        filters.append("tc.source_type = :source_type")
        params["source_type"] = source_type
    filters.append("tc.embedding <=> :embedding < :threshold")
    params["threshold"] = 1 - similarity_threshold
    filter_clause = " AND ".join(filters)

    query = text(f"""
        SELECT tc.id, tc.text, tc.embedding <=> :embedding as distance
        FROM text_chunks tc
        WHERE {filter_clause}
        ORDER BY distance
        LIMIT :limit
    """)
    params["limit"] = limit
    result = await db.execute(query, params)
    return result.fetchall()

async def semantic_search_images(
    db: AsyncSession,
    query_embedding: List[float],
    similarity_threshold: float = 0.8,
    category: Optional[str] = None,
    location: Optional[str] = None,
    limit: int = 10,
):
    params = {"embedding": query_embedding}
    filters = []
    if category:
        filters.append("ie.category = :category")
        params["category"] = category
    if location:
        filters.append("ie.location = :location")
        params["location"] = location
    filters.append("ie.embedding <=> :embedding < :threshold")
    params["threshold"] = 1 - similarity_threshold
    filter_clause = " AND ".join(filters)

    query = text(f"""
        SELECT ie.id, ie.url, ie.embedding <=> :embedding as distance
        FROM image_embeddings ie
        WHERE {filter_clause}
        ORDER BY distance
        LIMIT :limit
    """)
    params["limit"] = limit
    result = await db.execute(query, params)
    return result.fetchall()

async def find_related_images_for_text(
    db: AsyncSession,
    text_chunk_ids: List[int],
):
    query = text("""
        SELECT ie.*
        FROM image_embeddings ie
        JOIN chunk_image_links cml ON ie.id = cml.image_id
        WHERE cml.text_chunk_id = ANY(:text_ids)
    """)
    params = {"text_ids": text_chunk_ids}
    result = await db.execute(query, params)
    return result.fetchall()