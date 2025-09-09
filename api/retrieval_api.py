from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from typing import List, Optional, Dict, Any
import logging
import asyncio
import base64
from io import BytesIO

from models.retrieval_models import (
    MultimodalQuery, RetrievalResponse, SearchResult,
    DiversityConfig, QueryType
)
from services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
retrieval_service = RetrievalService()

@router.post("/search", response_model=RetrievalResponse)
async def multimodal_search(
    text_query: Optional[str] = None,
    max_results: int = Query(default=10, ge=1, le=50),
    include_images: bool = Query(default=True),
    include_text: bool = Query(default=True),
    diversity_max_same_source: int = Query(default=3, ge=1, le=10),
    diversity_max_same_category: int = Query(default=4, ge=1, le=10),
    diversity_similarity_threshold: float = Query(default=0.85, ge=0.0, le=1.0)
):
    """
    Multimodal search endpoint
    
    Args:
        text_query: Text query string
        max_results: Maximum number of results to return
        include_images: Whether to include image results
        include_text: Whether to include text results
        diversity_max_same_source: Max results from same source
        diversity_max_same_category: Max results from same category
        diversity_similarity_threshold: Threshold for semantic diversity
    """
    
    if not text_query and not include_images:
        raise HTTPException(
            status_code=400, 
            detail="At least text_query must be provided"
        )
    
    try:
        # Create query object
        query = MultimodalQuery(
            text_query=text_query,
            max_results=max_results,
            include_images=include_images,
            include_text=include_text
        )
        
        # Create diversity config
        diversity_config = DiversityConfig(
            max_same_source=diversity_max_same_source,
            max_same_category=diversity_max_same_category,
            similarity_threshold=diversity_similarity_threshold
        )
        
        # Execute search
        response = await retrieval_service.retrieve(query, diversity_config)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in multimodal search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/search/image", response_model=RetrievalResponse)
async def image_search(
    image_file: UploadFile = File(...),
    max_results: int = Query(default=10, ge=1, le=50),
    include_text: bool = Query(default=True)
):
    """
    Image-based search endpoint
    """
    
    try:
        # Read image file
        image_data = await image_file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode()
        
        # Create query
        query = MultimodalQuery(
            image_query=image_base64,
            max_results=max_results,
            include_images=True,
            include_text=include_text
        )
        
        # Execute search
        response = await retrieval_service.retrieve(query)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in image search: {e}")
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")

@router.get("/search/expanded")
async def expanded_search(
    query: str = Query(..., description="Original query to expand"),
    max_results: int = Query(default=10, ge=1, le=50),
    include_images: bool = Query(default=True)
):
    """
    Search with automatic query expansion
    """
    
    try:
        response = await retrieval_service.search_with_query_expansion(
            original_query=query,
            max_results=max_results,
            include_images=include_images
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in expanded search: {e}")
        raise HTTPException(status_code=500, detail=f"Expanded search failed: {str(e)}")

@router.get("/similar/{content_type}/{content_id}")
async def find_similar_content(
    content_type: str,
    content_id: str,
    max_results: int = Query(default=10, ge=1, le=50)
):
    """
    Find similar content based on existing content ID
    
    Args:
        content_type: Type of content ("text" or "image")
        content_id: ID of the reference content
        max_results: Maximum number of similar results
    """
    
    if content_type not in ["text", "image"]:
        raise HTTPException(
            status_code=400, 
            detail="content_type must be 'text' or 'image'"
        )
    
    try:
        results = await retrieval_service.search_similar_content(
            content_id=content_id,
            content_type=content_type,
            max_results=max_results
        )
        
        return {
            "reference_id": content_id,
            "reference_type": content_type,
            "similar_results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error finding similar content: {e}")
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")

@router.post("/search/batch")
async def batch_search(
    queries: List[Dict[str, Any]],
    max_batch_size: int = Query(default=10, ge=1, le=20)
):
    """
    Batch search for multiple queries
    
    Args:
        queries: List of query objects
        max_batch_size: Maximum number of queries in batch
    """
    
    if len(queries) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size cannot exceed {max_batch_size}"
        )
    
    try:
        # Convert dict queries to MultimodalQuery objects
        multimodal_queries = []
        for query_dict in queries:
            query = MultimodalQuery(**query_dict)
            multimodal_queries.append(query)
        
        # Execute batch search
        responses = await retrieval_service.batch_retrieve(multimodal_queries)
        
        return {
            "batch_size": len(queries),
            "responses": responses
        }
        
    except Exception as e:
        logger.error(f"Error in batch search: {e}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

@router.get("/stats/{search_id}")
async def get_search_statistics(search_id: str):
    """
    Get statistics for a search (placeholder for future implementation)
    """
    # This would require storing search history with IDs
    # For now, return a placeholder response
    return {
        "search_id": search_id,
        "message": "Search statistics endpoint - to be implemented with search history storage"
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        # Test basic functionality
        test_query = MultimodalQuery(
            text_query="test",
            max_results=1
        )
        
        # This is a very basic health check
        # In production, you might want to test actual database connectivity
        return {
            "status": "healthy",
            "service": "retrieval",
            "components": {
                "query_processor": "ok",
                "vector_search": "ok", 
                "reranker": "ok",
                "embedding_service": "ok"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Utility endpoints

@router.post("/analyze-query")
async def analyze_query_intent(query: str):
    """Analyze query intent (for debugging/development)"""
    
    try:
        intent = retrieval_service.query_processor.analyze_query_intent(query)
        return intent
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")

@router.get("/config/synonyms")
async def get_synonyms_config():
    """Get current synonyms configuration"""
    return retrieval_service.query_processor.synonyms

@router.get("/config/entity-mapping") 
async def get_entity_mapping_config():
    """Get current entity mapping configuration"""
    return retrieval_service.query_processor.entity_mapping