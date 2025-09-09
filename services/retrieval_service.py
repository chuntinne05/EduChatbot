import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np

from models.retrieval_models import (
    MultimodalQuery, RetrievalResponse, SearchResult, 
    SearchIntent, DiversityConfig
)
from services.query_processor import QueryProcessor
from services.vector_search import VectorSearchEngine
from services.reranker import RerankerService
from services.embedding_service import TextEmbeddingService
from config.settings import settings
logger = logging.getLogger(__name__)

class RetrievalService:
    """Main retrieval service orchestrating the pipeline"""
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.vector_search = VectorSearchEngine()
        self.reranker = RerankerService()
        self.embedding_service = TextEmbeddingService(
            db_url=settings.DATABASE_URL,
            model_name="intfloat/multilingual-e5-base",
            batch_size=32
        )
        
    async def retrieve(
        self, 
        query: MultimodalQuery,
        diversity_config: Optional[DiversityConfig] = None
    ) -> RetrievalResponse:
        """Main retrieval pipeline"""
        
        start_time = time.time()
        
        # Step 1: Query analysis
        if query.text_query:
            query_intent = self.query_processor.analyze_query_intent(query.text_query)
            processed_query = self.query_processor.preprocess_query(query.text_query)
            filters = self.query_processor.extract_filters(query.text_query)
        else:
            # Image-only query (TODO: implement image query analysis)
            query_intent = SearchIntent()
            processed_query = ""
            filters = {}
        
        search_stages = {}
        all_results = []
        
        try:
            # Step 2: Stage 1 - Semantic Search (top 100)
            stage1_results = await self._stage1_semantic_search(
                query, query_intent, processed_query, filters
            )
            search_stages["stage1_semantic"] = len(stage1_results)
            
            # Step 3: Stage 2 - Reranking (top 20)
            if stage1_results and query.text_query:
                stage2_results = await self._stage2_reranking(
                    query.text_query, stage1_results
                )
            else:
                stage2_results = stage1_results[:20]
            
            search_stages["stage2_rerank"] = len(stage2_results)
            
            # Step 4: Stage 3 - Diversity Filtering (top 10)
            final_results = await self._stage3_diversity_filtering(
                stage2_results, query.max_results, diversity_config
            )
            search_stages["stage3_diversity"] = len(final_results)
            
            all_results = final_results
            
        except Exception as e:
            logger.error(f"Error in retrieval pipeline: {e}")
            all_results = []
        
        processing_time = time.time() - start_time
        
        return RetrievalResponse(
            query_intent=query_intent,
            results=all_results,
            total_found=len(all_results),
            processing_time=processing_time,
            search_stages=search_stages
        )
    
    async def _stage1_semantic_search(
        self,
        query: MultimodalQuery,
        query_intent: SearchIntent,
        processed_query: str,
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Stage 1: Semantic vector search"""
        
        results = []
        
        # Text query processing
        if query.text_query and query.include_text:
            # Get text embedding
            import asyncio
            text_embedding = await asyncio.to_thread(
                self.embedding_service.create_embedding_for_query,
                processed_query
            )
            if query_intent.needs_images and query.include_images:
                # Cross-modal search
                cross_modal_results = await self.vector_search.cross_modal_search(
                    query.text_query,
                    text_embedding,
                    top_k=100
                )
                results.extend(cross_modal_results)
            else:
                # Text-only search
                text_results = await self.vector_search.semantic_search_text(
                    text_embedding,
                    top_k=100,
                    filters=filters
                )
                results.extend(text_results)
        
        # Image query processing (if implemented)
        if query.image_query and query.include_images:
            # TODO: Implement image query processing
            pass
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_results = []
        for result in results:
            if result.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.id)
        
        return unique_results[:100]
    
    async def _stage2_reranking(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Stage 2: Reranking với cross-encoder"""
        
        if not results:
            return results
        
        reranked_results = await self.reranker.advanced_rerank(
            query, 
            results, 
            top_k=20,
            diversify=False  # Diversity sẽ được áp dụng ở stage 3
        )
        
        return reranked_results
    
    async def _stage3_diversity_filtering(
        self,
        results: List[SearchResult],
        max_results: int,
        diversity_config: Optional[DiversityConfig] = None
    ) -> List[SearchResult]:
        """Stage 3: Diversity filtering"""
        
        if not results:
            return results
        
        if not diversity_config:
            diversity_config = DiversityConfig()
        
        # Apply diversity constraints
        diverse_results = self.reranker.apply_diversity_filter(
            results,
            top_k=max_results,
            max_same_source=diversity_config.max_same_source,
            max_same_category=diversity_config.max_same_category
        )
        
        # Apply semantic diversity
        if diversity_config.similarity_threshold < 1.0:
            diverse_results = self.reranker.calculate_semantic_diversity(
                diverse_results,
                similarity_threshold=diversity_config.similarity_threshold
            )
        
        return diverse_results[:max_results]
    
    async def search_with_query_expansion(
        self,
        original_query: str,
        max_results: int = 10,
        include_images: bool = True
    ) -> RetrievalResponse:
        """Search với query expansion"""
        
        # Analyze và expand query
        query_intent = self.query_processor.analyze_query_intent(original_query)
        expanded_query = query_intent.expanded_query
        
        # Create multimodal query
        multimodal_query = MultimodalQuery(
            text_query=expanded_query,
            max_results=max_results,
            include_images=include_images,
            include_text=True
        )
        
        return await self.retrieve(multimodal_query)
    
    async def search_similar_content(
        self,
        content_id: str,
        content_type: str,
        max_results: int = 10
    ) -> List[SearchResult]:
        """Tìm nội dung tương tự"""
        
        try:
            if content_type == "text":
                # Get text embedding từ database
                embedding = await self._get_text_embedding_by_id(content_id)
                if embedding:
                    results = await self.vector_search.semantic_search_text(
                        embedding, top_k=max_results + 1  # +1 để loại bỏ chính nó
                    )
                    # Remove the original content
                    return [r for r in results if r.id != content_id][:max_results]
            
            elif content_type == "image":
                # Get image embedding từ database
                embedding = await self._get_image_embedding_by_id(content_id)
                if embedding:
                    results = await self.vector_search.semantic_search_images(
                        embedding, top_k=max_results + 1
                    )
                    return [r for r in results if r.id != content_id][:max_results]
            
        except Exception as e:
            logger.error(f"Error in similar content search: {e}")
        
        return []
    
    async def _get_text_embedding_by_id(self, text_id: str) -> Optional[List[float]]:
        """Get text embedding from database"""
        # Implementation to fetch embedding from database
        # This should be implemented based on your database schema
        pass
    
    async def _get_image_embedding_by_id(self, image_id: str) -> Optional[List[float]]:
        """Get image embedding from database"""
        # Implementation to fetch embedding from database
        pass
    
    async def batch_retrieve(
        self,
        queries: List[MultimodalQuery]
    ) -> List[RetrievalResponse]:
        """Batch retrieval cho multiple queries"""
        
        tasks = [self.retrieve(query) for query in queries]
        return await asyncio.gather(*tasks)
    
    def get_search_statistics(self, response: RetrievalResponse) -> Dict[str, Any]:
        """Get search statistics"""
        
        stats = {
            "total_results": response.total_found,
            "processing_time": response.processing_time,
            "stages": response.search_stages,
            "content_distribution": {
                "text": len([r for r in response.results if r.content_type.value == "text"]),
                "image": len([r for r in response.results if r.content_type.value == "image"]),
            },
            "avg_similarity_score": sum(r.similarity_score for r in response.results) / len(response.results) if response.results else 0,
            "avg_fusion_score": sum(r.fusion_score for r in response.results) / len(response.results) if response.results else 0
        }
        
        return stats