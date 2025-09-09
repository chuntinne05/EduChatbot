import logging
from typing import List, Dict, Any, Tuple
import asyncio
import numpy as np
from sentence_transformers import CrossEncoder
import torch
from transformers import AutoTokenizer, AutoModel
import re

from models.retrieval_models import SearchResult, RerankInput

logger = logging.getLogger(__name__)

class RerankerService:
    """Service để rerank kết quả search"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cross_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load reranking models"""
        try:
            # Sử dụng cross-encoder cho reranking
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Hoặc multilingual model
            self.cross_encoder = CrossEncoder(model_name, device=self.device)
            logger.info(f"Loaded cross-encoder: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load reranking models: {e}")
            self.cross_encoder = None
    
    async def rerank_results(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: int = 20
    ) -> List[SearchResult]:
        """Rerank search results"""
        
        if not results:
            return results
        
        if len(results) <= top_k:
            return results
        
        # Nếu không có cross-encoder, return results gốc
        if self.cross_encoder is None:
            return results[:top_k]
        
        try:
            # Chuẩn bị input cho cross-encoder
            pairs = []
            for result in results:
                # Combine title và content để rerank
                text_content = f"{result.title} {result.content}".strip()
                pairs.append([query, text_content])
            
            # Batch rerank để tối ưu performance
            batch_size = 32
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.cross_encoder.predict(batch_pairs)
                all_scores.extend(batch_scores.tolist())
            
            # Cập nhật rerank scores
            for i, result in enumerate(results):
                result.rerank_score = float(all_scores[i])
            
            # Sort theo rerank score
            reranked_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Fallback to original results
            return results[:top_k]
    
    def calculate_text_relevance(self, query: str, text: str) -> float:
        """Tính relevance score dựa trên rule-based"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        score = 0.0
        
        # Exact match bonus
        if query_lower in text_lower:
            score += 0.3
        
        # Word overlap
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        overlap = len(query_words.intersection(text_words))
        overlap_ratio = overlap / len(query_words) if query_words else 0
        score += overlap_ratio * 0.4
        
        # Length penalty for very long texts
        text_length = len(text.split())
        if text_length > 200:
            score *= 0.9
        
        # Education domain bonus
        education_terms = [
            "đại học", "trường", "ngành", "tuyển sinh", "học phí", 
            "chương trình", "môn học", "giảng viên", "sinh viên"
        ]
        
        for term in education_terms:
            if term in text_lower:
                score += 0.05
        
        return min(score, 1.0)
    
    def calculate_image_relevance(self, query: str, image_result: SearchResult) -> float:
        """Tính relevance cho image results"""
        score = 0.0
        
        # Caption relevance
        if image_result.content:
            score += self.calculate_text_relevance(query, image_result.content) * 0.7
        
        # Category relevance
        category = image_result.metadata.get("category", "")
        if category:
            query_lower = query.lower()
            category_mapping = {
                "campus": ["trường", "khuôn viên", "cơ sở"],
                "classroom": ["lớp học", "phòng học", "học tập"],
                "dormitory": ["ký túc xá", "nơi ở", "chỗ ở"],
                "activity": ["hoạt động", "sinh hoạt", "sự kiện"],
                "facility": ["cơ sở vật chất", "trang thiết bị", "phòng lab"]
            }
            
            for cat, keywords in category_mapping.items():
                if cat.lower() == category.lower():
                    for keyword in keywords:
                        if keyword in query_lower:
                            score += 0.2
                            break
        
        return min(score, 1.0)
    
    async def advanced_rerank(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: int = 20,
        diversify: bool = True
    ) -> List[SearchResult]:
        """Advanced reranking với nhiều factors"""
        
        if not results:
            return results
        
        # Stage 1: Cross-encoder reranking
        reranked = await self.rerank_results(query, results, top_k * 2)
        
        # Stage 2: Rule-based scoring
        for result in reranked:
            if result.content_type.value == "text":
                rule_score = self.calculate_text_relevance(query, result.content)
            else:
                rule_score = self.calculate_image_relevance(query, result)
            
            # Combine scores
            result.fusion_score = (
                result.rerank_score * 0.6 + 
                rule_score * 0.3 + 
                result.similarity_score * 0.1
            )
        
        # Stage 3: Diversity filtering (if enabled)
        if diversify:
            final_results = self.apply_diversity_filter(reranked, top_k)
        else:
            final_results = sorted(reranked, key=lambda x: x.fusion_score, reverse=True)[:top_k]
        
        return final_results
    
    def apply_diversity_filter(
        self, 
        results: List[SearchResult], 
        top_k: int,
        max_same_source: int = 3,
        max_same_category: int = 4
    ) -> List[SearchResult]:
        """Áp dụng diversity filtering"""
        
        selected_results = []
        source_counts = {}
        category_counts = {}
        
        # Sort by fusion score
        sorted_results = sorted(results, key=lambda x: x.fusion_score, reverse=True)
        
        for result in sorted_results:
            if len(selected_results) >= top_k:
                break
            
            source = result.source_type
            category = result.metadata.get("category", "unknown")
            
            # Check diversity constraints
            if source_counts.get(source, 0) >= max_same_source:
                continue
            
            if category_counts.get(category, 0) >= max_same_category:
                continue
            
            # Add result
            selected_results.append(result)
            source_counts[source] = source_counts.get(source, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Nếu không đủ results do diversity constraint, fill với remaining
        if len(selected_results) < top_k:
            remaining = [r for r in sorted_results if r not in selected_results]
            selected_results.extend(remaining[:top_k - len(selected_results)])
        
        return selected_results
    
    def calculate_semantic_diversity(
        self, 
        results: List[SearchResult],
        similarity_threshold: float = 0.85
    ) -> List[SearchResult]:
        """Remove semantically similar results"""
        
        if len(results) <= 1:
            return results
        
        # Simple approach: compare content overlap
        filtered_results = [results[0]]  # Always keep the top result
        
        for i in range(1, len(results)):
            current = results[i]
            is_diverse = True
            
            for selected in filtered_results:
                # Calculate content similarity
                similarity = self._calculate_content_similarity(
                    current.content, 
                    selected.content
                )
                
                if similarity > similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                filtered_results.append(current)
        
        return filtered_results
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Tính similarity giữa 2 text (simple Jaccard similarity)"""
        
        if not text1 or not text2:
            return 0.0
        
        # Tokenize and normalize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def batch_rerank(
        self,
        queries_and_results: List[Tuple[str, List[SearchResult]]],
        top_k: int = 20
    ) -> List[List[SearchResult]]:
        """Batch reranking cho multiple queries"""
        
        tasks = []
        for query, results in queries_and_results:
            task = self.advanced_rerank(query, results, top_k)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)