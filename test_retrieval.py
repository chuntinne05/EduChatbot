import asyncio
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.retrieval_models import MultimodalQuery, DiversityConfig
from services.retrieval_service import RetrievalService
from services.query_processor import QueryProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_query_processor():
    """Test query processor"""
    print("\n=== Testing Query Processor ===")
    
    processor = QueryProcessor()
    
    test_queries = [
        "Đại học Bách Khoa có những ngành nào?",
        "Học phí trường đại học kinh tế TP.HCM",
        "Hình ảnh campus đại học FPT",
        "Cơ sở vật chất trường y Hà Nội",
        "HUST tuyển sinh bằng những phương thức nào?",
        "Điểm tuyển sinh ngành KHMT trường ĐHBK TPHCM năm 2024"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # phân tích intent
        intent = processor.analyze_query_intent(query)
        print(f"Intent: needs_text={intent.needs_text}, needs_images={intent.needs_images}")
        print(f"Query type: {intent.query_type}")
        print(f"Entities: {intent.entities}")
        print(f"Normalized: {intent.normalized_entities}")
        print(f"Expanded: {intent.expanded_query}")
        
        # Process query
        processed = processor.preprocess_query(query)
        print(f"Processed: {processed}")
        
        # Extract filters
        filters = processor.extract_filters(query)
        print(f"Filters: {filters}")

async def test_retrieval_service():
    """Test full retrieval service"""
    print("\n=== Testing Retrieval Service ===")
    
    service = RetrievalService()
    
    # Test queries
    test_cases = [
        {
            "query": "Đại học Bách Khoa Hà Nội có những ngành gì?",
            "include_images": True,
            "max_results": 5
        },
        {
            "query": "Học phí đại học kinh tế",
            "include_images": False,
            "max_results": 10
        },
        {
            "query": "Hình ảnh campus trường đại học",
            "include_images": True,
            "max_results": 8
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Query: {test_case['query']}")
        
        # Create multimodal query
        query = MultimodalQuery(
            text_query=test_case['query'],
            max_results=test_case['max_results'],
            include_images=test_case['include_images'],
            include_text=True
        )
        
        try:
            # Execute retrieval
            response = await service.retrieve(query)
            
            print(f"Processing time: {response.processing_time:.3f}s")
            print(f"Total results: {response.total_found}")
            print(f"Search stages: {response.search_stages}")
            
            # Print first few results
            print("\nTop results:")
            for j, result in enumerate(response.results[:3]):
                print(f"  {j+1}. [{result.content_type.value}] {result.title[:50]}...")
                print(f"      Similarity: {result.similarity_score:.3f}")
                print(f"      Fusion: {result.fusion_score:.3f}")
                if result.image_url:
                    print(f"      Image: {result.image_url}")
            
            # Get statistics
            stats = service.get_search_statistics(response)
            print(f"\nStatistics: {stats}")
            
        except Exception as e:
            print(f"Error: {e}")
            logger.exception("Test failed")

async def test_batch_retrieval():
    """Test batch retrieval"""
    print("\n=== Testing Batch Retrieval ===")
    
    service = RetrievalService()
    
    # Multiple queries
    queries = [
        MultimodalQuery(
            text_query="Đại học Bách Khoa có ngành CNTT không?",
            max_results=5,
            include_images=True
        ),
        MultimodalQuery(
            text_query="Học phí trường kinh tế",
            max_results=5,
            include_images=False
        ),
        MultimodalQuery(
            text_query="Cơ sở vật chất trường y",
            max_results=5,
            include_images=True
        )
    ]
    
    try:
        responses = await service.batch_retrieve(queries)
        
        print(f"Batch processed {len(responses)} queries")
        
        for i, response in enumerate(responses):
            print(f"\nQuery {i+1}: {queries[i].text_query}")
            print(f"Results: {response.total_found}, Time: {response.processing_time:.3f}s")
            
    except Exception as e:
        print(f"Batch retrieval error: {e}")
        logger.exception("Batch test failed")

async def test_diversity_filtering():
    """Test diversity filtering"""
    print("\n=== Testing Diversity Filtering ===")
    
    service = RetrievalService()
    
    query = MultimodalQuery(
        text_query="Đại học công nghệ thông tin",
        max_results=10,
        include_images=True
    )
    
    # Test with different diversity configs
    diversity_configs = [
        DiversityConfig(
            max_same_source=2,
            max_same_category=3,
            similarity_threshold=0.9
        ),
        DiversityConfig(
            max_same_source=5,
            max_same_category=5,
            similarity_threshold=0.7
        )
    ]
    
    for i, config in enumerate(diversity_configs):
        print(f"\n--- Diversity Config {i+1} ---")
        print(f"Max same source: {config.max_same_source}")
        print(f"Max same category: {config.max_same_category}")
        print(f"Similarity threshold: {config.similarity_threshold}")
        
        try:
            response = await service.retrieve(query, config)
            
            print(f"Total results: {response.total_found}")
            
            # Analyze diversity
            sources = {}
            categories = {}
            for result in response.results:
                source = result.source_type
                category = result.metadata.get("category", "unknown")
                
                sources[source] = sources.get(source, 0) + 1
                categories[category] = categories.get(category, 0) + 1
            
            print(f"Source distribution: {sources}")
            print(f"Category distribution: {categories}")
            
        except Exception as e:
            print(f"Diversity test error: {e}")

def print_test_header(title):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

async def main():
    """Run all tests"""
    print("Starting Retrieval System Tests...")
    
    try:
        # Test individual components
        await test_query_processor()
        
        # Test main retrieval service
        await test_retrieval_service()
        
        # Test batch processing
        await test_batch_retrieval()
        
        # Test diversity filtering
        await test_diversity_filtering()
        
        print("\n" + "="*60)
        print(" All tests completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        logger.exception("Test suite error")

if __name__ == "__main__":
    asyncio.run(main())