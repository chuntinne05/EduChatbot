"""
Main Embedding Pipeline - Orchestrate toàn bộ quá trình tạo embedding
Chạy text embedding → image embedding → cross-modal linking
"""

import os
import sys
import logging
import argparse
import asyncio
from typing import Dict, Any
import json
from datetime import datetime, timezone
import time
from services.embedding_service import TextEmbeddingService
from services.image_embedding import ImageEmbeddingService
from services.linking_service import CrossModalLinkingService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    """Main pipeline để orchestrate embedding process"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize services
        self.text_service = TextEmbeddingService(
            db_url=config['database']['url'],
            model_name=config['models']['text_model'],
            batch_size=config['processing']['text_batch_size']
        )
        
        self.image_service = ImageEmbeddingService(
            db_url=config['database']['url'],
            clip_model_name=config['models']['clip_model'],
            text_model_name=config['models']['text_model'],
            batch_size=config['processing']['image_batch_size'],
            aws_access_key=config.get('aws', {}).get('access_key'),
            aws_secret_key=config.get('aws', {}).get('secret_key'),
            s3_bucket=config.get('aws', {}).get('bucket')
        )
        
        self.linking_service = CrossModalLinkingService(
            db_url=config['database']['url'],
            similarity_threshold=config['linking']['similarity_threshold'],
            metadata_weight=config['linking']['metadata_weight'],
            semantic_weight=config['linking']['semantic_weight'],
            max_links_per_text=config['linking']['max_links_per_text'],
            max_links_per_image=config['linking']['max_links_per_image']
        )
        
        logger.info("Embedding Pipeline initialized")
    
    def run_text_embedding(self) -> Dict[str, Any]:
        """
        Chạy text embedding process
        
        Returns:
            Processing results
        """
        logger.info("=== STARTING TEXT EMBEDDING PROCESS ===")
        start_time = time.time()
        
        try:
            results = self.text_service.process_text_chunks(
                limit=self.config['processing']['text_limit']
            )
            
            processing_time = time.time() - start_time
            results['total_time'] = processing_time
            
            logger.info(f"Text embedding completed in {processing_time:.2f}s")
            logger.info(f"Results: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise
    
    def run_image_embedding(self) -> Dict[str, Any]:
        """
        Chạy image embedding process
        
        Returns:
            Processing results
        """
        logger.info("=== STARTING IMAGE EMBEDDING PROCESS ===")
        start_time = time.time()
        
        try:
            results = self.image_service.process_image_embeddings(
                limit=self.config['processing']['image_limit']
            )
            
            processing_time = time.time() - start_time
            results['total_time'] = processing_time
            
            logger.info(f"Image embedding completed in {processing_time:.2f}s")
            logger.info(f"Results: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            raise
    
    def run_cross_modal_linking(self) -> Dict[str, Any]:
        """
        Chạy cross-modal linking process
        
        Returns:
            Processing results
        """
        logger.info("=== STARTING CROSS-MODAL LINKING PROCESS ===")
        start_time = time.time()
        
        try:
            results = self.linking_service.process_cross_modal_linking(
                limit=self.config['processing']['linking_limit']
            )
            
            processing_time = time.time() - start_time
            results['total_time'] = processing_time
            
            logger.info(f"Cross-modal linking completed in {processing_time:.2f}s")
            logger.info(f"Results: {results}")
            
            # Get statistics
            stats = self.linking_service.get_linking_statistics()
            results['statistics'] = stats
            logger.info(f"Linking statistics: {stats}")
            
            return results
            
        except Exception as e:
            logger.error(f"Cross-modal linking failed: {e}")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline: text → image → linking
        
        Returns:
            Complete pipeline results
        """
        pipeline_start = time.time()
        pipeline_results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'text_embedding': {},
            'image_embedding': {},
            'cross_modal_linking': {},
            'total_time': 0,
            'success': False
        }
        
        try:
            logger.info("🚀 STARTING FULL EMBEDDING PIPELINE 🚀")
            
            # Step 1: Text Embedding
            if self.config['steps']['text_embedding']:
                text_results = self.run_text_embedding()
                pipeline_results['text_embedding'] = text_results
            else:
                logger.info("Skipping text embedding (disabled in config)")
            
            # Step 2: Image Embedding  
            if self.config['steps']['image_embedding']:
                image_results = self.run_image_embedding()
                pipeline_results['image_embedding'] = image_results
            else:
                logger.info("Skipping image embedding (disabled in config)")
            
            # Step 3: Cross-modal Linking
            if self.config['steps']['cross_modal_linking']:
                linking_results = self.run_cross_modal_linking()
                pipeline_results['cross_modal_linking'] = linking_results
            else:
                logger.info("Skipping cross-modal linking (disabled in config)")
            
            # Pipeline completion
            total_time = time.time() - pipeline_start
            pipeline_results['total_time'] = total_time
            pipeline_results['success'] = True
            pipeline_results['end_time'] = datetime.now(timezone.utc).isoformat()
            
            logger.info("✅ FULL PIPELINE COMPLETED SUCCESSFULLY ✅")
            logger.info(f"Total pipeline time: {total_time:.2f}s")
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now(timezone.utc).isoformat()
            
            logger.error("❌ PIPELINE FAILED ❌")
            logger.error(f"Error: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Lưu pipeline results ra file
        
        Args:
            results: Pipeline results
            filename: Output filename
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embedding_pipeline_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        "database": {
            "url": "postgresql://trinhtrantrungtin@localhost/admission_chatbot"
        },
        "models": {
            "text_model": "intfloat/multilingual-e5-base",
            "clip_model": "openai/clip-vit-base-patch32"
        },
        "processing": {
            "text_batch_size": 32,
            "image_batch_size": 16,
            "text_limit": 1000,
            "image_limit": 500,
            "linking_limit": 1000
        },
        "linking": {
            "similarity_threshold": 0.3,
            "metadata_weight": 0.4,
            "semantic_weight": 0.6,
            "max_links_per_text": 5,
            "max_links_per_image": 10
        },
        "steps": {
            "text_embedding": True,
            "image_embedding": True,
            "cross_modal_linking": True
        },
        "aws": {
            "access_key": None,
            "secret_key": None,
            "bucket": None
        }
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Education Data Embedding Pipeline")
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    parser.add_argument('--step', type=str, choices=['text', 'image', 'linking', 'all'],
                       default='all', help='Which step to run')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("Default configuration created: config.json")
        return
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logger.warning(f"Config file {args.config} not found, using default config")
        config = create_default_config()
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline(config)
    
    # Run specified step(s)
    try:
        if args.step == 'text':
            results = {'text_embedding': pipeline.run_text_embedding()}
        elif args.step == 'image':
            results = {'image_embedding': pipeline.run_image_embedding()}
        elif args.step == 'linking':
            results = {'cross_modal_linking': pipeline.run_cross_modal_linking()}
        else:  # all
            results = pipeline.run_full_pipeline()
        
        # Save results
        if args.output:
            pipeline.save_results(results, args.output)
        else:
            pipeline.save_results(results)
        
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()