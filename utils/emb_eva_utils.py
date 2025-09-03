"""
Embedding Utilities và Evaluation Tools
Cung cấp các tools để đánh giá chất lượng embeddings và debug
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from models.database_models import TextChunk, Image, text_image_association
import json

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingEvaluator:
    """Tool để đánh giá chất lượng embeddings"""
    
    def __init__(self, db_url: str):
        """
        Initialize evaluator
        
        Args:
            db_url: PostgreSQL connection URL
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
    
    def evaluate_text_embeddings(self, sample_size: int = 100) -> Dict[str, float]:
        """
        Đánh giá chất lượng text embeddings
        
        Args:
            sample_size: Number of samples to evaluate
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating text embeddings...")
        
        with Session(self.engine) as session:
            # Get sample of text chunks với embeddings
            chunks = session.query(TextChunk).filter(
                TextChunk.embedding.isnot(None)
            ).limit(sample_size).all()
            
            if len(chunks) < 10:
                logger.warning(f"Only {len(chunks)} chunks available for evaluation")
                return {}
            
            # Extract embeddings
            embeddings = np.array([chunk.embedding for chunk in chunks])
            
            # Evaluation metrics
            metrics = {}
            
            # 1. Embedding distribution analysis
            metrics['mean_norm'] = float(np.mean([np.linalg.norm(emb) for emb in embeddings]))
            metrics['std_norm'] = float(np.std([np.linalg.norm(emb) for emb in embeddings]))
            
            # 2. Similarity distribution
            similarities = []
            for i in range(min(50, len(embeddings))):
                for j in range(i+1, min(50, len(embeddings))):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(sim)
            
            metrics['mean_similarity'] = float(np.mean(similarities))
            metrics['std_similarity'] = float(np.std(similarities))
            
            # 3. Category clustering quality
            if len(set(chunk.category for chunk in chunks if chunk.category)) > 1:
                category_metrics = self._evaluate_category_clustering(chunks, embeddings)
                metrics.update(category_metrics)
            
            # 4. School clustering quality
            if len(set(chunk.school for chunk in chunks if chunk.school)) > 1:
                school_metrics = self._evaluate_school_clustering(chunks, embeddings)
                metrics.update(school_metrics)
            
            logger.info(f"Text embedding evaluation completed: {metrics}")
            return metrics
    
    def evaluate_image_embeddings(self, sample_size: int = 100) -> Dict[str, float]:
        """
        Đánh giá chất lượng image embeddings
        
        Args:
            sample_size: Number of samples to evaluate
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating image embeddings...")
        
        with Session(self.engine) as session:
            # Get sample of images với embeddings
            images = session.query(Image).filter(
                Image.embedding.isnot(None)
            ).limit(sample_size).all()
            
            if len(images) < 10:
                logger.warning(f"Only {len(images)} images available for evaluation")
                return {}
            
            # Extract embeddings
            embeddings = np.array([img.embedding for img in images])
            
            # Evaluation metrics
            metrics = {}
            
            # 1. Embedding distribution
            metrics['mean_norm'] = float(np.mean([np.linalg.norm(emb) for emb in embeddings]))
            metrics['std_norm'] = float(np.std([np.linalg.norm(emb) for emb in embeddings]))
            
            # 2. Category clustering (campus, classroom, etc.)
            if len(set(img.category for img in images if img.category)) > 1:
                category_metrics = self._evaluate_category_clustering(images, embeddings, 'category')
                metrics.update({f"image_{k}": v for k, v in category_metrics.items()})
            
            logger.info(f"Image embedding evaluation completed: {metrics}")
            return metrics
    
    def evaluate_cross_modal_links(self) -> Dict[str, float]:
        """
        Đánh giá chất lượng cross-modal links
        
        Returns:
            Link evaluation metrics
        """
        logger.info("Evaluating cross-modal links...")
        
        with Session(self.engine) as session:
            # Get link statistics
            query = """
                SELECT 
                    tip.relation_score,
                    tc.category as text_category,
                    i.category as image_category,
                    tc.school as text_school,
                    i.school as image_school
                FROM text_image_pairs tip
                JOIN text_chunks tc ON tip.text_chunk_id = tc.chunk_id
                JOIN images i ON tip.image_id = i.image_id
            """
            
            results = session.execute(text(query)).fetchall()
            
            if not results:
                logger.warning("No cross-modal links found")
                return {}
            
            metrics = {}
            
            # Score distribution
            scores = [row.relation_score for row in results]
            metrics['mean_score'] = float(np.mean(scores))
            metrics['std_score'] = float(np.std(scores))
            metrics['min_score'] = float(np.min(scores))
            metrics['max_score'] = float(np.max(scores))
            
            # Category alignment
            category_matches = sum(
                1 for row in results 
                if row.text_category and row.image_category and 
                self._categories_related(row.text_category, row.image_category)
            )
            metrics['category_alignment'] = category_matches / len(results)
            
            # School alignment
            school_matches = sum(
                1 for row in results
                if row.text_school and row.image_school and
                row.text_school.lower() == row.image_school.lower()
            )
            metrics['school_alignment'] = school_matches / len(results)
            
            # Score distribution by category
            score_by_category = {}
            for row in results:
                cat = row.text_category or 'unknown'
                if cat not in score_by_category:
                    score_by_category[cat] = []
                score_by_category[cat].append(row.relation_score)
            
            metrics['score_by_category'] = {
                cat: float(np.mean(scores))
                for cat, scores in score_by_category.items()
            }
            
            logger.info(f"Cross-modal evaluation completed: {metrics}")
            return metrics
    
    def _evaluate_category_clustering(self, items: List, embeddings: np.ndarray, attr: str = 'category') -> Dict[str, float]:
        """
        Đánh giá clustering quality theo category
        
        Args:
            items: List of objects (TextChunk hoặc Image)
            embeddings: Corresponding embeddings
            attr: Attribute to cluster by
            
        Returns:
            Clustering metrics
        """
        try:
            # Get categories
            categories = [getattr(item, attr) for item in items]
            unique_categories = list(set(cat for cat in categories if cat))
            
            if len(unique_categories) < 2:
                return {}
            
            # Calculate within-category vs between-category similarities
            within_similarities = []
            between_similarities = []
            
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    
                    if categories[i] == categories[j] and categories[i] is not None:
                        within_similarities.append(sim)
                    elif categories[i] != categories[j] and categories[i] and categories[j]:
                        between_similarities.append(sim)
            
            metrics = {}
            if within_similarities:
                metrics[f'{attr}_within_similarity'] = float(np.mean(within_similarities))
            if between_similarities:
                metrics[f'{attr}_between_similarity'] = float(np.mean(between_similarities))
            
            # Clustering quality (higher within, lower between is better)
            if within_similarities and between_similarities:
                metrics[f'{attr}_clustering_quality'] = (
                    np.mean(within_similarities) - np.mean(between_similarities)
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating clustering: {e}")
            return {}
    
    def _evaluate_school_clustering(self, chunks: List[TextChunk], embeddings: np.ndarray) -> Dict[str, float]:
        """Đánh giá clustering theo school"""
        return self._evaluate_category_clustering(chunks, embeddings, 'school')
    
    def _categories_related(self, text_cat: str, image_cat: str) -> bool:
        """
        Kiểm tra xem 2 categories có related không
        
        Args:
            text_cat: Text category
            image_cat: Image category
            
        Returns:
            Whether categories are related
        """
        if not text_cat or not image_cat:
            return False
        
        # Define category relationships
        relations = {
            'co_so_vat_chat': ['campus', 'classroom', 'dormitory', 'facility', 'library'],
            'sinh_vien': ['activity', 'student_life', 'dormitory', 'campus'],
            'tuyen_sinh': ['campus', 'ceremony', 'activity', 'facility'],
            'chuong_trinh': ['classroom', 'laboratory', 'library', 'facility']
        }
        
        text_cat = text_cat.lower()
        image_cat = image_cat.lower()
        
        # Exact match
        if text_cat == image_cat:
            return True
        
        # Related categories
        for cat, related in relations.items():
            if text_cat == cat and image_cat in related:
                return True
            if image_cat == cat and text_cat in related:
                return True
        
        return False
    
    def visualize_embeddings(self, 
                           sample_size: int = 200, 
                           save_path: str = "embedding_visualization.png"):
        """
        Tạo visualization cho embeddings bằng PCA
        
        Args:
            sample_size: Number of samples to visualize
            save_path: Path to save plot
        """
        logger.info("Creating embedding visualization...")
        
        with Session(self.engine) as session:
            # Get text embeddings
            text_chunks = session.query(TextChunk).filter(
                TextChunk.embedding.isnot(None)
            ).limit(sample_size).all()
            
            # Get image embeddings
            images = session.query(Image).filter(
                Image.embedding.isnot(None)
            ).limit(sample_size).all()
            
            if len(text_chunks) < 10 or len(images) < 10:
                logger.warning("Not enough data for visualization")
                return
            
            # Prepare data
            text_embeddings = np.array([chunk.embedding for chunk in text_chunks])
            image_embeddings = np.array([img.embedding for img in images])
            
            # Combine embeddings
            all_embeddings = np.vstack([text_embeddings, image_embeddings])
            
            # PCA reduction
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(all_embeddings)
            
            # Prepare labels
            text_labels = [f"Text-{chunk.category or 'unknown'}" for chunk in text_chunks]
            image_labels = [f"Image-{img.category or 'unknown'}" for img in images]
            all_labels = text_labels + image_labels
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot text embeddings
            text_reduced = reduced_embeddings[:len(text_chunks)]
            plt.scatter(text_reduced[:, 0], text_reduced[:, 1], 
                       c='blue', alpha=0.6, s=50, label='Text')
            
            # Plot image embeddings
            image_reduced = reduced_embeddings[len(text_chunks):]
            plt.scatter(image_reduced[:, 0], image_reduced[:, 1], 
                       c='red', alpha=0.6, s=50, marker='^', label='Image')
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('Text và Image Embeddings Distribution (PCA)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {save_path}")
    
    def generate_evaluation_report(self, output_file: str = "embedding_evaluation_report.json"):
        """
        Tạo comprehensive evaluation report
        
        Args:
            output_file: Output file path
        """
        logger.info("Generating comprehensive evaluation report...")
        
        report = {
            'evaluation_time': datetime.now(timezone.utc).isoformat(),
            'text_embedding_metrics': {},
            'image_embedding_metrics': {},
            'cross_modal_metrics': {},
            'database_statistics': {}
        }
        
        try:
            # Text embedding evaluation
            report['text_embedding_metrics'] = self.evaluate_text_embeddings()
            
            # Image embedding evaluation
            report['image_embedding_metrics'] = self.evaluate_image_embeddings()
            
            # Cross-modal evaluation
            report['cross_modal_metrics'] = self.evaluate_cross_modal_links()
            
            # Database statistics
            report['database_statistics'] = self._get_database_statistics()
            
            # Save report
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Evaluation report saved to {output_file}")
            
            # Print summary
            self._print_evaluation_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            raise
    
    def _get_database_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        with Session(self.engine) as session:
            stats = {}
            
            # Text chunks statistics
            stats['total_text_chunks'] = session.execute(
                text("SELECT COUNT(*) FROM text_chunks")
            ).scalar()
            
            stats['text_chunks_with_embedding'] = session.execute(
                text("SELECT COUNT(*) FROM text_chunks WHERE embedding IS NOT NULL")
            ).scalar()
            
            # Image statistics
            stats['total_images'] = session.execute(
                text("SELECT COUNT(*) FROM images")
            ).scalar()
            
            stats['images_with_embedding'] = session.execute(
                text("SELECT COUNT(*) FROM images WHERE embedding IS NOT NULL")
            ).scalar()
            
            # Cross-modal links
            stats['total_links'] = session.execute(
                text("SELECT COUNT(*) FROM text_image_pairs")
            ).scalar()
            
            # Categories distribution
            text_categories = session.execute(
                text("SELECT category, COUNT(*) FROM text_chunks GROUP BY category")
            ).fetchall()
            stats['text_categories'] = {row[0] or 'unknown': row[1] for row in text_categories}
            
            image_categories = session.execute(
                text("SELECT category, COUNT(*) FROM images GROUP BY category")
            ).fetchall()
            stats['image_categories'] = {row[0] or 'unknown': row[1] for row in image_categories}
            
            return stats
    
    def _print_evaluation_summary(self, report: Dict):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("📊 EMBEDDING EVALUATION SUMMARY")
        print("="*60)
        
        # Database stats
        db_stats = report['database_statistics']
        print(f"\n📁 Database Statistics:")
        print(f"   Text chunks: {db_stats['text_chunks_with_embedding']}/{db_stats['total_text_chunks']} have embeddings")
        print(f"   Images: {db_stats['images_with_embedding']}/{db_stats['total_images']} have embeddings")
        print(f"   Cross-modal links: {db_stats['total_links']}")
        
        # Text metrics
        text_metrics = report['text_embedding_metrics']
        if text_metrics:
            print(f"\n📝 Text Embedding Quality:")
            print(f"   Mean norm: {text_metrics.get('mean_norm', 0):.3f}")
            print(f"   Mean similarity: {text_metrics.get('mean_similarity', 0):.3f}")
            if 'category_clustering_quality' in text_metrics:
                print(f"   Category clustering: {text_metrics['category_clustering_quality']:.3f}")
        
        # Image metrics
        image_metrics = report['image_embedding_metrics']
        if image_metrics:
            print(f"\n🖼️ Image Embedding Quality:")
            print(f"   Mean norm: {image_metrics.get('mean_norm', 0):.3f}")
            if 'image_category_clustering_quality' in image_metrics:
                print(f"   Category clustering: {image_metrics['image_category_clustering_quality']:.3f}")
        
        # Cross-modal metrics
        cm_metrics = report['cross_modal_metrics']
        if cm_metrics:
            print(f"\n🔗 Cross-modal Link Quality:")
            print(f"   Mean score: {cm_metrics.get('mean_score', 0):.3f}")
            print(f"   Category alignment: {cm_metrics.get('category_alignment', 0):.1%}")
            print(f"   School alignment: {cm_metrics.get('school_alignment', 0):.1%}")
        
        print("\n" + "="*60)

class EmbeddingUtils:
    """Utility functions cho embedding operations"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(db_url)
    
    def find_similar_texts(self, 
                          query_embedding: np.ndarray, 
                          limit: int = 10,
                          filters: Dict[str, str] = None) -> List[Dict]:
        """
        Tìm text chunks tương tự với query embedding
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            filters: Additional filters (school, category, etc.)
            
        Returns:
            List of similar text chunks with scores
        """
        with Session(self.engine) as session:
            # Base query
            query_sql = """
                SELECT 
                    chunk_id, content, category, school, major, location,
                    1 - (embedding <=> :query_vector) as similarity_score
                FROM text_chunks 
                WHERE embedding IS NOT NULL
            """
            
            # Add filters
            params = {"query_vector": query_embedding.tolist()}
            if filters:
                conditions = []
                for key, value in filters.items():
                    if value:
                        conditions.append(f"{key} = :{key}")
                        params[key] = value
                
                if conditions:
                    query_sql += " AND " + " AND ".join(conditions)
            
            query_sql += " ORDER BY similarity_score DESC LIMIT :limit"
            params['limit'] = limit
            
            results = session.execute(text(query_sql), params).fetchall()
            
            return [
                {
                    'chunk_id': row.chunk_id,
                    'content': row.content[:300] + "..." if len(row.content) > 300 else row.content,
                    'category': row.category,
                    'school': row.school,
                    'major': row.major,
                    'location': row.location,
                    'similarity_score': float(row.similarity_score)
                }
                for row in results
            ]
    
    def find_similar_images(self, 
                           query_embedding: np.ndarray,
                           limit: int = 10,
                           filters: Dict[str, str] = None) -> List[Dict]:
        """
        Tìm images tương tự với query embedding
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            filters: Additional filters
            
        Returns:
            List of similar images with scores
        """
        with Session(self.engine) as session:
            # Base query
            query_sql = """
                SELECT 
                    image_id, file_path, s3_url, caption, category, school, location,
                    1 - (embedding <=> :query_vector) as similarity_score
                FROM images 
                WHERE embedding IS NOT NULL
            """
            
            # Add filters
            params = {"query_vector": query_embedding.tolist()}
            if filters:
                conditions = []
                for key, value in filters.items():
                    if value:
                        conditions.append(f"{key} = :{key}")
                        params[key] = value
                
                if conditions:
                    query_sql += " AND " + " AND ".join(conditions)
            
            query_sql += " ORDER BY similarity_score DESC LIMIT :limit"
            params['limit'] = limit
            
            results = session.execute(text(query_sql), params).fetchall()
            
            return [
                {
                    'image_id': row.image_id,
                    'file_path': row.file_path,
                    's3_url': row.s3_url,
                    'caption': row.caption,
                    'category': row.category,
                    'school': row.school,
                    'location': row.location,
                    'similarity_score': float(row.similarity_score)
                }
                for row in results
            ]
    
    def test_multimodal_retrieval(self, test_queries: List[str]) -> Dict[str, List[Dict]]:
        """
        Test multimodal retrieval với sample queries
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Test results
        """
        from services.embedding_service import TextEmbeddingService
        
        # Initialize text service for query embedding
        text_service = TextEmbeddingService(self.db_url)
        
        results = {}
        
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            
            # Create query embedding
            query_embedding = text_service.create_embedding_for_query(query)
            
            # Find similar texts
            similar_texts = self.find_similar_texts(query_embedding, limit=5)
            
            # Find similar images
            similar_images = self.find_similar_images(query_embedding, limit=5)
            
            results[query] = {
                'similar_texts': similar_texts,
                'similar_images': similar_images
            }
            
            # Print top results
            print(f"\n🔍 Query: {query}")
            print("📝 Top Text Results:")
            for i, text in enumerate(similar_texts[:3], 1):
                print(f"  {i}. [{text['similarity_score']:.3f}] {text['content'][:100]}...")
            
            print("🖼️ Top Image Results:")
            for i, img in enumerate(similar_images[:3], 1):
                print(f"  {i}. [{img['similarity_score']:.3f}] {img['caption'] or img['file_path']}")
        
        return results
    
    def backup_embeddings(self, backup_path: str = "embeddings_backup.json"):
        """
        Backup tất cả embeddings ra file JSON
        
        Args:
            backup_path: Path to backup file
        """
        logger.info("Creating embeddings backup...")
        
        backup_data = {
            'backup_time': datetime.now(timezone.utc).isoformat(),
            'text_embeddings': {},
            'image_embeddings': {}
        }
        
        with Session(self.engine) as session:
            # Backup text embeddings
            text_chunks = session.query(TextChunk).filter(
                TextChunk.embedding.isnot(None)
            ).all()
            
            for chunk in text_chunks:
                backup_data['text_embeddings'][chunk.chunk_id] = {
                    'embedding': chunk.embedding,
                    'metadata': {
                        'category': chunk.category,
                        'school': chunk.school,
                        'major': chunk.major,
                        'content_preview': chunk.content[:200] if chunk.content else None
                    }
                }
            
            # Backup image embeddings
            images = session.query(Image).filter(
                Image.embedding.isnot(None)
            ).all()
            
            for img in images:
                backup_data['image_embeddings'][img.image_id] = {
                    'embedding': img.embedding,
                    'metadata': {
                        'category': img.category,
                        'school': img.school,
                        'file_path': img.file_path,
                        'caption': img.caption
                    }
                }
        
        # Save backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Backup saved: {len(backup_data['text_embeddings'])} texts, "
                   f"{len(backup_data['image_embeddings'])} images")
    
    def restore_embeddings(self, backup_path: str):
        """
        Restore embeddings từ backup file
        
        Args:
            backup_path: Path to backup file
        """
        logger.info(f"Restoring embeddings from {backup_path}")
        
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        restored_texts = 0
        restored_images = 0
        
        with Session(self.engine) as session:
            # Restore text embeddings
            for chunk_id, data in backup_data['text_embeddings'].items():
                chunk = session.query(TextChunk).filter(
                    TextChunk.chunk_id == chunk_id
                ).first()
                
                if chunk:
                    chunk.embedding = data['embedding']
                    chunk.updated_at = datetime.now(timezone.utc)
                    restored_texts += 1
            
            # Restore image embeddings
            for image_id, data in backup_data['image_embeddings'].items():
                img = session.query(Image).filter(
                    Image.image_id == image_id
                ).first()
                
                if img:
                    img.embedding = data['embedding']
                    img.updated_at = datetime.now(timezone.utc)
                    restored_images += 1
            
            session.commit()
        
        logger.info(f"Restored {restored_texts} text embeddings, {restored_images} image embeddings")

# Example usage và testing
def run_evaluation_demo():
    """Demo function để test evaluation tools"""
    
    # Database configuration
    DB_URL = "postgresql://username:password@localhost/education_db"
    
    # Initialize evaluator
    evaluator = EmbeddingEvaluator(DB_URL)
    utils = EmbeddingUtils(DB_URL)
    
    # Run evaluation
    report = evaluator.generate_evaluation_report()
    
    # Create visualization
    evaluator.visualize_embeddings(sample_size=200)
    
    # Test queries
    test_queries = [
        "thông tin tuyển sinh đại học bách khoa",
        "ký túc xá sinh viên",
        "chương trình đào tạo kỹ thuật máy tính",
        "cơ sở vật chất trường đại học",
        "hoạt động sinh viên ngoại khóa"
    ]
    
    # Test multimodal retrieval
    retrieval_results = utils.test_multimodal_retrieval(test_queries)
    
    # Backup embeddings
    utils.backup_embeddings("embeddings_backup.json")
    
    print("\n✅ Evaluation demo completed!")
    print("📊 Check embedding_evaluation_report.json for detailed metrics")
    print("📈 Check embedding_visualization.png for PCA plot")

if __name__ == "__main__":
    run_evaluation_demo()