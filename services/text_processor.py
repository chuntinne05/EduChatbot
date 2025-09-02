"""
Text processing service.
NER với spaCy, phân loại với zero-shot classification, chunking text.
"""

import spacy
import re
from transformers import pipeline
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import hashlib

from config.settings import settings
from models.data_models import EntitiesModel, TextChunkMetadata, TextChunkCreate

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing và NER service"""
    
    def __init__(self):
        self.nlp = None
        self.classifier = None
        self._load_models()
    
    def _load_models(self):
        """Load spaCy và HuggingFace models"""
        try:
            # Load spaCy model cho Vietnamese
            logger.info(f"Loading spaCy model: {settings.SPACY_MODEL}")
            self.nlp = spacy.load(settings.SPACY_MODEL)
            
            # Load PhoBERT classifier cho text classification
            logger.info(f"Loading classifier model: {settings.HUGGINGFACE_MODEL_NAME}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=settings.HUGGINGFACE_MODEL_NAME,
                tokenizer=settings.HUGGINGFACE_MODEL_NAME
            )
            
            logger.info("Text processing models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Chia text thành chunks với overlap
        
        Args:
            text: Input text
            chunk_size: Size của mỗi chunk (characters)
            overlap: Số characters overlap giữa chunks
        
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or settings.TEXT_CHUNK_SIZE
        overlap = overlap or settings.TEXT_OVERLAP_SIZE
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Tìm sentence boundary gần nhất để avoid cắt giữa câu
            if end < len(text):
                # Look for sentence endings trong khoảng 50 characters
                sentence_ends = ['.', '!', '?', '\n']
                best_end = end
                
                for i in range(max(end - 50, start), min(end + 50, len(text))):
                    if text[i] in sentence_ends and i > start:
                        best_end = i + 1
                        break
                
                end = best_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        return chunks
    
    def extract_entities(self, text: str) -> EntitiesModel:
        """
        Extract named entities từ text
        
        Args:
            text: Input text
        
        Returns:
            EntitiesModel với extracted entities
        """
        entities = EntitiesModel()
        
        try:
            # Process với spaCy
            doc = self.nlp(text)
            
            # Extract từ spaCy NER
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:  # Organizations, Universities
                    entities.schools.extend(self._match_universities(ent.text))
                elif ent.label_ in ['GPE', 'LOC']:  # Locations
                    entities.locations.extend(self._match_locations(ent.text))
            
            # Pattern matching cho Vietnamese universities
            university_patterns = [
                r'(Đại học|Học viện|Trường đại học)\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ\s]+',
                r'(ĐH|DH)\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ\s]+'
            ]
            
            for pattern in university_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        university = ' '.join(match).strip()
                    else:
                        university = match.strip()
                    
                    if university and university not in entities.schools:
                        entities.schools.append(university)
            
            # Pattern matching cho majors
            major_keywords = [
                r'ngành\s+([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ\s]+)',
                r'chuyên ngành\s+([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ\s]+)',
                r'(Công nghệ thông tin|Khoa học máy tính|Kỹ thuật|Kinh tế|Y khoa|Luật)'
            ]
            
            for pattern in major_keywords:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        major = match[0].strip() if match[0] else match[1].strip()
                    else:
                        major = match.strip()
                    
                    if major and len(major) > 3 and major not in entities.majors:
                        entities.majors.append(major)
            
            # Remove duplicates và clean up
            entities.schools = list(set([s for s in entities.schools if len(s) > 3]))
            entities.majors = list(set([m for m in entities.majors if len(m) > 3]))
            entities.locations = list(set([l for l in entities.locations if len(l) > 2]))
            
            logger.debug(f"Extracted entities: {entities}")
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
        
        return entities
    
    def classify_text(self, text: str) -> str:
        """
        Classify text thành categories
        
        Args:
            text: Input text
        
        Returns:
            Category name
        """
        try:
            # Truncate text nếu quá dài để avoid token limit
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.classifier(text, settings.TEXT_CATEGORIES)
            
            # Return category với highest score
            if result['scores'][0] > 0.5:  # Threshold để ensure confidence
                return result['labels'][0]
            else:
                return "khác"  # Default category
                
        except Exception as e:
            logger.error(f"Failed to classify text: {e}")
            return "khác"
    
    def extract_year(self, text: str) -> int:
        """
        Extract năm từ text (năm tuyển sinh, năm học)
        
        Args:
            text: Input text
        
        Returns:
            Year hoặc None
        """
        # Pattern cho năm (2020-2030)
        year_patterns = [
            r'năm\s+(20[2-3]\d)',
            r'(20[2-3]\d)',
            r'khóa\s+(20[2-3]\d)',
            r'tuyển sinh\s+(20[2-3]\d)'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    year = int(matches[0])
                    if 2020 <= year <= 2030:
                        return year
                except ValueError:
                    continue
        
        return None
    
    def process_text_chunk(
        self, 
        content: str, 
        source_doc: str, 
        page_range: List[int] = None
    ) -> TextChunkCreate:
        """
        Process một text chunk hoàn chỉnh
        
        Args:
            content: Text content
            source_doc: Source document name
            page_range: Page range [start, end]
        
        Returns:
            TextChunkCreate object
        """
        try:
            # Extract entities
            entities = self.extract_entities(content)
            
            # Classify text
            category = self.classify_text(content)
            
            # Extract year
            year = self.extract_year(content)
            
            # Create metadata
            metadata = TextChunkMetadata(
                category=category,
                school=entities.schools[0] if entities.schools else None,
                major=entities.majors[0] if entities.majors else None,
                location=entities.locations[0] if entities.locations else None,
                year=year
            )
            
            return TextChunkCreate(
                content=content,
                source_doc=source_doc,
                page_range=page_range,
                metadata=metadata,
                entities=entities
            )
            
        except Exception as e:
            logger.error(f"Failed to process text chunk: {e}")
            raise
    
    def generate_chunk_id(self, content: str, source_doc: str) -> str:
        """Generate unique chunk ID từ content hash"""
        hash_input = f"{content[:100]}{source_doc}"  # First 100 chars + source
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _match_universities(self, text: str) -> List[str]:
        """Match text với known universities"""
        matches = []
        text_lower = text.lower()
        
        for university in settings.VIETNAMESE_UNIVERSITIES:
            if any(word in text_lower for word in university.lower().split()):
                matches.append(university)
        
        return matches
    
    def _match_locations(self, text: str) -> List[str]:
        """Match text với known locations"""
        matches = []
        text_lower = text.lower()
        
        for location in settings.VIETNAMESE_LOCATIONS:
            if location.lower() in text_lower:
                matches.append(location)
        
        return matches

# Singleton instance
text_processor = TextProcessor()

# Convenience functions
def process_text_chunk(content: str, source_doc: str, page_range: List[int] = None) -> TextChunkCreate:
    """Process text chunk"""
    return text_processor.process_text_chunk(content, source_doc, page_range)

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Chunk text into smaller pieces"""
    return text_processor.chunk_text(text, chunk_size, overlap)