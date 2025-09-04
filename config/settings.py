"""
Cài đặt chung cho project.
Load từ environment variables và định nghĩa default values.
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Main settings class"""
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://trinhtrantrungtin@localhost:5432/admission_chatbot"
    )
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "admission_chatbot")
    DB_USER: str = os.getenv("DB_USER", "username")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "password")

    # QDRANT 
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "admission-chatbot-images")
    S3_IMAGE_PREFIX: str = os.getenv("S3_IMAGE_PREFIX", "images/")
    
    # Model Configuration
    HUGGINGFACE_MODEL_NAME: str = os.getenv(
        "HUGGINGFACE_MODEL_NAME", 
        "vinai/phobert-base-v2"
    )
    SENTENCE_TRANSFORMER_MODEL: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", 
        "keepitreal/vietnamese-sbert"
    )
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "vi_core_news_lg")
    
    # Text Processing Configuration
    TEXT_CHUNK_SIZE: int = int(os.getenv("TEXT_CHUNK_SIZE", "500"))
    TEXT_OVERLAP_SIZE: int = int(os.getenv("TEXT_OVERLAP_SIZE", "50"))
    
    # Processing Configuration
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Similarity Thresholds
    TEXT_IMAGE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("TEXT_IMAGE_SIMILARITY_THRESHOLD", "0.7")
    )
    ENTITY_MATCH_THRESHOLD: float = float(
        os.getenv("ENTITY_MATCH_THRESHOLD", "0.8")
    )
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/processing.log")
    
    # Supported File Types
    SUPPORTED_DOCUMENT_TYPES: List[str] = ['.pdf', '.docx', '.txt']
    SUPPORTED_IMAGE_TYPES: List[str] = ['.jpg', '.jpeg', '.png', '.webp']
    
    # Text Categories (for classification)
    TEXT_CATEGORIES: List[str] = [
        "tuyển sinh",           # Admission information
        "thông tin trường",     # School information
        "ngành học",           # Major/program information
        "cơ sở vật chất",      # Infrastructure/facilities
        "hoạt động sinh viên", # Student activities
        "học phí",             # Tuition fees
        "chính sách",          # Policies
        "khác"                 # Other
    ]
    
    # Image Categories
    IMAGE_CATEGORIES: List[str] = [
        "campus",       # Campus photos
        "classroom",    # Classroom/lab photos
        "dormitory",    # Dormitory photos
        "activity",     # Student activity photos
        "building",     # Building photos
        "library",      # Library photos
        "lab",          # Laboratory photos
        "other"         # Other photos
    ]
    
    # Vietnamese Universities (sample list for NER)
    VIETNAMESE_UNIVERSITIES: List[str] = [
        "Đại học Bách Khoa Hà Nội",
        "Đại học Quốc gia Hà Nội",
        "Đại học Kinh tế Quốc dân",
        "Đại học Ngoại thương",
        "Đại học Bách Khoa TP.HCM",
        "Đại học Quốc gia TP.HCM",
        "Đại học Kinh tế TP.HCM",
        "Đại học Công nghệ Thông tin",
        "Đại học Y Hà Nội",
        "Đại học Luật Hà Nội",
        "Trường Đại học Quốc tế TPHCM",
        "Trường Đại học Bách Khoa TPHCM"
    ]
    
    # Common Majors in Vietnamese (for NER)
    VIETNAMESE_MAJORS: List[str] = [
        "Công nghệ Thông tin",
        "Khoa học Máy tính",
        "Kỹ thuật Điện",
        "Kỹ thuật Cơ khí",
        "Kinh tế",
        "Quản trị Kinh doanh",
        "Kế toán",
        "Tài chính Ngân hàng",
        "Luật",
        "Y khoa",
        "Dược",
        "Kiến trúc",
        "Xây dựng",
        "Ngoại ngữ",
        "Báo chí Truyền thông"
    ]
    
    # Vietnamese Provinces/Cities (for location NER)
    VIETNAMESE_LOCATIONS: List[str] = [
        "Hà Nội",
        "TP.HCM",
        "Tp. Hồ Chí Minh",
        "Đà Nẵng",
        "Hải Phòng",
        "Cần Thơ",
        "Huế",
        "Nha Trang",
        "Vũng Tàu",
        "Đà Lạt"
    ]
    
    @classmethod
    def validate_settings(cls) -> bool:
        """Validate required settings"""
        required_fields = [
            "DATABASE_URL",
            "AWS_ACCESS_KEY_ID", 
            "AWS_SECRET_ACCESS_KEY",
            "S3_BUCKET_NAME"
        ]
        
        for field in required_fields:
            if not getattr(cls, field):
                print(f"Warning: {field} is not set")
                return False
        
        return True

# Singleton instance
settings = Settings()

# Validate settings on import
if not settings.validate_settings():
    print("Warning: Some required settings are missing. Please check your .env file.")