import os
from typing import Dict, List, Any
from pydantic import BaseSettings, Field

class RetrievalConfig(BaseSettings):
    """Configuration for retrieval system"""
    
    # Vector Search Settings
    SIMILARITY_THRESHOLD: float = Field(default=0.5, description="Minimum similarity threshold")
    MAX_RESULTS_STAGE1: int = Field(default=100, description="Max results from stage 1 (semantic search)")
    MAX_RESULTS_STAGE2: int = Field(default=20, description="Max results from stage 2 (reranking)")
    MAX_RESULTS_STAGE3: int = Field(default=10, description="Max results from stage 3 (diversity)")
    
    # Reranking Settings
    CROSS_ENCODER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    RERANK_BATCH_SIZE: int = Field(default=32, description="Batch size for reranking")
    
    # Diversity Settings
    DEFAULT_MAX_SAME_SOURCE: int = Field(default=3, description="Max results from same source")
    DEFAULT_MAX_SAME_CATEGORY: int = Field(default=4, description="Max results from same category") 
    DEFAULT_SIMILARITY_THRESHOLD: float = Field(default=0.85, description="Semantic diversity threshold")
    DIVERSITY_WEIGHT: float = Field(default=0.3, description="Weight for diversity scoring")
    
    # Fusion Scoring Weights
    FUSION_RERANK_WEIGHT: float = Field(default=0.6, description="Weight for rerank score in fusion")
    FUSION_RULE_WEIGHT: float = Field(default=0.3, description="Weight for rule-based score in fusion")
    FUSION_SIMILARITY_WEIGHT: float = Field(default=0.1, description="Weight for similarity score in fusion")
    
    # Cross-modal Search Settings
    TEXT_WEIGHT: float = Field(default=0.7, description="Weight for text results in cross-modal search")
    IMAGE_WEIGHT: float = Field(default=0.3, description="Weight for image results in cross-modal search")
    
    # Query Processing
    MAX_EXPANDED_TERMS: int = Field(default=5, description="Max terms to add in query expansion")
    ENABLE_ENTITY_NORMALIZATION: bool = Field(default=True, description="Enable entity normalization")
    
    # Performance Settings
    ENABLE_CACHING: bool = Field(default=True, description="Enable result caching")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")
    MAX_BATCH_SIZE: int = Field(default=10, description="Max queries in batch request")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_QUERIES: bool = Field(default=True, description="Log user queries")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Entity mapping configuration
ENTITY_MAPPING = {
    # Universities
    "đh bách khoa": "Đại học Bách Khoa",
    "bách khoa": "Đại học Bách Khoa", 
    "bk": "Đại học Bách Khoa",
    "hust": "Đại học Bách Khoa Hà Nội",
    "đh bk hcm": "Đại học Bách Khoa TP.HCM",
    "bkhn": "Đại học Bách Khoa Hà Nội",
    
    "đh kinh tế": "Đại học Kinh Tế",
    "ueh": "Đại học Kinh Tế TP.HCM",
    "neu": "Đại học Kinh Tế Quốc Dân",
    
    "đh y": "Đại học Y",
    "đh y hà nội": "Đại học Y Hà Nội",
    "đh y dược": "Đại học Y Dược",
    
    "đh luật": "Đại học Luật",
    "đh huế": "Đại học Huế",
    "đh đà nẵng": "Đại học Đà Nẵng",
    "đh cần thơ": "Đại học Cần Thơ",
    
    "fpt": "Đại học FPT",
    "đh fpt": "Đại học FPT",
    "rmit": "Đại học RMIT",
    "biu": "Đại học Quốc Tế",
    
    # Fields of study
    "cntt": "Công nghệ thông tin",
    "it": "Công nghệ thông tin", 
    "ktmt": "Kỹ thuật máy tính",
    "khmt": "Khoa học máy tính",
    "attt": "An toàn thông tin",
    
    "kt": "Kinh tế",
    "qtkd": "Quản trị kinh doanh",
    "tc": "Tài chính",
    "nh": "Ngân hàng",
    "mk": "Marketing",
    
    "xd": "Xây dựng",
    "ktxd": "Kỹ thuật xây dựng",
    "kts": "Kiến trúc sư",
    
    "y": "Y học",
    "ds": "Dược sĩ",
    "răng hàm mặt": "Răng hàm mặt",
    
    # Cities
    "hcm": "TP.Hồ Chí Minh",
    "tphcm": "TP.Hồ Chí Minh",
    "hn": "Hà Nội", 
    "dn": "Đà Nẵng",
    "hp": "Hải Phòng",
    "ct": "Cần Thơ"
}

# Synonyms for query expansion
SYNONYMS = {
    "học phí": ["chi phí học", "mức phí", "tiền học", "học bổng", "hỗ trợ tài chính"],
    "tuyển sinh": ["xét tuyển", "nhập học", "điểm chuẩn", "thông tin tuyển sinh", "đăng ký"],
    "cơ sở vật chất": ["trang thiết bị", "phòng lab", "thư viện", "ký túc xá", "cơ sở hạ tầng"],
    "chương trình đào tạo": ["khung chương trình", "môn học", "tín chỉ", "giáo trình"],
    "việc làm": ["nghề nghiệp", "cơ hội việc làm", "tỷ lệ có việc làm", "ra trường"],
    "hoạt động": ["sinh hoạt", "câu lạc bộ", "sự kiện", "festival", "tổ chức"],
    "giảng viên": ["thầy cô", "giáo sư", "tiến sĩ", "giảng dạy", "thạc "],
    "sinh viên": ["học sinh", "student", "học viên"],
    "campus": ["khuôn viên", "cơ sở", "trường"],
    "ngành học": ["chuyên ngành", "bộ môn", "khoa", "lĩnh vực"]
}

# Education domain keywords
EDUCATION_KEYWORDS = {
    "admission": ["tuyển sinh", "xét tuyển", "điểm chuẩn", "thông tin tuyển sinh", "đăng ký"],
    "tuition": ["học phí", "chi phí", "học bổng", "hỗ trợ tài chính", "miễn giảm"],
    "program": ["chương trình", "ngành học", "khoa", "bộ môn", "chuyên ngành"],
    "facility": ["cơ sở vật chất", "thư viện", "lab", "ký túc xá", "trang thiết bị"],
    "activity": ["hoạt động", "sinh hoạt", "câu lạc bộ", "sự kiện", "festival"],
    "location": ["địa điểm", "campus", "cơ sở", "chi nhánh", "khuôn viên"],
    "career": ["việc làm", "nghề nghiệp", "cơ hội", "tỷ lệ có việc làm"],
    "faculty": ["giảng viên", "thầy cô", "giáo sư", "tiến sĩ"]
}

# Image category mapping
IMAGE_CATEGORIES = {
    "campus": ["khuôn viên", "trường", "cơ sở", "toàn cảnh"],
    "classroom": ["lớp học", "phòng học", "giảng đường"], 
    "laboratory": ["phòng lab", "thí nghiệm", "thực hành"],
    "library": ["thư viện", "đọc sách", "tài liệu"],
    "dormitory": ["ký túc xá", "nơi ở", "chỗ ở"],
    "activity": ["hoạt động", "sinh hoạt", "sự kiện", "festival"],
    "graduation": ["tốt nghiệp", "lễ trao bằng"],
    "sports": ["thể thao", "sân bóng", "gym"],
    "facility": ["cơ sở vật chất", "trang thiết bị"]
}

# Load configuration
config = RetrievalConfig()