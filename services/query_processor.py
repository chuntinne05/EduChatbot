import re
import logging
from typing import List, Dict
from underthesea import word_tokenize, ner
from models.retrieval_models import SearchIntent, QueryType
import json

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Xử lý và phân tích query người dùng"""
    
    def __init__(self):
        self.entity_mapping = self._load_entity_mapping()
        self.synonyms = self._load_synonyms()
        self.education_keywords = self._load_education_keywords()
        
    def _load_entity_mapping(self) -> Dict[str, str]:
        """Load mapping các entity chuẩn hóa"""
        return {
            # Đại học
            "đhqg tphcm": "Đại học Quốc gia TP.HCM",
            "đh kinh tế tphcm": "Đại học Kinh Tế TP.HCM",
            "đhqg hà nội": "Đại học Quốc gia Hà Nội",

            # Trường đại học
            "đh bách khoa tphcm": "Đại học Bách Khoa TP.HCM",
            "hcmut": "Đại học Bách Khoa TP.HCM",
            "bách khoa": "Đại học Bách Khoa",
            "bk": "Đại học Bách Khoa",
            "đh bkhoa": "Đại học Bách Khoa",
            "trường đh an ninh nhân dân": "Trường Đại học An ninh Nhân dân",
            "đh annd": "Trường Đại học An ninh Nhân dân",
            "trường đh cảnh sát nhân dân": "Trường Đại học Cảnh sát Nhân dân",
            "đh csnd": "Trường Đại học Cảnh sát Nhân dân",
            "trường đh công nghiệp tphcm": "Trường Đại học Công nghiệp TP.HCM",
            "iuh": "Trường Đại học Công nghiệp TP.HCM",
            "đh cn tphcm": "Trường Đại học Công nghiệp TP.HCM",
            "trường đh công nghệ thông tin tphcm": "Trường Đại học Công nghệ Thông tin TP.HCM",
            "đh cntt": "Trường Đại học Công nghệ Thông tin",
            "trường đh công thương tphcm": "Trường Đại học Công Thương TP.HCM",
            "huit": "Trường Đại học Công Thương TP.HCM",
            "đh ct tphcm": "Trường Đại học Công Thương TP.HCM",
            "trường đh gtvt tphcm": "Trường Đại học Giao thông Vận tải TP.HCM",
            "uth": "Trường Đại học Giao thông Vận tải TP.HCM",
            "đh gtvt": "Trường Đại học Giao thông Vận tải",
            "trường đh gtvt phân hiệu tphcm": "Trường Đại học Giao thông Vận tải - Phân hiệu TP.HCM",
            "utc2": "Trường Đại học Giao thông Vận tải - Phân hiệu TP.HCM",
            "đh gtvt ph": "Trường Đại học Giao thông Vận tải - Phân hiệu TP.HCM",
            "trường đh khtn tphcm": "Trường Đại học Khoa học Tự nhiên TP.HCM",
            "đh khtn": "Trường Đại học Khoa học Tự nhiên",
            "hcmus": "Trường Đại học Khoa học Tự nhiên TP.HCM",
            "trường đh khoa học xã hội và nhân văn tphcm": "Trường Đại học Khoa học Xã hội và Nhân văn TP.HCM",
            "trường đh khxh&nv": "Trường Đại học Khoa học Xã hội và Nhân văn TP.HCM",
            "ussh": "Trường Đại học Khoa học Xã hội và Nhân văn TP.HCM",
            "trường đh kinh tế luật tphcm": "Trường Đại học Kinh tế - Luật TP.HCM",
            "trường đh kt luật": "Trường Đại học Kinh tế - Luật TP.HCM",
            "uel": "Trường Đại học Kinh tế - Luật TP.HCM",
            "trường đh lao động-xã hội tphcm": "Trường Đại học Lao động - Xã hội TP.HCM",
            "đh lđxh": "Trường Đại học Lao động - Xã hội",
            "trường đh kiến trúc tphcm": "Trường Đại học Kiến trúc TP.HCM",
            "uah": "Trường Đại học Kiến trúc TP.HCM",
            "trường đh luật tphcm": "Trường Đại học Luật TP.HCM",
            "ulaw": "Trường Đại học Luật TP.HCM",
            "trường đh mở tphcm": "Trường Đại học Mở TP.HCM",
            "trường đh mỹ thuật tphcm": "Trường Đại học Mỹ thuật TP.HCM",
            "trường đh ngoại thương cs 2": "Trường Đại học Ngoại thương Cơ sở 2",
            "trường đh ngoại thương cơ sở 2": "Trường Đại học Ngoại thương Cơ sở 2",
            "FTU2": "Trường Đại học Ngoại thương Cơ sở 2",
            "trường đh ngân hàng tphcm": "Trường Đại học Ngân hàng TP.HCM",
            "hub": "Trường Đại học Ngân hàng TP.HCM",
            "trường đh nông lam tphcm": "Trường Đại học Nông Lâm TP.HCM",
            "nlu": "Trường Đại học Nông Lâm TP.HCM",
            "trường đh quốc tế tphcm": "Trường Đại học Quốc tế TP.HCM",
            "hcmiu": "Trường Đại học Quốc tế TP.HCM",
            "trường đh sài gòn": "Trường Đại học Sài Gòn",
            "đh sg": "Trường Đại học Sài Gòn",
            "sgu": "Trường Đại học Sài Gòn",
            "trường đh sân khấu - điện ảnh tphcm": "Trường Đại học Sân khấu - Điện ảnh TP.HCM",
            "trường đại học spkt tphcm": "Trường Đại học Sư phạm Kỹ thuật TP.HCM",
            "đh spkt": "Trường Đại học Sư phạm Kỹ thuật",
            "hcmute": "Trường Đại học Sư phạm Kỹ thuật TP.HCM",
            "trường đh sư phạm tdtt tphcm": "Trường Đại học Sư phạm Thể dục Thể thao TP.HCM",
            "đh sp tdtt": "Trường Đại học Sư phạm Thể dục Thể thao",
            "trường đh sư phạm tphcm": "Trường Đại học Sư phạm TP.HCM",
            "đh sp tphcm": "Trường Đại học Sư phạm TP.HCM",
            "hcmue": "Trường Đại học Sư phạm TP.HCM",
            "trường đh tdtt tphcm": "Trường Đại học Thể dục Thể thao TP.HCM",
            "phân hiệu trường đh thuỷ lợi tại tphcm": "Phân hiệu Trường Đại học Thủy lợi tại TP.HCM",
            "trường đh trần đại nghĩa": "Trường Đại học Trần Đại Nghĩa",
            "trường đh tài chính - marketing": "Trường Đại học Tài chính - Marketing",
            "đh tài chính - marketing": "Trường Đại học Tài chính - Marketing",
            "ufm": "Trường Đại học Tài chính - Marketing",
            "trường đh tài nguyên môi trường tphcm": "Trường Đại học Tài nguyên Môi trường TP.HCM",
            "trường đh tdt": "Trường Đại học Tôn Đức Thắng",
            "tdtu": "Trường Đại học Tôn Đức Thắng",
            "trường đh tôn đức thắng": "Trường Đại học Tôn Đức Thắng",
            "trường đh văn hoá tphcm": "Trường Đại học Văn hóa TP.HCM",
            "trường đh Việt-Đức": "Trường Đại học Việt-Đức",
            "trường đh y dược tphcm": "Trường Đại học Y Dược TP.HCM",
            "ump": "Trường Đại học Y Dược TP.HCM",
            "trường đh y khoa phạm ngọc thạch": "Trường Đại học Y khoa Phạm Ngọc Thạch",
            "trường đh y khoa pnt": "Trường Đại học Y khoa Phạm Ngọc Thạch",
            "pnt": "Trường Đại học Y khoa Phạm Ngọc Thạch",
            "trường đh sức khoẻ tphcm": "Trường Đại học Sức khỏe TP.HCM",
            "uit": "Trường Đại học Công nghệ Thông tin",
            "hust": "Đại học Bách Khoa Hà Nội",
            "ueh": "Đại học Kinh Tế TP.HCM",
            "đh y": "Đại học Y",
            "đh luật": "Đại học Luật",

            # đh tư thục
            "trường đh hoa sen": "Trường Đại học Hoa Sen",
            "hsu": "Trường Đại học Hoa Sen",
            "trường đh ngoại ngữ - tin học tp.hcm": "Trường Đại học Ngoại ngữ - Tin học TP.HCM",
            "huflit": "Trường Đại học Ngoại ngữ - Tin học TP.HCM",
            "trường đh hùng vương tphcm": "Trường Đại học Hùng Vương TP.HCM",
            "hvuh": "Trường Đại học Hùng Vương TP.HCM",
            "trường đh văn lang": "Trường Đại học Văn Lang",
            "vluh": "Trường Đại học Văn Lang",
            "trường đh công nghệ tp.hcm": "Trường Đại học Công nghệ TP.HCM",
            "hutech": "Trường Đại học Công nghệ TP.HCM",
            "trường đh quốc tế hồng bàng": "Trường Đại học Quốc tế Hồng Bàng",
            "hiu": "Trường Đại học Quốc tế Hồng Bàng",            
            "trường đh văn hiến": "Trường Đại học Văn Hiến",
            "vhu": "Trường Đại học Văn Hiến",
            "trường đh công nghệ sg": "Trường Đại học Công nghệ Sài Gòn",
            "stu": "Trường Đại học Công nghệ Sài Gòn",
            "trường đh nguyễn tất thành": "Trường Đại học Nguyễn Tất Thành",
            "nttu": "Trường Đại học Nguyễn Tất Thành",
            "trường đh fpt" : "Trường Đại học FPT",
            "fpt": "Trường Đại học FPT",
            "trường đh gia định": "Trường Đại học Gia Định",
            "trường đh kinh tế - tài chính tphcm": "Trường Đại học Kinh tế - Tài chính TP.HCM",
            "uef": "Trường Đại học Kinh tế - Tài chính TP.HCM",
            "trường đh quốc tế sài gòn": "Trường Đại học Quốc tế Sài Gòn",
            "siu": "Trường Đại học Quốc tế Sài Gòn",
            
            # học viện
            "học viện cán bộ tphcm": "Học viện Cán bộ TP.HCM",
            "học viện công nghệ bưu chính viễn thông": "Học viện Công nghệ Bưu chính Viễn thông",
            "học viện hàng không vn": "Học viện Hàng không Việt Nam",
            "học viện kỹ thuật mật mã cơ sở phía nam": "Học viện Kỹ thuật Mật mã cơ sở phía Nam",
            "nhạc viện tphcm": "Nhạc viện TP.HCM",

            # đh do ngước ngoài quản lý
            "đại học rmit": "Đại học RMIT",
            "đh rmit": "Đại học RMIT",
            "đh fullbright": "Đại học Fulbright",
            "đại học fulbright": "Đại học Fulbright",
            "đh quốc tế pacific vn": "Đại học Quốc tế Pacific Việt Nam",
            "đh greenwich": "Đại học Greenwich",
            "đh swinburne": "Đại học Swinburne",
            "đh columbia southern vietnam"  : "Đại học Columbia Southern Vietnam",
            
            # Ngành học
            "kỹ thuật mt": "kỹ thuật máy tính",
            "khmt": "khoa học máy tính",
            "điện điện tử": "Điện - Điện tử",
            "điện tử viễn thông": "Điện tử - Viễn thông",
            "vi mạch": "Vi mạch",
            "thiết kế vi mạch": "Thiết kế Vi mạch",
            "kỹ thuật ck": "Kỹ thuật Cơ khí",
            "kỹ thuật cơ điện tử": "Kỹ thuật Cơ điện tử",
            "dệt may": "Dệt - May",
            "logicstics": "Quản lý chuỗi cung ứng",
            "logistics": "Logistics",
            "kỹ thuật hàng hải": "Kỹ thuật Hàng hải",
            "kỹ thuật tàu thủy": "Kỹ thuật Tàu thủy",
            "kỹ thuật xây dựng": "Kỹ thuật Xây dựng",
            "kỹ thuật hàng không": "Kỹ thuật Hàng không",
            "kỹ thuật ô tô": "Kỹ thuật Ô tô",
            "cntt": "Công nghệ thông tin",
            "it": "Công nghệ thông tin",
            "ktmt": "Kỹ thuật máy tính",
            "kts": "Kiến trúc sư",
            "xd": "Xây dựng",
            "ql": "Quản lý",
            "kt": "Kinh tế",
            
            # Thành phố
            "hcm": "TP.Hồ Chí Minh",
            "hn": "Hà Nội",
            "dn": "Đà Nẵng",
        }
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load từ đồng nghĩa"""
        return {
            "học phí": ["chi phí học", "mức phí", "tiền học", "học bổng"],
            "tuyển sinh": ["xét tuyển", "nhập học", "điểm chuẩn", "thông tin tuyển sinh"],
            "cơ sở vật chất": ["trang thiết bị", "phòng lab", "thư viện", "ký túc xá"],
            "chương trình đào tạo": ["khung chương trình", "môn học", "tín chỉ"],
            "việc làm": ["nghề nghiệp", "cơ hội việc làm", "tỷ lệ có việc làm"],
            "hoạt động": ["sinh hoạt", "câu lạc bộ", "sự kiện", "festival"]
        }
    
    def _load_education_keywords(self) -> Dict[str, List[str]]:
        """Load các keyword theo domain giáo dục"""
        return {
            "admission": ["tuyển sinh", "xét tuyển", "điểm chuẩn", "thông tin tuyển sinh"],
            "tuition": ["học phí", "chi phí", "học bổng", "hỗ trợ tài chính"],
            "program": ["chương trình", "ngành học", "khoa", "bộ môn"],
            "facility": ["cơ sở vật chất", "thư viện", "lab", "ký túc xá"],
            "activity": ["hoạt động", "sinh hoạt", "câu lạc bộ", "sự kiện"],
            "location": ["địa điểm", "campus", "cơ sở", "chi nhánh"]
        }
    
    def analyze_query_intent(self, query: str) -> SearchIntent:
        """Phân tích intent của query"""
        query_lower = query.lower().strip()
        
        # Tokenize query
        tokens = word_tokenize(query_lower)
        
        # Named Entity Recognition
        entities = self._extract_entities(query)
        normalized_entities = self._normalize_entities(entities)
        
        # Phân tích xem cần text, image hay cả hai
        needs_images = self._needs_images(query_lower, tokens)
        needs_text = True  # Luôn cần text
        
        # Xác định query type
        query_type = QueryType.MULTIMODAL if needs_images else QueryType.TEXT_ONLY
        
        # Expand query với synonyms
        expanded_query = self._expand_query(query, tokens)
        
        return SearchIntent(
            needs_text=needs_text,
            needs_images=needs_images,
            query_type=query_type,
            entities=entities,
            normalized_entities=normalized_entities,
            expanded_query=expanded_query
        )
    
    def _extract_entities(self, query: str) -> List[str]:
        """Trích xuất entities từ query"""
        entities = []
        
        try:
            ner_result = ner(query)
            for item in ner_result:
                if isinstance(item, tuple) and len(item) >= 3:
                    token, pos, label = item[0], item[1], item[2]
                elif isinstance(item, dict):
                    token = item.get('word', '')
                    label = item.get('entity', '')
                else:
                    continue
                    
                if label in ["B-ORG", "B-LOC", "B-PER", "I-ORG", "I-LOC", "I-PER"]:
                    entities.append(token)
        except Exception as e:
            logger.warning(f"NER failed: {e}")
        
        query_lower = query.lower()
        for entity_key in self.entity_mapping.keys():
            if entity_key in query_lower:
                entities.append(entity_key)
        
        return list(set(entities))
    
    def _normalize_entities(self, entities: List[str]) -> Dict[str, str]:
        """Chuẩn hóa entities"""
        normalized = {}
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in self.entity_mapping:
                normalized[entity] = self.entity_mapping[entity_lower]
            else:
                normalized[entity] = entity
        
        return normalized
    
    def _needs_images(self, query_lower: str, tokens: List[str]) -> bool:
        """Xác định query có cần image không"""
        image_keywords = [
            "hình ảnh", "ảnh", "hình", "chụp", "photo", "picture",
            "campus", "cơ sở vật chất", "trang thiết bị", "phòng lab",
            "thư viện", "ký túc xá", "lớp học", "hoạt động", "sự kiện"
        ]
        
        for keyword in image_keywords:
            if keyword in query_lower:
                return True
        
        if any(word in tokens for word in ["xem", "cho", "thấy"]) and \
           any(word in tokens for word in ["trường", "ngành", "cơ sở"]):
            return True
        
        return False
    
    def _expand_query(self, query: str, tokens: List[str]) -> str:
        """Mở rộng query với synonyms"""
        expanded_terms = []
        
        expanded_terms.append(query)
        
        for term, synonyms in self.synonyms.items():
            if term in query.lower():
                expanded_terms.extend(synonyms[:2]) 
        
        query_lower = query.lower()
        for key, value in self.entity_mapping.items():
            if key in query_lower and value not in expanded_terms:
                expanded_terms.append(value)
        
        return " ".join(expanded_terms)
    
    def preprocess_query(self, query: str) -> str:
        """Tiền xử lý query cho vector search"""
        # Remove special characters
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Remove extra spaces
        query = ' '.join(query.split())
        
        # Normalize entities in query
        query_lower = query.lower()
        for key, value in self.entity_mapping.items():
            query_lower = query_lower.replace(key, value.lower())
        
        return query_lower.strip()
    
    def extract_filters(self, query: str) -> Dict[str, any]:
        """Trích xuất filters từ query"""
        filters = {}
        query_lower = query.lower()
        
        # Location filters
        locations = ["hà nội", "tp.hcm", "đà nẵng", "hải phòng", "miền nam", "miền bắc", "miền trung", "tây nguyên"]
        for loc in locations:
            if loc in query_lower:
                filters["location"] = loc
                break
        
        if "chỉ text" in query_lower or "không cần ảnh" in query_lower:
            filters["content_type"] = "text"
        elif "chỉ ảnh" in query_lower or "hình ảnh" in query_lower:
            filters["content_type"] = "image"
        
        categories = {
            "tuyển sinh": ["tuyển sinh", "xét tuyển", "điểm chuẩn"],
            "học phí": ["học phí", "chi phí", "học bổng"],
            "cơ sở vật chất": ["cơ sở vật chất", "thư viện", "lab", "ký túc xá", ],
            "chương trình": ["chương trình", "môn học", "tín chỉ"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                filters["category"] = category
                break
        
        return filters