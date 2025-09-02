from config.database import get_db_session
from models.database_models import TextChunk, Image, text_image_association
from models.data_models import EntitiesModel, TextChunkMetadata, ImageMetadata
from sqlalchemy import insert
from config.aws_config import upload_image_to_s3

text_chunks = [
    {
        "content": "2. Điều kiện xét học bổng",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "year": 2024,
        "entities": EntitiesModel(
            schools=["Trường Đại học Quốc tế TPHCM"],
            majors=["unknown"],
            locations=["TP Hồ Chí Minh"]
        )
    },
    {
        "content": "2. Điều kiện xét học bổng: a. Điều kiện xét học bổng tuyển sinh cho phương thức 1 và 4: Tổng điểm 3 môn đăng ký xét tuyển đạt mức điểm cao hơn hoặc bằng mức điểm cụ thể như sau:",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "year": 2024,
        "entities": EntitiesModel(
            schools=["Trường Đại học Quốc tế TPHCM"],
            majors=["unknown"],
            locations=["TP Hồ Chí Minh"]
        )
    },
    {
        "content": "2. Điều kiện xét học bổng : b. Điều kiện xét học bổng tuyển sinh cho phương thức 2",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "year": 2024,
        "entities": EntitiesModel(
            schools=["Trường Đại học Quốc tế TPHCM"],
            majors=["unknown"],
            locations=["TP Hồ Chí Minh"]
        )
    },
    {
        "content": "2. Điều kiện xét học bổng : c. Điều kiện xét học bổng tuyển sinh cho phương thức 5 (xét kết quả kỳ thi ĐGNL của ĐHQG-HCM)",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "year" : 2024,
        "entities" : EntitiesModel(
            schools=["Trường Đại học Quốc tế TPHCM"],
            majors=["unknown"],
            locations=["TP Hồ Chí Minh"]
        )
    },
    {
        "content": "2. Điều kiện xét học bổng : Điều kiện xét học bổng tuyển sinh cho phương thức 7 (xét điểm học bạ dành cho chương trình liên kết)",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "year" : 2024,
        "entities" : EntitiesModel(
            schools=["Trường Đại học Quốc tế TPHCM"],
            majors=["unknown"],
            locations=["TP Hồ Chí Minh"]
        )
    },
    {
        "content": "3. Điều kiện duy trì học bổng tuyển sinh : Sinh viên phải đạt điểm trung bình các môn từ 50 điểm trở lên ,  Điểm trung bình học kỳ phải từ 70/100 trở lên",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "year" : 2024,
        "entities" : EntitiesModel(
            schools=["Trường Đại học Quốc tế TPHCM"],
            majors=["unknown"],
            locations=["TP Hồ Chí Minh"]
        )
    },
    {
        "content": "B. HỌC BỔNG KHUYẾN KHÍCH HỌC TẬP. Sinh viên được xét Học bổng khuyến khích học tập dựa trên kết quả học tập mỗi học kỳ. Số suất học bổng: 40 suất cho các học kỳ chính (15tr đồng/suất), 20 suất cho học kỳ hè (7.5tr đồng/suất). Tiêu chí xét học bổng: - Hoàn tất chương trình tiếng Anh AE1 tính đến học kỳ xem xét học bổng; - Không có môn học có điểm dưới 50 kể cả môn giáo dục thể chất và giáo dục quốc phòng; - Điểm trung bình học kỳ trên 70; - Số tín chỉ đăng ký tối thiểu; 14 tín chỉ trong học kỳ chính hoặc 6 tín chỉ trong học kỳ hè; - Số suất học bổng sẽ được cấp theo thứ tự từ cao đến thấp.",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "year" : 2024,
        "entities" : EntitiesModel(
            schools=["Trường Đại học Quốc tế TPHCM"],
            majors=["unknown"],
            locations=["TP Hồ Chí Minh"]
        )
    }
]

# Dữ liệu mẫu cho ảnh
images = [
    {
        "file_path": "data/images/Hoc-bong-pt-1-4-2024-1.png",
        "caption": "Biểu đồ điều kiện học bổng phương thức 1 và 4",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "location": "TP Hồ Chí Minh"
    },
    {
        "file_path": "data/images/HB-PT-3.png",
        "caption": "Biểu đồ điều kiện học bổng phương thức 2",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "location": "TP Hồ Chí Minh"
    },
    {
        "file_path": "data/images/Hoc-bong-pt5-2024-2.png",
        "caption": "Biểu đồ Điều kiện xét học bổng tuyển sinh cho phương thức 5",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "location": "TP Hồ Chí Minh"
    },
    {
        "file_path": "data/images/PT6.png",
        "caption": "Biểu đồ điều kiện học bổng phương thức 7",
        "category": "chính sách",
        "school": "Trường Đại học Quốc tế TPHCM",
        "location": "TP Hồ Chí Minh"
    }
]

# Mối quan hệ text-image (chunk_id sẽ được gán sau khi insert)
text_image_relations = [
    {"text_idx": 0, "image_idx": 0, "relation_score": 0.7}, 
    {"text_idx": 0, "image_idx": 1, "relation_score": 0.7},
    {"text_idx": 0, "image_idx": 2, "relation_score": 0.7},  
    {"text_idx": 0, "image_idx": 3, "relation_score": 0.7},    
    {"text_idx": 1, "image_idx": 0, "relation_score": 0.9},  
    {"text_idx": 2, "image_idx": 1, "relation_score": 0.9},  
    {"text_idx": 3, "image_idx": 2, "relation_score": 0.9},   
    {"text_idx": 4, "image_idx": 3, "relation_score": 0.9},   
 
]

if __name__ == "__main__":
    with get_db_session() as session:
        try:
            inserted_texts = []
            for chunk_data in text_chunks:
                existing = session.query(TextChunk).filter_by(content=chunk_data["content"]).first()
                if existing:
                    print(f"TextChunk đã tồn tại với ID: {existing.chunk_id}")
                    inserted_texts.append(existing)
                else:
                    new_text = TextChunk(
                        content=chunk_data["content"],
                        category=chunk_data["category"],
                        school=chunk_data["school"],
                        year=chunk_data["year"],
                        entities=chunk_data["entities"].model_dump()
                    )
                    session.add(new_text)
                    inserted_texts.append(new_text)
            session.commit()  

            # Insert Images
            inserted_images = []
            for image_data in images:
                upload_result = upload_image_to_s3(image_data["file_path"])
                new_image = Image(
                    file_path=image_data["file_path"],
                    s3_url=upload_result["s3_url"], 
                    caption=image_data["caption"],
                    category=image_data["category"],
                    school=image_data["school"],
                    location=image_data["location"],
                    width=upload_result["width"],
                    height=upload_result["height"],
                    file_size=upload_result["file_size"],
                    file_format=upload_result["file_format"]
                )
                session.add(new_image)
                inserted_images.append(new_image)
            session.commit()  

            # Insert Relationships
            for relation in text_image_relations:
                stmt = insert(text_image_association).values(
                    text_chunk_id=inserted_texts[relation["text_idx"]].chunk_id,
                    image_id=inserted_images[relation["image_idx"]].image_id,
                    relation_score=relation["relation_score"]
                )
                session.execute(stmt)
            session.commit()

            # In kết quả
            for text in inserted_texts:
                print(f"Inserted TextChunk ID: {text.chunk_id}")
            for image in inserted_images:
                print(f"Inserted Image ID: {image.image_id}")
            print("Data inserted successfully!")

        except Exception as e:
            print(f"Error inserting data: {e}")