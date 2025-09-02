from config.database import get_db_session  
from models.database_models import TextChunk, Image, text_image_association  
from models.data_models import EntitiesModel, TextChunkMetadata, ImageMetadata  
from sqlalchemy import insert 

text_content = "2. Điều kiện xét học bổng : a. Điều kiện xét học bổng tuyển sinh cho phương thức 1 và 4: Tổng điểm 3 môn đăng ký xét tuyển đạt mức điểm cao hơn hoặc bằng mức điểm cụ thể như sau:"
image_path = None

if __name__ == "__main__":
    with get_db_session() as session:  
        try:
            entities = EntitiesModel(schools=["Trường Đại học Quốc tế TPHCM"], majors=["unknown"], locations=["TP Hồ Chí Minh"])
            metadata_text = TextChunkMetadata(category="chính sách", school="Trường Đại học Quốc tế TPHCM", year=2024)
            metadata_image = ImageMetadata(category="campus", school="Trường Đại học Quốc tế TPHCM", location="Hà Nội")

            existing = session.query(TextChunk).filter_by(content=text_content).first()
            new_text = TextChunk(
                content=text_content,
                source_doc=None,
                category=metadata_text.category,
                school=metadata_text.school,
                year=metadata_text.year,
                entities=entities.model_dump()
            )
            if existing:
                print(f"TextChunk đã tồn tại với ID: {existing.chunk_id}")
            else:
                session.add(new_text)
                session.commit()
                print(f"Inserted TextChunk ID: {new_text.chunk_id}")

            if image_path:
                new_image = Image(
                    file_path=image_path,
                    s3_url="https://s3.amazonaws.com/admission-chatbot-images/campus.jpg",
                    caption="Ảnh campus Bách Khoa",
                    category=metadata_image.category,
                    school=metadata_image.school,
                    location=metadata_image.location,
                    width=800,
                    height=600,
                    file_size=1024,
                    file_format="jpg"
                )
                session.add(new_image)
                session.commit()

                # Insert relationship
                stmt = insert(text_image_association).values(
                    text_chunk_id=new_text.chunk_id,
                    image_id=new_image.image_id,
                    relation_score=0.85
                )
                session.execute(stmt)
                session.commit()

            print(f"Inserted TextChunk ID: {new_text.chunk_id}")
            if image_path:
                print(f"Inserted Image ID: {new_image.image_id}")
            print("Data inserted successfully!")

        except Exception as e:
            print(f"Error inserting data: {e}")