from config.database import init_db, drop_db  # Import function từ database.py

if __name__ == "__main__":
    init_db()  # Tạo tables
    # drop_db()
    print("Tables created successfully!")