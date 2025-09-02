"""
Database configuration và connection management.
SQLAlchemy setup với connection pooling.
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from .settings import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            # Create SQLAlchemy engine với connection pooling
            self.engine = create_engine(
                settings.DATABASE_URL,
                pool_size=20,           # Số connection tối đa trong pool
                max_overflow=30,        # Số connection overflow
                pool_recycle=3600,      # Recycle connection sau 1 giờ
                pool_pre_ping=True,     # Test connection trước khi sử dụng
                echo=False              # Set True để log SQL queries
            )
            
            # Tạo session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager để get database session"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Tạo tất cả tables trong database"""
        from models.database_models import Base
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop tất cả tables (chỉ dùng cho development)"""
        from models.database_models import Base
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    def get_engine(self):
        """Get SQLAlchemy engine"""
        return self.engine

# Singleton instance
db_manager = DatabaseManager()

# Convenience functions
def get_db_session():
    """Get database session - để sử dụng trong dependency injection"""
    return db_manager.get_session()

def init_db():
    """Initialize database với tables"""
    db_manager.create_tables()

def drop_db():
    db_manager.drop_tables()