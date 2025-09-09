from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from contextlib import asynccontextmanager

# Import your existing services
from services.embedding_service import EmbeddingService
from services.text_processor import TextProcessor
from services.image_embedding import ImageEmbeddingService
from services.linking_service import LinkingService

# Import new retrieval components
from api.retrieval_api import router as retrieval_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global services
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Retrieval System...")
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        services['embedding'] = EmbeddingService()
        services['text_processor'] = TextProcessor()
        services['image_embedding'] = ImageEmbeddingService()
        services['linking'] = LinkingService()
        
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        logger.info("Shutting down Retrieval System...")

# Create FastAPI app
app = FastAPI(
    title="Multimodal Education Retrieval System",
    description="A comprehensive retrieval system for educational content with text and image search capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(retrieval_router)

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multimodal Education Retrieval System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "retrieval": "/retrieval/*",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    
    try:
        # Check service status
        service_status = {
            "embedding": "ok" if services.get('embedding') else "not_initialized",
            "text_processor": "ok" if services.get('text_processor') else "not_initialized",
            "image_embedding": "ok" if services.get('image_embedding') else "not_initialized",
            "linking": "ok" if services.get('linking') else "not_initialized"
        }
        
        all_ok = all(status == "ok" for status in service_status.values())
        
        return {
            "status": "healthy" if all_ok else "degraded",
            "services": service_status,
            "timestamp": "2025-01-01T00:00:00Z"  # Use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }

if __name__ == "__main__":
    uvicorn.run(
        "main_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )