"""
Main FastAPI application for RAG backend.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database.connection import db_manager
from backend.api.routes import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting RAG Backend API...")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"LLM Model: {settings.ollama_llm_model}")
    logger.info(f"Embedding Model: {settings.ollama_embedding_model}")
    
    try:
        # Initialize DB connection
        await db_manager.initialize()
        logger.info("Database initialized successfully")

        # Ensure tables exist
        logger.info("Creating database tables if not present...")
        await db_manager.create_tables()

        # Health check
        if await db_manager.health_check():
            logger.info("Database connection verified")
        else:
            logger.warning("Database health check failed")

        yield

    finally:
        logger.info("Shutting down RAG Backend API...")
        await db_manager.close()
        logger.info("Database connections closed")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="RAG Knowledge Assistant API powered by Ollama",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Knowledge Assistant API",
        "version": settings.api_version,
        "llm_model": settings.ollama_llm_model,
        "embedding_model": settings.ollama_embedding_model,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
