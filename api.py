"""
FastAPI Backend for RAG Chatbot
================================
Provides REST API endpoints for the RAG knowledge assistant
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_agent import agent, initialize_db, close_db

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the assistant")
    message_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional conversation history for context"
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    message_history: List[ChatMessage] = Field(..., description="Updated conversation history")


class HealthResponse(BaseModel):
    status: str
    database: str
    knowledge_base: dict


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup and cleanup on shutdown."""
    logger.info("Starting up FastAPI application...")
    
    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL environment variable is required")
    
    # Check provider-specific requirements
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is required when using OpenAI provider")
    
    # Initialize database connection pool
    await initialize_db()
    logger.info("Database initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down FastAPI application...")
    await close_db()
    logger.info("Database connection closed")


# Create FastAPI app
app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="REST API for conversational access to knowledge base using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Knowledge Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and database health status."""
    try:
        from rag_agent import db_pool
        
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        # Check database connection and get stats
        async with db_pool.acquire() as conn:
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        
        return HealthResponse(
            status="healthy",
            database="connected",
            knowledge_base={
                "documents": doc_count,
                "chunks": chunk_count
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the RAG assistant and get a response.
    
    The assistant will search the knowledge base and provide contextual answers
    with source citations.
    """
    try:
        # Convert message history to the format expected by pydantic-ai
        message_history = []
        if request.message_history:
            for msg in request.message_history:
                message_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Run the agent
        result = await agent.run(
            request.message,
            message_history=message_history
        )
        
        # Get the response text
        response_text = result.output if hasattr(result, 'output') else str(result)
        
        # Build updated message history
        updated_history = []
        for msg in result.all_messages():
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                # Handle different content types
                content = msg.content
                if isinstance(content, list):
                    # Extract text from list of content parts
                    text_parts = [p.get('text', '') if isinstance(p, dict) else str(p) for p in content]
                    content_str = ' '.join(text_parts)
                elif isinstance(content, str):
                    content_str = content
                else:
                    content_str = str(content)
                
                updated_history.append(
                    ChatMessage(
                        role=msg.role,
                        content=content_str
                    )
                )
        
        return ChatResponse(
            response=response_text,
            message_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream responses from the RAG assistant (Server-Sent Events).
    
    Note: This endpoint requires SSE-compatible client implementation.
    """
    from fastapi.responses import StreamingResponse
    
    async def generate_stream():
        try:
            # Convert message history
            message_history = []
            if request.message_history:
                for msg in request.message_history:
                    message_history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Stream the response
            async with agent.run_stream(
                request.message,
                message_history=message_history
            ) as result:
                async for text in result.stream_text(delta=True):
                    yield f"data: {text}\n\n"
                
                yield "data: [DONE]\n\n"
                
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
