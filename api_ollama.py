"""
FastAPI Backend for RAG Chatbot with Native Ollama Support
===========================================================
Simplified version that always searches knowledge base without function calling
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_agent_ollama import chat, initialize_db, close_db, search_knowledge_base, generate_response_stream

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the assistant")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional conversation history for context"
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    conversation_history: List[ChatMessage] = Field(..., description="Updated conversation history")


class HealthResponse(BaseModel):
    status: str
    database: str
    knowledge_base: dict
    model_info: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup and cleanup on shutdown."""
    logger.info("Starting up FastAPI application...")
    
    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL environment variable is required")
    
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
    title="RAG Knowledge Assistant API (Ollama)",
    description="REST API for conversational access to knowledge base using RAG with Ollama",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Knowledge Assistant API (Ollama Native)",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and database health status."""
    try:
        from rag_agent_ollama import db_pool
        
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
            },
            model_info={
                "provider": os.getenv("LLM_PROVIDER", "ollama"),
                "model": os.getenv("LLM_CHOICE", "mistral"),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to the RAG assistant and get a response.
    
    The assistant automatically searches the knowledge base and provides
    contextual answers with source citations.
    """
    try:
        # Convert message history to dict format
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        # Get response
        result = await chat(request.message, conversation_history)
        
        # Convert back to ChatMessage format
        updated_history = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in result["conversation_history"]
        ]
        
        return ChatResponse(
            response=result["response"],
            conversation_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(request: ChatRequest):
    """
    Send a message to the RAG assistant and get a streaming response.
    
    Returns server-sent events (SSE) with the response being generated in real-time.
    """
    async def generate():
        try:
            # Convert message history to dict format
            conversation_history = None
            if request.conversation_history:
                conversation_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.conversation_history
                ]
            
            # Search knowledge base first
            logger.info(f"Searching knowledge base for: {request.message}")
            yield f"data: {{'status': 'searching'}}\n\n"
            
            context = await search_knowledge_base(request.message)
            
            yield f"data: {{'status': 'generating'}}\n\n"
            
            # Stream the response
            full_response = ""
            async for chunk in generate_response_stream(request.message, context, conversation_history):
                full_response += chunk
                # Send chunk as SSE
                import json
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Send completion event with full response
            yield f"data: {json.dumps({'status': 'done', 'response': full_response})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            import json
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "api_ollama:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
