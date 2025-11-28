"""
Simplified RAG Agent for Ollama - Always searches knowledge base
================================================================
This version doesn't rely on function calling, making it work better with Ollama models.
"""

import asyncio
import asyncpg
import logging
import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Global database pool
db_pool = None

# Get provider configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama').lower()
LLM_MODEL = os.getenv('LLM_CHOICE', 'mistral')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')


async def initialize_db():
    """Initialize database connection pool."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database connection pool initialized")


async def close_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


async def search_knowledge_base(query: str, limit: int = 5) -> str:
    """
    Search the knowledge base using semantic similarity.
    """
    try:
        if not db_pool:
            await initialize_db()

        # Generate embedding for query
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)

        # Convert to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Search using match_chunks function
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        if not results:
            return "No relevant information found in the knowledge base."

        # Build context with sources
        context_parts = []
        for i, row in enumerate(results, 1):
            content = row['content']
            doc_title = row['document_title']
            context_parts.append(f"[Source {i}: {doc_title}]\n{content}")

        return "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"Error searching knowledge base: {str(e)}"


def _build_prompt(user_query: str, context: str, conversation_history: Optional[list] = None) -> str:
    """Build prompt for Ollama."""
    # Limit context to avoid timeouts
    max_context_length = 3000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "\n...(context truncated)"
    
    # Build the prompt - simple format works better with Mistral
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. Synthesize the information into a clear, natural answer. DO NOT just list the sources - provide a coherent response.

Context from knowledge base:
{context}

User Question: {user_query}

Your Answer (synthesize the information above into a clear response):"""
    
    return prompt


async def generate_response_stream(
    user_query: str,
    context: str,
    conversation_history: Optional[list] = None
):
    """
    Generate streaming response using Ollama with retrieved context.
    Yields chunks of text as they're generated.
    """
    try:
        import httpx
        import json
        
        full_prompt = _build_prompt(user_query, context, conversation_history)
        
        # Make streaming request to Ollama using native API
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024,
                    }
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            continue
            
    except httpx.ReadTimeout:
        logger.error("Model response timed out")
        yield "⏱️ The model took too long to respond. Please try a simpler question or use a smaller/faster model."
    except Exception as e:
        logger.error(f"Response generation failed: {e}", exc_info=True)
        yield f"❌ Error generating response: {str(e)}"


async def generate_response(
    user_query: str,
    context: str,
    conversation_history: Optional[list] = None
) -> str:
    """
    Generate response using Ollama with retrieved context (non-streaming).
    """
    try:
        import httpx
        
        full_prompt = _build_prompt(user_query, context, conversation_history)
        
        # Make request to Ollama using native API
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            return result.get("response", "").strip()
            
    except httpx.ReadTimeout:
        logger.error("Model response timed out")
        return "The model took too long to respond. Please try a simpler question or use a smaller/faster model."
    except Exception as e:
        logger.error(f"Response generation failed: {e}", exc_info=True)
        return f"Error generating response: {str(e)}"


async def chat(user_query: str, conversation_history: Optional[list] = None) -> dict:
    """
    Main chat function that searches and generates response.
    
    Returns:
        dict with 'response' and 'conversation_history'
    """
    try:
        # Search knowledge base
        logger.info(f"Searching knowledge base for: {user_query}")
        context = await search_knowledge_base(user_query)
        
        # Generate response
        logger.info("Generating response...")
        response = await generate_response(user_query, context, conversation_history)
        
        # Update conversation history
        if conversation_history is None:
            conversation_history = []
        
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": response})
        
        return {
            "response": response,
            "conversation_history": conversation_history
        }
        
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        return {
            "response": f"Sorry, I encountered an error: {str(e)}",
            "conversation_history": conversation_history or []
        }


async def run_cli():
    """Run the agent in an interactive CLI."""
    await initialize_db()

    print("=" * 60)
    print("RAG Knowledge Assistant (Ollama)")
    print("=" * 60)
    print("Ask me anything about the knowledge base!")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()

    conversation_history = []

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for using the knowledge assistant. Goodbye!")
                break

            print("Assistant: ", end="", flush=True)

            result = await chat(user_input, conversation_history)
            print(result['response'])
            conversation_history = result['conversation_history']
            
            print()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await close_db()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\n\nShutting down...")
