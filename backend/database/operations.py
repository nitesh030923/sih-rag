"""
Database operations - CRUD and vector search functions.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, func, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database.models import Document, Chunk
from backend.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Document Operations
# ============================================================================

async def create_document(
    session: AsyncSession,
    title: str,
    source: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Document:
    """
    Create a new document.
    
    Args:
        session: Database session
        title: Document title
        source: Document source path
        content: Full document content
        metadata: Optional metadata dictionary
        
    Returns:
        Created Document instance
    """
    document = Document(
        title=title,
        source=source,
        content=content,
        metadata_=metadata or {}
    )
    session.add(document)
    await session.flush()
    return document


async def get_document(session: AsyncSession, document_id: UUID) -> Optional[Document]:
    """
    Get document by ID.
    
    Args:
        session: Database session
        document_id: Document UUID
        
    Returns:
        Document instance or None
    """
    result = await session.execute(
        select(Document).where(Document.id == document_id)
    )
    return result.scalar_one_or_none()


async def list_documents(
    session: AsyncSession,
    limit: int = 100,
    offset: int = 0
) -> List[Document]:
    """
    List all documents with pagination.
    
    Args:
        session: Database session
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        List of Document instances
    """
    result = await session.execute(
        select(Document)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, document_id: UUID) -> bool:
    """
    Delete a document and all its chunks.
    
    Args:
        session: Database session
        document_id: Document UUID
        
    Returns:
        True if deleted, False if not found
    """
    result = await session.execute(
        delete(Document).where(Document.id == document_id)
    )
    return result.rowcount > 0


async def delete_all_documents(session: AsyncSession) -> int:
    """
    Delete all documents and chunks.
    
    Args:
        session: Database session
        
    Returns:
        Number of documents deleted
    """
    result = await session.execute(delete(Document))
    return result.rowcount


async def get_document_count(session: AsyncSession) -> int:
    """
    Get total number of documents.
    
    Args:
        session: Database session
        
    Returns:
        Document count
    """
    result = await session.execute(select(func.count(Document.id)))
    return result.scalar_one()


# ============================================================================
# Chunk Operations
# ============================================================================

async def create_chunk(
    session: AsyncSession,
    document_id: UUID,
    content: str,
    embedding: List[float],
    chunk_index: int,
    token_count: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Chunk:
    """
    Create a new chunk.
    
    Args:
        session: Database session
        document_id: Parent document UUID
        content: Chunk text content
        embedding: Embedding vector
        chunk_index: Index of chunk in document
        token_count: Optional token count
        metadata: Optional metadata dictionary
        
    Returns:
        Created Chunk instance
    """
    chunk = Chunk(
        document_id=document_id,
        content=content,
        embedding=embedding,
        chunk_index=chunk_index,
        token_count=token_count,
        metadata_=metadata or {}
    )
    session.add(chunk)
    await session.flush()
    return chunk


async def get_chunk_count(session: AsyncSession) -> int:
    """
    Get total number of chunks.
    
    Args:
        session: Database session
        
    Returns:
        Chunk count
    """
    result = await session.execute(select(func.count(Chunk.id)))
    return result.scalar_one()


async def get_chunks_by_document(
    session: AsyncSession,
    document_id: UUID
) -> List[Chunk]:
    """
    Get all chunks for a document.
    
    Args:
        session: Database session
        document_id: Document UUID
        
    Returns:
        List of Chunk instances
    """
    result = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    return list(result.scalars().all())


# ============================================================================
# Vector Search Operations
# ============================================================================

class SearchResult:
    """Container for search results with metadata."""
    
    def __init__(
        self,
        chunk_id: UUID,
        document_id: UUID,
        content: str,
        similarity: float,
        chunk_metadata: Dict[str, Any],
        document_title: str,
        document_source: str
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.similarity = similarity
        self.chunk_metadata = chunk_metadata
        self.document_title = document_title
        self.document_source = document_source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "content": self.content,
            "similarity": self.similarity,
            "metadata": self.chunk_metadata,
            "document_title": self.document_title,
            "document_source": self.document_source
        }


async def vector_search(
    session: AsyncSession,
    query_embedding: List[float],
    limit: int = None,
    similarity_threshold: float = None
) -> List[SearchResult]:
    """
    Perform vector similarity search using cosine distance.
    
    Args:
        session: Database session
        query_embedding: Query embedding vector
        limit: Maximum number of results (defaults to settings.top_k_results)
        similarity_threshold: Minimum similarity score (defaults to settings.similarity_threshold)
        
    Returns:
        List of SearchResult instances ordered by similarity (descending)
    """
    if limit is None:
        limit = settings.top_k_results
    
    if similarity_threshold is None:
        similarity_threshold = settings.similarity_threshold
    
    # Use pgvector's cosine distance operator (<=>)
    # Similarity = 1 - distance (so higher is better)
    query = text("""
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> :query_embedding) AS similarity,
            c.metadata,
            d.title AS document_title,
            d.source AS document_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
            AND 1 - (c.embedding <=> :query_embedding) >= :threshold
        ORDER BY c.embedding <=> :query_embedding
        LIMIT :limit
    """)
    
    # Convert embedding to string format for pgvector
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    logger.info(f"Vector search: threshold={similarity_threshold}, limit={limit}")
    
    result = await session.execute(
        query,
        {
            "query_embedding": embedding_str,
            "threshold": similarity_threshold,
            "limit": limit
        }
    )
    
    rows = result.fetchall()
    
    logger.info(f"Vector search returned {len(rows)} results")
    for row in rows:
        logger.info(f"  - {row.document_title}: similarity={row.similarity:.4f}")
    
    return [
        SearchResult(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            content=row.content,
            similarity=float(row.similarity),
            chunk_metadata=row.metadata,
            document_title=row.document_title,
            document_source=row.document_source
        )
        for row in rows
    ]


async def search_knowledge_base(
    session: AsyncSession,
    query_embedding: List[float],
    limit: int = None
) -> str:
    """
    Search knowledge base and format results as context string.
    
    Args:
        session: Database session
        query_embedding: Query embedding vector
        limit: Maximum number of results
        
    Returns:
        Formatted context string with sources
    """
    results = await vector_search(session, query_embedding, limit=limit)
    
    if not results:
        return "No relevant information found in the knowledge base."
    
    # Format results with sources
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}: {result.document_title}]\n{result.content}"
        )
    
    return "\n\n".join(context_parts)
