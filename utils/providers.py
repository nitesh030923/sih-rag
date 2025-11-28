"""
Provider configuration supporting both OpenAI and Ollama models.
"""

import os
from typing import Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm_provider() -> str:
    """Get the LLM provider (openai or ollama)."""
    return os.getenv('LLM_PROVIDER', 'ollama').lower()


def get_llm_model_name() -> str:
    """Get the LLM model name."""
    provider = get_llm_provider()
    if provider == 'ollama':
        return os.getenv('LLM_CHOICE', 'llama3.1')
    else:
        return os.getenv('LLM_CHOICE', 'gpt-4o-mini')


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get client for embeddings (OpenAI or Ollama).
    
    Returns:
        Configured client for embeddings
    """
    provider = get_llm_provider()
    
    if provider == 'ollama':
        # Use Ollama's OpenAI-compatible API
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        # Ollama's OpenAI-compatible endpoint is at /v1
        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'
        return openai.AsyncOpenAI(
            base_url=base_url,
            api_key='ollama'  # Ollama doesn't require API key but client needs one
        )
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return openai.AsyncOpenAI(api_key=api_key)


def get_embedding_model() -> str:
    """
    Get embedding model name.
    
    Returns:
        Embedding model name
    """
    provider = get_llm_provider()
    if provider == 'ollama':
        return os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
    else:
        return os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')


def get_embedding_dimensions() -> int:
    """
    Get the embedding dimensions for the current model.
    
    Returns:
        Number of dimensions in embeddings
    """
    provider = get_llm_provider()
    embedding_model = get_embedding_model()
    
    if provider == 'ollama':
        # Common Ollama embedding dimensions
        if 'nomic-embed-text' in embedding_model:
            return 768
        elif 'mxbai-embed-large' in embedding_model:
            return 1024
        elif 'all-minilm' in embedding_model:
            return 384
        else:
            return 768  # Default
    else:
        # OpenAI dimensions
        if 'text-embedding-3-small' in embedding_model:
            return 1536
        elif 'text-embedding-3-large' in embedding_model:
            return 3072
        elif 'text-embedding-ada-002' in embedding_model:
            return 1536
        else:
            return 1536  # Default


def validate_configuration() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if configuration is valid
    """
    provider = get_llm_provider()
    
    required_vars = ['DATABASE_URL']
    
    if provider == 'openai':
        required_vars.append('OPENAI_API_KEY')
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True


def get_model_info() -> dict:
    """
    Get information about current model configuration.
    
    Returns:
        Dictionary with model configuration info
    """
    provider = get_llm_provider()
    return {
        "llm_provider": provider,
        "llm_model": get_llm_model_name(),
        "embedding_provider": provider,
        "embedding_model": get_embedding_model(),
        "embedding_dimensions": get_embedding_dimensions()
    }