"""
Configuration settings for the RAG backend.
All settings are loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings using Pydantic Settings v2."""
    
    # Database Configuration
    database_url: str
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_command_timeout: int = 60
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "mistral"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout: int = 300  # 5 minutes for slow responses
    
    # Embedding Configuration
    embedding_dimensions: int = 768  # nomic-embed-text default
    max_tokens_per_chunk: int = 512
    
    # RAG Configuration
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "RAG Knowledge Assistant API"
    api_version: str = "2.0.0"
    cors_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
