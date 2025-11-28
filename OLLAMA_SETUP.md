# Setting Up Offline RAG System with Ollama

This guide will help you set up a fully offline RAG system using Ollama and Llama 3.1.

## Prerequisites

1. **Install Ollama** from https://ollama.ai/download
2. **PostgreSQL with pgvector** (already running via Docker)

## Step 1: Install and Pull Ollama Models

```powershell
# Install Ollama (if not already installed)
# Download from: https://ollama.ai/download

# Pull Llama 3.1 model (8B variant recommended for local use)
ollama pull llama3.1

# Pull the embedding model
ollama pull nomic-embed-text
```

## Step 2: Verify Ollama is Running

```powershell
# Check if Ollama is running
ollama list

# Test the model
ollama run llama3.1 "Hello, how are you?"
```

## Step 3: Update Database Schema

The embedding model `nomic-embed-text` uses 768 dimensions (vs 1536 for OpenAI).
You need to recreate the database with the new schema:

```powershell
# Drop and recreate with Ollama schema
Get-Content sql/schema_ollama.sql | docker exec -i rag_postgres psql -U postgres -d offrag
```

## Step 4: Re-ingest Documents

Since we changed the embedding dimensions, re-ingest your documents:

```powershell
# Ingest documents with Ollama embeddings
uv run python -m ingestion.ingest --documents documents/
```

## Step 5: Run the System

### Option A: CLI Interface
```powershell
uv run python cli.py
```

### Option B: FastAPI Backend
```powershell
uv run python api.py
```

## Model Options

### LLM Models (for chat)
- `llama3.1` (8B) - Fast, good quality
- `llama3.1:70b` - Higher quality, slower
- `mistral` - Alternative option
- `qwen2.5` - Good for coding

### Embedding Models
- `nomic-embed-text` (768 dims) - Recommended, fast
- `mxbai-embed-large` (1024 dims) - Higher quality
- `all-minilm` (384 dims) - Faster, smaller

## Configuration

Your `.env` file is already configured for Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
LLM_CHOICE=llama3.1
EMBEDDING_MODEL=nomic-embed-text
```

## Troubleshooting

### Ollama not responding
```powershell
# Check if Ollama service is running
Get-Process ollama

# Restart Ollama if needed
# Close Ollama and restart from Start menu
```

### Wrong embedding dimensions
If you see errors about embedding dimensions, make sure you:
1. Used the correct schema file (`schema_ollama.sql`)
2. Re-ingested documents after changing the schema

### Model not found
```powershell
# List available models
ollama list

# Pull missing model
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Performance Tips

1. **Use smaller models** for faster responses:
   - `llama3.1` (8B) instead of `llama3.1:70b`
   
2. **Adjust batch size** in ingestion for faster processing:
   ```powershell
   uv run python -m ingestion.ingest --documents documents/ --chunk-size 500
   ```

3. **GPU acceleration**: Ollama automatically uses GPU if available

## Switching Back to OpenAI

If you want to switch back to OpenAI:

1. Update `.env`:
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   LLM_CHOICE=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-small
   ```

2. Restore OpenAI schema:
   ```powershell
   Get-Content sql/schema.sql | docker exec -i rag_postgres psql -U postgres -d offrag
   ```

3. Re-ingest documents:
   ```powershell
   uv run python -m ingestion.ingest --documents documents/
   ```
