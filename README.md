# RAG Backend with Ollama

A production-ready **Retrieval-Augmented Generation (RAG)** backend built with FastAPI, SQLAlchemy, PostgreSQL/PGVector, and Ollama for fully local/offline LLM inference.

## 🌟 Features

- ✅ **Ollama-based** - fully offline
- ✅ **FastAPI Backend** - RESTful API with streaming support
- ✅ **SQLAlchemy ORM** - Clean database layer with async support
- ✅ **PGVector Integration** - Semantic search with 768-dim embeddings
- ✅ **Multi-format Support** - PDF, Word, Excel, PowerPoint, Markdown, Audio
- ✅ **Hybrid Chunking** - Intelligent document splitting with Docling
- ✅ **Audio Transcription** - Whisper ASR for MP3/WAV files
- ✅ **Clean Architecture** - Organized, maintainable, testable code

## 📁 Project Structure

```
sih-rag/
├── backend/
│   ├── api/                    # API layer
│   │   ├── routes.py           # All endpoints
│   │   └── schemas.py          # Pydantic models
│   ├── core/                   # Business logic
│   │   ├── ollama_client.py    # Ollama HTTP client
│   │   └── rag_engine.py       # RAG orchestration
│   ├── database/               # Data layer
│   │   ├── connection.py       # SQLAlchemy setup
│   │   ├── models.py           # Document & Chunk models
│   │   └── operations.py       # CRUD & vector search
│   ├── ingestion/              # Document processing
│   │   ├── pipeline.py         # Main ingestion script
│   │   ├── chunker.py          # Document chunking
│   │   └── embedder.py         # Embedding generation
│   ├── config.py               # Centralized settings
│   └── main.py                 # FastAPI app
├── documents/                  # Place documents here
├── sql/
│   └── schema.sql              # PostgreSQL schema
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Backend container
├── pyproject.toml              # Dependencies
└── .env.example                # Environment template
```

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended)
- **Ollama** installed and running locally
- **FFmpeg** (optional, for audio transcription)

### Option 1: Docker Deployment (Recommended)

#### 1. Install Ollama

Download from [ollama.ai](https://ollama.ai)

```bash
# Pull required models
ollama pull mistral          # LLM for chat
ollama pull nomic-embed-text # 768-dim embeddings

# Verify
ollama list
```

#### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` to point to your host's Ollama (Docker containers need host network):
```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/offrag
OLLAMA_BASE_URL=http://host.docker.internal:11434  # For Mac/Windows
# OLLAMA_BASE_URL=http://172.17.0.1:11434           # For Linux
OLLAMA_LLM_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

#### 3. Start All Services

```bash
# Start PostgreSQL + Backend
docker-compose up -d

# Check logs
docker-compose logs -f backend
```

API available at:
- **Base**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

#### 4. Upload Documents

**Option A: Upload via API (Recommended)**

Upload individual files through the REST API:

```bash
# Using curl
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"

# Or use Swagger UI
# Go to http://localhost:8000/docs → /upload endpoint
```

**Option B: Batch Ingestion**

Process all files in the `documents/` folder at once:

```bash
# Using API endpoint
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_path": "documents", "clean_existing": false}'

# Or use ingestion container
docker-compose --profile ingestion up ingestion
```

**Supported formats:**
- 📄 PDF, Word, PowerPoint, Excel
- 📝 Markdown, Text
- 🎵 MP3, WAV, M4A, FLAC

#### 5. Stop Services

```bash
docker-compose down
```

---

### Option 2: Local Development (Without Docker)

Use this for development only. Requires Python 3.9+ and PostgreSQL 15.

#### 1. Install Ollama

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

#### 2. Start PostgreSQL

```bash
docker-compose up -d postgres
```

#### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/offrag
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

#### 4. Install Dependencies

```bash
pip install uv
uv pip install -e .
```

#### 5. Start Backend

```bash
python -m backend.main
```

API available at http://localhost:8000/docs

#### 6. Upload Documents

Use the `/upload` API endpoint at http://localhost:8000/docs or:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@documents/your-file.pdf"
```

---

## 🐳 Docker Services

- `postgres` - PostgreSQL 15 with PGVector extension (port 5433)
- `backend` - FastAPI application with auto-reload (port 8000)
- `ingestion` - One-time document ingestion job (profile: ingestion)

## 🔌 API Endpoints

### Health Check

```bash
GET /health
```

Returns database status, Ollama status, and knowledge base stats.

### Upload File

```bash
POST /upload
Content-Type: multipart/form-data

file: <your-file>
```

Upload and process a single document (PDF, DOCX, PPTX, XLSX, MD, TXT, MP3, WAV).

### Batch Ingestion

```bash
POST /ingest
Content-Type: application/json

{
  "documents_path": "documents",
  "clean_existing": false
}
```

Process all documents in a folder. Useful for initial bulk import.

### Chat (Non-streaming)

```bash
POST /chat
Content-Type: application/json

{
  "message": "What are the main topics?",
  "conversation_history": [...]  # optional
}
```

### Chat (Streaming)

```bash
POST /chat/stream
Content-Type: application/json

{
  "message": "Explain the key concepts",
  "conversation_history": [...]  # optional
}
```

Returns Server-Sent Events (SSE) with streaming response.

### Search Knowledge Base

```bash
POST /search
Content-Type: application/json

{
  "query": "machine learning",
  "limit": 5
}
```

Returns relevant chunks with similarity scores.

### List Documents

```bash
GET /documents?limit=100&offset=0
```

## ⚙️ Configuration

All settings in `.env` or environment variables:

### Database
- `DATABASE_URL` - PostgreSQL connection string
- `DB_POOL_MIN_SIZE` - Minimum pool size (default: 5)
- `DB_POOL_MAX_SIZE` - Maximum pool size (default: 20)

### Ollama
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_LLM_MODEL` - Chat model (default: mistral)
- `OLLAMA_EMBEDDING_MODEL` - Embedding model (default: nomic-embed-text)
- `OLLAMA_TIMEOUT` - Request timeout in seconds (default: 300)

### RAG
- `EMBEDDING_DIMENSIONS` - Vector dimensions (default: 768)
- `TOP_K_RESULTS` - Number of chunks to retrieve (default: 5)
- `SIMILARITY_THRESHOLD` - Minimum similarity score (default: 0.7)
- `MAX_TOKENS_PER_CHUNK` - Max tokens per chunk (default: 512)

### API
- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8000)
- `LOG_LEVEL` - Logging level (default: INFO)

## 🏗️ Architecture

### Data Flow

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Ingestion Pipeline             │
│  1. Read (Docling converts)     │
│  2. Chunk (HybridChunker)       │
│  3. Embed (Ollama)              │
│  4. Store (PostgreSQL/PGVector) │
└──────┬──────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  PostgreSQL          │
│  - documents table   │
│  - chunks table      │
│  - vector(768)       │
└──────┬───────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  RAG Engine                     │
│  1. User Query                  │
│  2. Generate Embedding (Ollama) │
│  3. Vector Search (Cosine)      │
│  4. Build Prompt + Context      │
│  5. LLM Response (Ollama)       │
└─────────────────────────────────┘
```

### Components

1. **Ollama Client** (`backend/core/ollama_client.py`)
   - HTTP client for Ollama API
   - Handles embeddings and chat completions
   - Streaming support

2. **RAG Engine** (`backend/core/rag_engine.py`)
   - Always-search pattern (no function calling)
   - Combines retrieval + generation
   - Context injection into prompts

3. **Database Layer** (`backend/database/`)
   - SQLAlchemy async models
   - PGVector integration
   - CRUD operations + vector search

4. **Ingestion Pipeline** (`backend/ingestion/`)
   - Multi-format document processing
   - Docling HybridChunker for intelligent splitting
   - Batch embedding generation

## 🧪 Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format
black backend/

# Lint
ruff check backend/

# Type check
mypy backend/
```

## 📊 Database Schema

### Documents Table

| Column     | Type      | Description              |
|------------|-----------|--------------------------|
| id         | UUID      | Primary key              |
| title      | TEXT      | Document title           |
| source     | TEXT      | Source path              |
| content    | TEXT      | Full document content    |
| metadata   | JSONB     | Additional metadata      |
| created_at | TIMESTAMP | Creation timestamp       |
| updated_at | TIMESTAMP | Last update timestamp    |

### Chunks Table

| Column       | Type       | Description                |
|--------------|------------|----------------------------|
| id           | UUID       | Primary key                |
| document_id  | UUID       | Foreign key to documents   |
| content      | TEXT       | Chunk text content         |
| embedding    | vector(768)| Embedding vector           |
| chunk_index  | INTEGER    | Chunk position in document |
| metadata     | JSONB      | Chunk metadata             |
| token_count  | INTEGER    | Number of tokens           |
| created_at   | TIMESTAMP  | Creation timestamp         |

## 🐛 Troubleshooting

### Ollama Not Responding

```bash
# Check Ollama is running
ollama list

# Restart Ollama
ollama serve
```

### Database Connection Error

```bash
# Check PostgreSQL is running
docker-compose ps

# Check logs
docker-compose logs postgres
```

### Import Errors

```bash
# Reinstall dependencies
uv pip install -e . --force-reinstall
```

---

**Built with ❤️ using Ollama, FastAPI, SQLAlchemy, and PostgreSQL/PGVector**
