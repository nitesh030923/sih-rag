# Docling RAG Agent

An intelligent RAG (Retrieval Augmented Generation) system with FastAPI backend and Streamlit UI that provides conversational access to a knowledge base stored in PostgreSQL with PGVector. Uses **Ollama** for fully offline/local LLM inference and embeddings. Supports multiple document formats including audio files with Whisper transcription.

## 🌟 Features

- 🌐 **FastAPI Backend** - RESTful API with streaming support
- 🎨 **Streamlit UI** - Beautiful chat interface with real-time streaming
- 🏠 **Fully Local/Offline** - Uses Ollama (no OpenAI API required)
- 🔍 **Semantic Search** - Vector-embedded documents with PGVector
- 📚 **Context-Aware Responses** - RAG pipeline with source citations
- 🔄 **Real-time Streaming** - See responses as they're generated
- 💾 **Scalable Storage** - PostgreSQL with PGVector extension
- 🧠 **Conversation History** - Multi-turn context maintenance
- 🎙️ **Audio Transcription** - Whisper ASR for MP3 files (optional)
- 📄 **Multi-Format Support** - PDF, Word, Excel, PowerPoint, Markdown, Text, Audio

## 🎓 New to Docling?

**Start with the tutorials!** Check out the [`docling_basics/`](./docling_basics/) folder for progressive examples:

1. **Simple PDF Conversion** - Basic document processing
2. **Multiple Format Support** - PDF, Word, PowerPoint handling  
3. **Audio Transcription** - Speech-to-text with Whisper (optional)
4. **Hybrid Chunking** - Intelligent chunking for RAG systems

[**→ Go to Docling Basics**](./docling_basics/)

## 📋 Prerequisites

- **Python 3.9+**
- **PostgreSQL with PGVector** (Docker provided)
- **Ollama** installed and running locally
- **FFmpeg** (optional, for audio transcription)

## 🚀 Quick Start

### 1. Install Ollama

Download and install from [ollama.ai](https://ollama.ai)

```bash
# Pull required models
ollama pull mistral          # LLM for chat
ollama pull nomic-embed-text # Embeddings (768-dim)

# Verify installation
ollama list
```

### 2. Start PostgreSQL with Docker

```bash
# Start PostgreSQL with PGVector
docker-compose up -d postgres

# Verify it's running
docker ps
```

The database will automatically initialize with the correct schema (768-dimensional vectors for nomic-embed-text).

### 3. Install Python Dependencies

```bash
# Install dependencies using UV
uv sync

# Or with pip
pip install -r requirements.txt
```

### 4. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

**Required variables:**
```env
# Database (Docker PostgreSQL on port 5433)
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/offrag

# Ollama Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_CHOICE=mistral
EMBEDDING_MODEL=nomic-embed-text
```

### 5. Ingest Documents

Add your documents to the `documents/` folder:

**Supported Formats:**
- 📄 PDF (`.pdf`)
- 📝 Word (`.docx`)
- 📊 PowerPoint (`.pptx`)
- 📈 Excel (`.xlsx`)
- 📋 Markdown (`.md`)
- 📃 Text (`.txt`)
- 🎵 Audio (`.mp3`) - requires FFmpeg and torch (optional)

```bash
# Start Ollama (if not running)
ollama serve

# Ingest documents
uv run python -m ingestion.ingest --documents documents/
```

**⚠️ Note:** Ingestion clears existing data before adding new documents.

### 6. Start the Backend API

```bash
# Start FastAPI server
uv run python api_ollama.py
```

API will be available at:
- **Base**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### 7. Launch Streamlit UI

```bash
# In a new terminal
uv run streamlit run app.py
```

Open browser to: **http://localhost:8501**

## 🏗️ Architecture

```
┌──────────────┐
│  Streamlit   │ ◄── User Interface (Port 8501)
│      UI      │
└──────┬───────┘
       │ HTTP
       ▼
┌──────────────┐
│   FastAPI    │ ◄── REST API (Port 8000)
│   Backend    │     /health, /chat, /chat/stream
└──────┬───────┘
       │
   ┌───┴────┬──────────┐
   │        │          │
   ▼        ▼          ▼
┌─────┐ ┌─────┐ ┌──────────┐
│Ollama│ │PG   │ │ Docling  │
│(LLM)│ │Vec  │ │(Ingest)  │
└─────┘ └─────┘ └──────────┘
localhost  Docker   Document
:11434    :5433    Processing
```

**Components:**
- **Ollama** - Runs on host machine (mistral + nomic-embed-text)
- **PostgreSQL** - Docker container with PGVector (768-dim vectors)
- **FastAPI** - Backend API with streaming endpoints
- **Streamlit** - Frontend UI with real-time chat
- **Docling** - Document processing and chunking

## 🎙️ Audio Transcription (Optional)

Audio files are transcribed using **OpenAI Whisper Turbo** via Docling:

### Setup for Audio:

1. **Install FFmpeg**:
```bash
# Windows (Chocolatey - run as admin)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
apt-get install ffmpeg
```

2. **Install audio dependencies**:
```bash
# Already included in pyproject.toml
uv sync
```

3. **Add MP3 files** to `documents/` and run ingestion

### How it works:
- Whisper Turbo model transcribes audio
- Generates markdown with timestamps: `[time: 0.0-4.0] Text here`
- Audio content becomes fully searchable
- Works with 90+ languages

See [AUDIO_SETUP.md](./AUDIO_SETUP.md) for detailed setup instructions.

## 🔑 Key Components

### FastAPI Backend (`api_ollama.py`)
- **REST API** with `/health`, `/chat`, `/chat/stream` endpoints
- **Streaming support** for real-time responses
- **Async architecture** with asyncpg connection pooling
- **Native Ollama integration** using `/api/generate` endpoint

### Streamlit UI (`app.py`)
- **Real-time chat interface** with streaming responses
- **Sidebar stats** showing KB size and model info
- **Dark theme** with custom CSS styling
- **Conversation history** maintained across messages

### RAG Agent (`rag_agent_ollama.py`)
- **Always-search pattern** (no function calling dependency)
- **Semantic search** using nomic-embed-text (768-dim)
- **Context injection** into prompts for LLM
- **Native Ollama API** for better compatibility

### Ingestion Pipeline (`ingestion/`)
- **Document processing** with Docling (multi-format)
- **Hybrid chunking** for semantic coherence
- **Embedding generation** with Ollama
- **PostgreSQL storage** with vector similarity

### Database Schema

**Documents table:**
- Stores original documents with metadata
- Fields: `id`, `title`, `source`, `content`, `metadata`

**Chunks table:**
- Stores text chunks with embeddings
- `embedding`: vector(768) for nomic-embed-text
- `chunk_index`, `token_count`, `metadata`

**match_chunks() function:**
- Vector similarity search with cosine distance
- Returns chunks with similarity scores
- Configurable threshold and result limit

## Performance Optimization

### Database Connection Pooling
```python
db_pool = await asyncpg.create_pool(
    DATABASE_URL,
    min_size=2,
    max_size=10,
    command_timeout=60
)
```

### Embedding Cache
The embedder includes built-in caching for frequently searched queries, reducing API calls and latency.

### Streaming Responses
Token-by-token streaming provides immediate feedback to users while the LLM generates responses:
```python
async with agent.run_stream(user_input, message_history=history) as result:
    async for text in result.stream_text(delta=False):
        print(f"\rAssistant: {text}", end="", flush=True)
```

## 🐳 Docker Deployment

### Start with Docker Compose

```bash
# Start PostgreSQL only
docker-compose up -d postgres

# Or start PostgreSQL + API + UI (requires Ollama on host)
docker-compose up -d
```

**Services:**
- `postgres` - PostgreSQL with PGVector (port 5433)
- `api` - FastAPI backend (port 8000)
- `ui` - Streamlit interface (port 8501)
- `ingestion` - Document ingestion (manual profile)

### Run Ingestion in Docker

```bash
docker-compose --profile ingestion up ingestion
```

### Notes:
- **Ollama runs on host** machine at `http://host.docker.internal:11434`
- Models must be pulled on host: `ollama pull mistral`
- PostgreSQL data persists in Docker volume
- Audio transcription works but increases image size (~2GB due to PyTorch)

## API Reference

### search_knowledge_base Tool

```python
async def search_knowledge_base(
    ctx: RunContext[None],
    query: str,
    limit: int = 5
) -> str:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query to find relevant information
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with source citations
    """
```

### Database Functions

```sql
-- Vector similarity search
SELECT * FROM match_chunks(
    query_embedding::vector(1536),
    match_count INT,
    similarity_threshold FLOAT DEFAULT 0.7
)
```

Returns chunks with:
- `id`: Chunk UUID
- `content`: Text content
- `embedding`: Vector embedding
- `similarity`: Cosine similarity score (0-1)
- `document_title`: Source document title
- `document_source`: Source document path

## Project Structure

```
docling-rag-agent/
├── cli.py                   # Enhanced CLI with colors and features (recommended)
├── rag_agent.py             # Basic CLI agent with PydanticAI
├── ingestion/
│   ├── ingest.py            # Document ingestion pipeline
│   ├── embedder.py          # Embedding generation with caching
│   └── chunker.py           # Document chunking logic
├── utils/
│   ├── providers.py         # OpenAI model/client configuration
│   ├── db_utils.py          # Database connection pooling
│   └── models.py            # Pydantic models for config
├── sql/
│   └── schema.sql           # PostgreSQL schema with PGVector
├── documents/               # Sample documents for ingestion
├── pyproject.toml           # Project dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```