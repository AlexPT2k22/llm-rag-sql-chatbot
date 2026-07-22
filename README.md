# LLM RAG + SQL Chatbot

A production-grade AI assistant that answers natural language questions about agricultural data by combining **RAG (Retrieval-Augmented Generation)** for documentation with a **Text-to-SQL Agent** for querying multiple PostgreSQL databases. Fully local, privacy-first, built with LangChain, ChromaDB, and Ollama.

## Architecture

```
User Question
     |
     v
Query Classifier (keyword-based)
     |
     +----> RAG Pipeline -----> ChromaDB Vectorstore
     |         |                       |
     |         v                       v
     |     Condense Q  -->  BM25 + MMR  -->  Ollama LLM  -->  Answer
     |
     +----> SQL Pipeline -----> PostgreSQL (3 DBs)
     |         |                       |
     |         v                       v
     |     Search Tables  -->  Generate SQL  -->  Execute  -->  Format
     |
     +----> BOTH (merge RAG + SQL)
     |
     +----> GREETING / CHITCHAT / META
```

## Key Features

### RAG Pipeline (Documentation Q&A)
- Document ingestion from Markdown, TXT, and PDF files
- Semantic chunking with Markdown header-aware splitting
- **Hybrid Retrieval**: BM25 (lexical) + MMR (semantic) via EnsembleRetriever
- Conversational memory with automatic question condensation
- Source attribution for transparency

### Text-to-SQL Agent (Database Queries)
- Queries across **3 PostgreSQL databases** with different schemas
- **Controlled pipeline** (primary): domain search → schema retrieval → SQL generation → execution → formatting
- **ReAct agent** (fallback): autonomous tool-calling agent with recursion recovery
- Vector-based **schema pruning** (max 4 tables to reduce LLM confusion)
- **View-first strategy**: pre-joined views prioritized over raw tables
- **FK graph** for automatic JOIN path discovery between tables
- SQL validation with `sqlglot` + schema-aware column verification
- **Few-shot retrieval**: cosine-similarity matching of similar question-SQL pairs
- 5-minute result caching for repeated queries

### Query Classification
- Instant keyword-based router classifies questions into:
  - `RAG` — documentation/procedure questions
  - `SQL` — data/metric queries
  - `BOTH` — requires documentation + data
  - `GREETING` / `CHITCHAT` — conversational
  - `META` — capabilities overview

### Streaming
- **Server-Sent Events (SSE)** for real-time token streaming from FastAPI
- Token-level delivery for responsive UX

### Observability
- Optional **Langfuse** integration for LLM tracing, scoring, and prompt management
- Automatic "no-info" scoring when the model can't answer
- User feedback collection (like/dislike) with trace-level scoring

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.12+, FastAPI, Uvicorn |
| **LLM Runtime** | Ollama (local GPU inference) |
| **LLM Model** | Qwen2.5 7B (RAG + SQL + Condense) |
| **Embeddings** | intfloat/multilingual-e5-small (HuggingFace) |
| **Vector Store** | ChromaDB (3 collections: docs, tables, domains) |
| **RAG Framework** | LangChain (ConversationalRetrievalChain) |
| **Agent Framework** | LangGraph (ReAct agent with tool calling) |
| **Retrieval** | BM25 (rank-bm25) + MMR hybrid, EnsembleRetriever |
| **Database** | PostgreSQL × 3 (psycopg2 + SQLAlchemy engines) |
| **Frontend** | Vanilla HTML/CSS/JS with SSE streaming |
| **Observability** | Langfuse (optional) |
| **Testing** | pytest, Locust (load testing) |
| **NLP** | spaCy, sentence-transformers |

## Project Structure

```
main.py                        FastAPI entry point with lifespan management
core/
  config.py                    Centralized configuration (models, DBs, chunking, prompts)
  vectorstore.py               ChromaDB vectorstore initialization and ingestion
agents/
  rag_agent.py                 RAG pipeline (ConversationalRetrievalChain + hybrid retrieval)
  sql_agent.py                 SQL Agent with controlled pipeline + ReAct fallback
api/
  chat.py                      /chat/stream SSE endpoint, /feedback, /admin/*
  router_classify.py           Keyword-based query classifier
  meta_handler.py              Module overview endpoint
scripts/
  ingest.py                    Document ingestion pipeline
  setup_views.sql              Database views and materialized views (3 DBs)
frontend/
  index.html                   Chat interface
  login.html                   Entity/year selection screen
  admin.html                   Document management admin panel
  script.js                    Chat logic (SSE streaming, sessions, feedback)
  admin.js                     Admin panel logic
documents/
  *.md                         RAG knowledge base (generic agricultural documentation)
  few_shot_sql.json            Synthetic question-SQL example pairs
  table_descriptions.json      Semantic table descriptions for schema retrieval
benchmarks/                    Evaluation scripts (golden set, RAG, vector store scaling)
tests/                         Unit tests (router classification)
```

## Setup

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com/) installed and running
- PostgreSQL with 3 databases (operations, plots, cellar)
- GPU with 8+ GB VRAM (tested with NVIDIA RTX 3070)

### Installation

```bash
git clone <repo-url>
cd llm-rag-sql-chatbot

python -m venv venv
source venv/bin/activate      # Linux/Mac
pip install -r requirements.txt

ollama pull qwen2.5:7b
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:
```env
DB_OPERATIONS_URL=postgresql://user:password@host:5432/operations
DB_PLOTS_URL=postgresql://user:password@host:5432/plots
DB_CELLAR_URL=postgresql://user:password@host:5432/cellar

# Langfuse (optional)
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=http://localhost:3000
```

### Database Setup

Run the view setup script before first launch:

```bash
# Execute each section connected to the respective database
psql -d operations -f scripts/setup_views.sql   # Section 1
psql -d plots -f scripts/setup_views.sql        # Section 2
psql -d cellar -f scripts/setup_views.sql       # Section 3
```

### Launch

```bash
uvicorn main:app --reload
```

Frontend available at `http://localhost:8000/frontend/index.html`.

On first run, vector stores are built automatically (requires internet for embedding model download).

## How It Works

### RAG Flow
1. User question received via SSE endpoint
2. Question condensed using conversation history (resolves pronouns, follow-ups)
3. Classified by keyword router
4. If RAG: documents retrieved with BM25+MMR hybrid → Ollama generates answer from context
5. Response streamed token-by-token via SSE

### SQL Flow
1. Question classified as SQL/BOTH
2. **Domain search**: vector similarity to find relevant table groups
3. **Schema retrieval**: schemas of top tables fetched (max 4 + views)
4. **Few-shot injection**: similar question-SQL pairs appended to context
5. **SQL generation**: Ollama generates query with schema + examples
6. **Validation**: `sqlglot` syntax check + column verification against in-memory schemas
7. **Execution**: query runs on the appropriate database with 10s timeout
8. **Auto-retry**: on error, LLM corrects the query (1 retry)
9. **Formatting**: results formatted as user-friendly markdown table

### Fallback (ReAct Agent)
If the controlled pipeline fails, a LangGraph ReAct agent with 5 tools takes over:
- `search_tables` — semantic table search
- `get_table_schema` — fetch column definitions
- `sample_table_data` — preview real column values
- `run_sql` — execute SELECT queries
- `find_join_path` — discover FK paths between tables

## Benchmarks

```bash
# Unit tests
python -m pytest tests/ -v

# RAG evaluation (retrieval accuracy)
python benchmarks/eval_rag.py

# SQL golden set evaluation
python benchmarks/eval_golden_set.py

# Vector store scalability
python benchmarks/eval_vectorstore.py

# Load testing (Locust)
python benchmarks/load_test.py
```

## Design Decisions

- **100% local processing**: No cloud APIs, full data privacy via Ollama
- **Multi-DB architecture**: Each domain (Monitoring, Operations, Cellar) has its own database
- **View-first SQL queries**: Pre-joined views reduce LLM errors vs complex JOINs
- **Schema pruning**: Limiting to 4 tables prevents LLM confusion with large schemas
- **Hybrid retrieval**: BM25 handles exact terminology while MMR covers semantic similarity
- **Controlled pipeline + ReAct fallback**: Two-tier approach balances reliability and flexibility
- **Portuguese language**: System designed for Portuguese-speaking users (PT-PT)

## Future Work

- **Multimodal support**: Voice interaction (Whisper STT + Coqui TTS)
- **User authentication**: JWT-based auth integrated with the main application
- **Automatic MV refresh**: pg_cron scheduled refresh of materialized views
- **Server GPU scaling**: Migration to higher VRAM GPUs for parallel inference and larger models
- **Continuous evaluation**: LLM-as-Judge automatic regression testing from production queries
- **Fine-tuning**: Supervised fine-tuning of Qwen2.5 7B on domain-specific question-SQL pairs via Unsloth
