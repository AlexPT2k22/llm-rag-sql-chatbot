# LLM RAG + SQL Chatbot (Sanitized Version From Internship, Some Features Haven't Been Added)

A production-grade AI assistant that answers natural language questions about agricultural data by combining **RAG (Retrieval-Augmented Generation)** for documentation with a **Text-to-SQL Agent** for querying multiple PostgreSQL databases. Fully local, privacy-first, built with LangChain, ChromaDB, and Ollama.

## Interesting Graphs

### 1 User VS 5 Users at the same time (Graph is in Portuguese)

<img width="3000" height="1800" alt="grafico_comparacao_1_vs_5" src="https://github.com/user-attachments/assets/95ceffbb-a178-44a0-bca7-bbb47fbc3552" />

### Latency under 5 concurrent users (Graph is in Portuguese)

<img width="3000" height="1500" alt="grafico_evolucao_5_users" src="https://github.com/user-attachments/assets/64841acb-d4de-4a78-afc3-79844550569b" />

### Accuracy by RAG Strategy (Graph is in Portuguese)

<img width="1200" height="750" alt="1_accuracy" src="https://github.com/user-attachments/assets/e2f7d1d1-50e5-4903-814d-6aafaee12290" />

### Latency by RAG Strategy (Graph is in Portuguese)

<img width="1200" height="750" alt="3_latency" src="https://github.com/user-attachments/assets/635fa31d-1bd9-4ad6-ad9b-73161c9f6bc4" />

### Latency VS Accuracy by RAG Strategy (Graph is in Portuguese)

<img width="1200" height="825" alt="5_pareto" src="https://github.com/user-attachments/assets/f0eb7ba4-33f6-4a3e-ad6b-09ffa72cc930" />

### Optimizations Evolution

<img width="3570" height="1815" alt="golden_set_evolution" src="https://github.com/user-attachments/assets/622d0913-3618-4cce-85d8-4e07c25be4c5" />

## Architecture

```plantuml
@startuml
skinparam backgroundColor #0d1117
skinparam defaultFontColor #c9d1d9
skinparam defaultFontSize 12
skinparam arrowColor #e94560
skinparam rectangleBackgroundColor #161b22
skinparam rectangleBorderColor #30363d
skinparam rectangleFontColor #c9d1d9
skinparam databaseBackgroundColor #161b22
skinparam databaseBorderColor #e94560
skinparam databaseFontColor #c9d1d9
skinparam cloudBackgroundColor #161b22
skinparam cloudBorderColor #e94560
skinparam cloudFontColor #c9d1d9
skinparam noteBackgroundColor #161b22
skinparam noteBorderColor #30363d
skinparam noteFontColor #c9d1d9
skinparam actorBackgroundColor #161b22
skinparam actorBorderColor #e94560
skinparam actorFontColor #c9d1d9

title LLM RAG + SQL Chatbot Architecture

rectangle "User Question" as User #161b22

rectangle "Condense Question\n(resolve pronouns, follow-ups)" as Condense #161b22

rectangle "Query Classifier\n(keyword-based)" as Classifier #161b22

rectangle "RAG Pipeline" as RAG #161b22
rectangle "SQL Pipeline" as SQL #161b22
rectangle "Merge RAG + SQL" as BOTH #161b22
rectangle "Static Response" as CHAT #161b22
rectangle "Modules Overview" as META #161b22

' RAG Pipeline
rectangle "Hybrid Retriever\nBM25 + MMR" as Retriever #161b22
database "ChromaDB\n3 Collections" as ChromaDB #161b22
rectangle "Ollama Qwen2.5 7B" as LLM_RAG #161b22
rectangle "SSE Token Stream" as Stream_RAG #161b22

' SQL Pipeline
package "SQL Pipeline (Controlled)" as SQL_PIPELINE #1a1a2e;line:e94560 {
  rectangle "Domain Search\nBM25 + MMR Hybrid" as DomainSearch #161b22
  rectangle "Schema Pruning\n(max 4 tables)" as SchemaPruning #161b22
  rectangle "View Injection\n(v_cellar_*, mv_plot_*)" as ViewInjection #161b22
  rectangle "Few-shot Retrieval\ncosine similarity" as FewShot #161b22
  rectangle "SQL Generation\nOllama Qwen2.5 7B" as GenSQL #161b22
  rectangle "Validate & Fix\nsqlglot + column check" as Validate #161b22
  rectangle "Execute SQL\nPostgreSQL (10s timeout)" as Execute #161b22
  rectangle "Error?" as Retry #161b22
  rectangle "LLM Auto-correct" as FixLLM #161b22
  rectangle "Format as Markdown" as Format #161b22
}

rectangle "SSE Token Stream" as Stream_SQL #161b22

' Fallback
package "ReAct Agent (Fallback)" as FALLBACK #1a1a2e;line:e94560 {
  rectangle "LangGraph Agent" as Agent #161b22
  rectangle "search_tables" as Tool1 #161b22
  rectangle "get_table_schema" as Tool2 #161b22
  rectangle "sample_table_data" as Tool3 #161b22
  rectangle "run_sql" as Tool4 #161b22
  rectangle "find_join_path" as Tool5 #161b22
}

rectangle "Fallback Triggered" as Fallback #161b22
rectangle "SSE Token Stream" as Stream_FB #161b22

' Merge
rectangle "Merge Responses\n(LLM or Concatenate)" as Merge #161b22
rectangle "SSE Token Stream" as Stream_Merge #161b22

' Flow
User --> Condense
Condense --> Classifier

Classifier --> RAG : RAG
Classifier --> SQL : SQL
Classifier --> BOTH : BOTH
Classifier --> CHAT : GREETING/CHITCHAT
Classifier --> META : META

RAG --> Retriever
Retriever --> ChromaDB
ChromaDB --> LLM_RAG
LLM_RAG --> Stream_RAG

SQL --> SQL_PIPELINE
DomainSearch --> SchemaPruning
SchemaPruning --> ViewInjection
ViewInjection --> FewShot
FewShot --> GenSQL
GenSQL --> Validate
Validate --> Execute
Execute --> Retry
Retry --> FixLLM : Yes (1 retry)
FixLLM --> Execute
Retry --> Format : No
SQL_PIPELINE --> Stream_SQL
SQL_PIPELINE --> Fallback : Failed

Fallback --> FALLBACK
Agent --> Tool1
Agent --> Tool2
Agent --> Tool3
Agent --> Tool4
Agent --> Tool5
FALLBACK --> Stream_FB

BOTH --> RAG
BOTH --> SQL
RAG --> Merge
SQL --> Merge
Merge --> Stream_Merge

@enduml
```

### Pipeline Statistics (Live)

The system tracks pipeline performance in real-time:

| Metric | Description |
|---|---|
| `pipeline_ok` | Questions answered by controlled pipeline |
| `pipeline_fail` | Pipeline failures (triggers fallback) |
| `fallback_ok` | ReAct agent recovered the answer |
| `fallback_fail` | Both pipeline and fallback failed |
| `cache_hit` | Repeated questions served from cache (5min TTL) |

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

Create `.env`:
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

### System Scale
- **3 PostgreSQL databases**: Operations, Plots, Cellar
- **500+ tables** across all databases
- **50,000+ rows** of agricultural, operational, and cellar data
- **50+ pre-joined views** for optimized querying
- **3 ChromaDB collections**: documents, tables, domains

### RAG Flow
1. User question received via SSE endpoint
2. Question condensed using conversation history (resolves pronouns, follow-ups)
3. Classified by keyword router
4. If RAG: documents retrieved with BM25+MMR hybrid → Ollama generates answer from context
5. Response streamed token-by-token via SSE

### SQL Flow (Controlled Pipeline)
1. Question classified as SQL/BOTH
2. **Domain search**: BM25+MMR hybrid finds relevant table groups (70/30 lexical/semantic)
3. **Schema pruning**: max 4 regular tables + views injected (prevents LLM context overflow)
4. **View injection**: domain-specific views prioritized (v_cellar_*, mv_plot_resource_full)
5. **Few-shot retrieval**: cosine-similarity matches similar question-SQL pairs
6. **SQL generation**: Ollama generates query with schema + context + examples
7. **Validation**: sqlglot syntax check + column verification against in-memory schemas
8. **Execution**: query runs on appropriate database with 10s timeout
9. **Auto-retry**: on error, LLM corrects the query (1 retry with schema context)
10. **Formatting**: results formatted as user-friendly markdown table

### Fallback (ReAct Agent)
If the controlled pipeline fails (schema pruning miss, complex joins, etc.), a LangGraph ReAct agent with 5 tools takes over:
- `search_tables` — semantic table search across 3 DBs
- `get_table_schema` — fetch column definitions with FK hints
- `sample_table_data` — preview real column values (M/F, date formats, etc.)
- `run_sql` — execute SELECT queries with 10s timeout
- `find_join_path` — discover FK paths via NetworkX graph traversal


## Design Decisions

- **100% local processing**: No cloud APIs, full data privacy via Ollama
- **Multi-DB architecture**: Each domain (Operations, Plots, Cellar) has its own database with 500+ tables
- **Controlled pipeline + ReAct fallback**: Two-tier approach balances reliability and flexibility
- **Schema pruning**: Limiting to 4 tables prevents LLM confusion with large schemas
- **View-first SQL queries**: Pre-joined views reduce LLM errors vs complex JOINs
- **Hybrid retrieval**: BM25 handles exact terminology while MMR covers semantic similarity
- **FK graph traversal**: NetworkX-based JOIN path discovery between tables without direct foreign keys
- **Auto-retry with LLM correction**: Schema-aware error fixing on first SQL failure
- **SQL validation**: sqlglot syntax check + column verification before execution
- **Result caching**: 5-minute TTL cache for repeated queries reduces load

## Future Work

- **Multimodal support**: Voice interaction (Whisper STT + Coqui TTS)
- **User authentication**: JWT-based auth integrated with the main application
- **Automatic MV refresh**: pg_cron scheduled refresh of materialized views
- **Server GPU scaling**: Migration to higher VRAM GPUs for parallel inference and larger models
- **Continuous evaluation**: LLM-as-Judge automatic regression testing from production queries
- **Fine-tuning**: Supervised fine-tuning of Qwen2.5 7B on domain-specific question-SQL pairs via Unsloth





