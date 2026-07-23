import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR   = os.path.join(BASE_DIR, "documents")
EMBEDDING_MODEL  = "intfloat/multilingual-e5-small"
EMBEDDING_DEVICE = "cpu"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
LLM_MODEL       = "qwen2.5:7b"
LLM_TEMPERATURE = 0.3
LLM_NUM_CTX     = 8192
CONDENSE_MODEL       = "qwen2.5:7b"
CONDENSE_TEMPERATURE = 0
CONDENSE_NUM_CTX     = 8192
SQL_MODEL       = "qwen2.5:7b"
SQL_TEMPERATURE = 0
SQL_NUM_CTX     = 8192
DATABASES = {
    "Operations": os.getenv("DB_OPERATIONS_URL"),
    "Plots":      os.getenv("DB_PLOTS_URL"),
    "Cellar":     os.getenv("DB_CELLAR_URL"),
}


def get_langfuse_handler():
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if pk and sk:
        try:
            from langfuse.langchain import CallbackHandler
            return CallbackHandler()
        except ImportError:
            pass
    return None


def get_langfuse_client():
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if pk and sk:
        try:
            from langfuse import Langfuse
            return Langfuse()
        except ImportError:
            pass
    return None


SEARCH_TYPE        = "mmr"
SEARCH_K           = 5
SEARCH_FETCH_K     = 20
SEARCH_LAMBDA_MULT = 0.7
USE_MULTI_QUERY    = os.getenv("USE_MULTI_QUERY", "0") == "0"
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
HYBRID_MMR_WEIGHT  = float(os.getenv("HYBRID_MMR_WEIGHT",  "0.5"))
MEMORY_WINDOW_K = 3
SESSION_MAX     = 500
SESSION_TTL     = 3600

CONDENSE_PROMPT_TEMPLATE = """Rewrite the follow-up question to be self-contained, replacing pronouns (he, she, it, they, them, this, that, where) with the exact subject from the conversation history.

RULES:
1. Return ONLY the rewritten question, no explanations or formatting.
2. Do NOT include content from the assistant's answers in the question.
3. Do NOT change the meaning or add new information.
4. If the question is already clear and self-contained (has its own subject and verb, doesn't depend on history), return it EXACTLY as is.
5. NEVER introduce topics, entities, or concepts that aren't in the user's original question. History is only for resolving pronouns — never to change the subject.
6. The rewritten question must be SHORT (maximum 1 sentence).

EXAMPLES:

History:
User: how many plots are there?
Assistant: There are 15 registered plots.
Question: and how many are vineyards?
Rewritten: and how many plots are vineyards?

History:
User: what is an irrigation type?
Assistant: An irrigation type defines the method of irrigating a crop.
Question: list my properties
Rewritten: list my properties

History:
User: how many pesticide applications were there in 2025?
Assistant: There were 42 applications in 2025.
Question: and which products were used?
Rewritten: which pesticides were used in the 2025 applications?

History:
{chat_history}

Follow-up question: {question}
Standalone question:"""

QA_PROMPT_TEMPLATE = """You are an AgriSystem support assistant for AgriTech.
Answer ONLY based on the provided context. Always respond in English.

IMPORTANT RULES:
- The prefixes [Module:] and [Section:] are INTERNAL DOCUMENT TITLES — they are NOT application menus. NEVER use them as navigation steps. NEVER cite file names or section paths in your answer.
- To indicate where to find a feature, use ONLY what is explicitly written after "Where to find in the application:" in the context. If that field doesn't exist, use the buttons/menus/icons described in the text.
- If the context does NOT contain enough information to answer the question, respond EXACTLY with: "I don't have enough information in the documentation to answer that question."
- NEVER invent steps, menus, buttons, or features that are not explicitly described in the context.

Context:
{context}

Question: {question}
Answer:"""
