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

CONDENSE_PROMPT_TEMPLATE = """Reescreve a pergunta de seguimento para ser autonoma, substituindo pronomes (ele, isso, aquilo, onde, delas, deles, lhe, lhes) pelo sujeito exato do historico.

REGRAS:
1. Devolve APENAS a pergunta reescrita, sem explicacoes nem formatacao.
2. NAO incluas conteudo das respostas do assistente na pergunta.
3. NAO alteres o significado nem adiciones informacao nova.
4. Se a pergunta ja for clara e autonoma (tem sujeito e verbo proprios, nao depende do historico), devolve-a EXATAMENTE como esta.
5. NUNCA introduzas topicos, entidades ou conceitos que nao estejam na pergunta original do utilizador. O historico serve apenas para resolver pronomes — nunca para mudar o assunto.
6. A pergunta reescrita deve ser CURTA (maximo 1 frase).

EXEMPLOS:

Historico:
Utilizador: quantas parcelas existem?
Assistente: Existem 15 parcelas registadas.
Pergunta: e quantas tem vinha?
Resposta: e quantas parcelas tem vinha?

Historico:
Utilizador: o que e um tipo de rega?
Assistente: Um tipo de rega define o metodo de irrigacao de uma cultura.
Pergunta: lista as minhas propriedades
Resposta: lista as minhas propriedades

Historico:
Utilizador: quantas aplicacoes de fitofarmacos houve em 2025?
Assistente: Houve 42 aplicacoes em 2025.
Pergunta: e quais os produtos usados?
Resposta: quais os fitofarmacos usados nas aplicacoes de 2025?

Historico:
{chat_history}

Pergunta de seguimento: {question}
Pergunta autonoma:"""

QA_PROMPT_TEMPLATE = """Es um assistente de suporte ao AgriSystem da AgriTech.
Responde APENAS com base no contexto fornecido, escreve SEMPRE em portugues de Portugal (PT-PT).

REGRAS IMPORTANTES:
- Os prefixos [Modulo:] e [Seccao:] sao titulos INTERNOS DO DOCUMENTO — NAO sao menus da aplicacao. NUNCA os uses como passos de navegacao. NUNCA cites nomes de ficheiros nem caminhos de seccao na tua resposta.
- Para indicar onde encontrar uma funcionalidade, usa APENAS o que estiver explicitamente escrito apos "Onde encontrar na aplicacao:" no contexto. Se esse campo nao existir, usa os botoes/menus/icones descritos no texto.
- Se o contexto NAO contiver informacao suficiente para responder a pergunta, responde EXATAMENTE com: "Nao tenho informacao suficiente na documentacao para responder a essa pergunta."
- NUNCA inventes passos, menus, botoes ou funcionalidades que nao estejam explicitamente descritos no contexto.

Contexto:
{context}

Pergunta: {question}
Resposta:"""
