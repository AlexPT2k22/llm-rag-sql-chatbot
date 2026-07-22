import asyncio
import glob as glob_module
import json
import os
import time
import uuid
from fastapi import APIRouter, Form, Query, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cachetools import TTLCache
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from api.router_classify import (
    classify,
    GREETING_RESPONSE,
    CHITCHAT_RESPONSE,

)
from langchain_core.messages import AIMessage, HumanMessage
from api.meta_handler import get_modules_overview
from sqlalchemy import create_engine, text as sql_text
from core.config import (MEMORY_WINDOW_K, SESSION_MAX, SESSION_TTL, get_langfuse_handler,
                         get_langfuse_client, DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
                         SEARCH_TYPE, SEARCH_K, SEARCH_FETCH_K, SEARCH_LAMBDA_MULT,
                         HYBRID_BM25_WEIGHT, HYBRID_MMR_WEIGHT, DATABASES)
from core.vectorstore import HEADER_KEYS
DEBUG_SQL = os.getenv("DEBUG_SQL", "0") == "1"
from agents.rag_agent import CONDENSE_PROMPT, QA_PROMPT
router = APIRouter()
qa_chain         = None
sql_agent        = None
llm              = None
llm_condense     = None
retriever        = None
lf_callbacks     = []
vectorstore      = None 
sessions: TTLCache = TTLCache(maxsize=SESSION_MAX, ttl=SESSION_TTL)

class SourceDoc(BaseModel):
    content: str
    source:  str | None = None
    module:  str | None = None

class ChatResponse(BaseModel):
    answer:    str
    category:  str
    trace_id:  str | None = None
    num_docs:  int | None = None
    sql_query: str | None = None
    sources:   list[SourceDoc] | None = None
    elapsed:   float

def _lf():
    handler = get_langfuse_handler()
    if handler:
        return [handler], handler
    return lf_callbacks, None

def _trace_id(lf_handler):
    return getattr(lf_handler, "last_trace_id", None) if lf_handler else None

_NO_INFO_PHRASES = [
    "não tenho informação suficiente",
    "não foram encontrados registos",
    "não encontrei informação",
    "não tenho dados",
    "sem dados",
]

def _auto_score(tid, resposta):
    """Send automatic 'no-info' score when the LLM can't answer."""
    if not tid:
        return
    lower = resposta.lower()
    if any(phrase in lower for phrase in _NO_INFO_PHRASES):
        lf = get_langfuse_client()
        if lf:
            lf.create_score(
                trace_id=tid,
                name="auto-no-info",
                value=0.0,
                data_type="NUMERIC",
                comment=resposta[:200],
            )

def _compress_history(messages):
    """
    Comprime o histórico passado ao agente SQL:
    - AIMessages com tabelas → resumo curto (evita "preguiça cognitiva")
    - HumanMessages do histórico → substituídas por placeholder genérico
      para evitar que nomes de entidades de perguntas anteriores contaminem
      a resposta actual. A pergunta actual é sempre injectada separadamente.
    """
    compressed = []
    for msg in messages:
        if msg.type == "ai" and "|" in msg.content and "---" in msg.content:
            row_count = len(msg.content.split("\n")) - 2
            resumo = f"[Forneci os dados pedidos numa tabela com {max(0, row_count)} linhas. Não tenho estes dados em memória. Para novos filtros, SOU OBRIGADO a executar nova query SQL.]"
            compressed.append(AIMessage(content=resumo))
        elif msg.type == "human":
            compressed.append(HumanMessage(content="[Pergunta anterior — ver resposta acima]"))
        else:
            compressed.append(msg)
    return compressed

def _get_session(session_id: str) -> ConversationBufferWindowMemory:
    mem = sessions.get(session_id)
    if mem is None:
        mem = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=MEMORY_WINDOW_K,
        )
        sessions[session_id] = mem
    return mem

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

_STOPWORDS_PT = {
    "a", "ao", "aos", "as", "à", "às", "com", "da", "das", "de", "do", "dos",
    "e", "é", "em", "esta", "este", "isto", "mas", "na", "nas", "nem", "no",
    "nos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo",
    "pelos", "por", "que", "se", "sem", "sua", "suas", "um", "uma", "umas",
    "uns", "tem", "têm", "sao", "são", "foi", "era", "sao", "há",
}

def _content_words(text: str) -> set[str]:
    return {
        w for w in text.lower().split()
        if len(w) > 3 and w not in _STOPWORDS_PT
    }

def _jaccard_similarity(original: str, condensed: str) -> float:
    orig_words = _content_words(original)
    cond_words = _content_words(condensed)
    if not orig_words:
        return 1.0
    return len(orig_words & cond_words) / len(orig_words | cond_words)


def _condense_question(pergunta: str, memory: ConversationBufferWindowMemory) -> str:
    history = memory.chat_memory.messages
    if not history:
        return pergunta
    chat_history_text = "\n".join(
        f"{'Utilizador' if i % 2 == 0 else 'Assistente'}: {m.content}"
        for i, m in enumerate(history)
    )
    prompt = CONDENSE_PROMPT.format(chat_history=chat_history_text, question=pergunta)
    condensed = llm_condense.invoke(prompt).strip()
    if len(condensed) > len(pergunta) * 3:
        return pergunta
    if _jaccard_similarity(pergunta, condensed) < 0.20:
        return pergunta
    return condensed

def _retrieve_and_build_prompt(pergunta_standalone: str) -> tuple[str, int]:
    docs = retriever.invoke(pergunta_standalone)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt_text = QA_PROMPT.format(context=context, question=pergunta_standalone)
    return prompt_text, len(docs)

def _build_sources(src_docs) -> list[SourceDoc]:
    return [
        SourceDoc(
            content=doc.page_content[:200],
            source=os.path.basename(doc.metadata.get("source", "")),
            module=doc.metadata.get("module"),
        )
        for doc in src_docs
    ]

def _router_query(pergunta: str, memory: ConversationBufferWindowMemory) -> str:
    """Injecta contexto da pergunta anterior se a actual for muito curta."""
    palavras = pergunta.strip().split()
    if len(palavras) <= 3 and memory.chat_memory.messages:
        for msg in reversed(memory.chat_memory.messages):
            if msg.type == "human":
                return f"{msg.content} {pergunta}"
    return pergunta

import datetime as _dt

@router.get("/api/entidades")
async def list_entidades():
    """Lista entidades disponíveis com contagem de propriedades."""
    db_url = DATABASES.get("Operations")
    if not db_url:
        raise HTTPException(status_code=500, detail="BD Operations não configurada.")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(sql_text("""
            SELECT e.gid, COUNT(p.gid) as num_propriedades
            FROM entidade e
            LEFT JOIN propriedade p ON p.id_entidade = e.gid
            GROUP BY e.gid
            ORDER BY e.gid
        """))
        entidades = [{"id": row[0], "num_propriedades": row[1]} for row in rows]
    engine.dispose()
    return {"entidades": entidades}


@router.get("/api/anos-agricolas")
async def list_anos_agricolas(entidade_id: int = Query(...)):
    """Lista anos agrícolas de uma entidade."""
    db_url = DATABASES.get("Plots")
    if not db_url:
        raise HTTPException(status_code=500, detail="BD Plots não configurada.")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(sql_text("""
            SELECT gid, denominacao, data_inicio, data_fim, estado
            FROM ano_agricola
            WHERE id_entidade = :eid AND estado = 'A'
            ORDER BY data_inicio DESC
        """), {"eid": entidade_id})
        hoje = _dt.date.today()
        anos = []
        for row in rows:
            corrente = row[2] <= hoje and (row[3] is None or row[3] >= hoje)
            anos.append({
                "id": row[0],
                "denominacao": row[1],
                "data_inicio": str(row[2]),
                "data_fim": str(row[3]) if row[3] else None,
                "corrente": corrente,
            })
    engine.dispose()
    return {"anos": anos}


@router.post("/chat", response_model=ChatResponse)
async def chat(message: str = Form(...), session_id: str = Form("default")):
    pergunta = message.strip()
    start    = time.time()
    categoria = classify(pergunta)
    if categoria == "GREETING":
        return ChatResponse(answer=GREETING_RESPONSE, category="GREETING", elapsed=time.time() - start)
    if categoria == "CHITCHAT":
        return ChatResponse(answer=CHITCHAT_RESPONSE, category="CHITCHAT", elapsed=time.time() - start)
    if categoria == "META":
        return ChatResponse(answer=get_modules_overview(), category="META", elapsed=time.time() - start)
    memory = _get_session(session_id)
    pergunta_standalone = await asyncio.to_thread(_condense_question, pergunta, memory)
    categoria = classify(pergunta_standalone)
    if categoria == "SQL":
        compressed_history = _compress_history(memory.chat_memory.messages)
        resposta, sql, _ = sql_agent.answer(pergunta_standalone, chat_history=compressed_history)
        memory.chat_memory.add_user_message(pergunta)
        memory.chat_memory.add_ai_message(resposta)
        return ChatResponse(answer=resposta, category="SQL", sql_query=sql if DEBUG_SQL else None, elapsed=time.time() - start)
    callbacks, lf_handler = _lf()
    if categoria == "BOTH":
        qa_chain.memory = memory
        rag_result = qa_chain.invoke({"question": pergunta}, config={"callbacks": callbacks})
        resposta_rag = rag_result["answer"]
        src_docs = rag_result.get("source_documents", [])
        compressed_history = _compress_history(memory.chat_memory.messages)
        resposta_sql, sql, _ = sql_agent.answer(pergunta, chat_history=compressed_history)
        merge_prompt = f"""Combina as seguintes duas respostas numa resposta única e coerente em português de Portugal.
Se uma das respostas disser 'Não tenho essa informação', usa apenas a outra.

Resposta da documentação:
{resposta_rag}

Dados da base de dados:
{resposta_sql}

Pergunta original: {pergunta}

Resposta combinada:"""
        try:
            resposta = llm.invoke(merge_prompt, config={"callbacks": callbacks})
        except Exception:
            resposta = f"{resposta_rag}\n\nDados da base de dados:\n{resposta_sql}"
        memory.chat_memory.add_user_message(pergunta)
        memory.chat_memory.add_ai_message(resposta)
        tid = _trace_id(lf_handler)
        return ChatResponse(
            answer=resposta, category="BOTH", trace_id=tid, sql_query=sql if DEBUG_SQL else None,
            num_docs=len(src_docs), sources=_build_sources(src_docs),
            elapsed=time.time() - start,
        )
    qa_chain.memory = memory
    result = qa_chain.invoke({"question": pergunta}, config={"callbacks": callbacks})
    resposta = result["answer"]
    src_docs = result.get("source_documents", [])
    memory.chat_memory.add_user_message(pergunta)
    memory.chat_memory.add_ai_message(resposta)
    tid = _trace_id(lf_handler)
    return ChatResponse(
        answer=resposta, category="RAG", trace_id=tid,
        num_docs=len(src_docs), sources=_build_sources(src_docs),
        elapsed=time.time() - start,
    )


@router.get("/chat/stream")
async def chat_stream(message: str = Query(...), session_id: str = Query("default"), entidade_id: int | None = Query(None), ano_agricola_id: int | None = Query(None)):
    pergunta  = message.strip()
    start     = time.time()
    categoria = classify(pergunta)
    if categoria in ("GREETING", "CHITCHAT", "META"):
        if categoria == "GREETING":
            resposta_fixa = GREETING_RESPONSE
        elif categoria == "CHITCHAT":
            resposta_fixa = CHITCHAT_RESPONSE
        else:
            resposta_fixa = get_modules_overview()
        async def fixed_gen():
            yield _sse({"type": "category", "category": categoria})
            yield _sse({"type": "token", "token": resposta_fixa})
            yield _sse({"type": "done", "elapsed": time.time() - start})
        return StreamingResponse(fixed_gen(), media_type="text/event-stream")
    memory = _get_session(session_id)
    callbacks, lf_handler = _lf()
    async def stream_gen():
        pergunta_standalone = await asyncio.to_thread(_condense_question, pergunta, memory)
        categoria = classify(pergunta_standalone)

        print(f"\n--- CONDENSE ---")
        print(f"Original:  {pergunta}")
        print(f"Condensed: {pergunta_standalone}")
        print(f"Categoria: {categoria}")
        print(f"----------------\n")

        yield _sse({"type": "category", "category": categoria})

        if categoria == "SQL":
            compressed_history = _compress_history(memory.chat_memory.messages)
            resposta, sql, _ = await asyncio.to_thread(
                sql_agent.answer, pergunta_standalone, compressed_history, entidade_id, ano_agricola_id
            )
            if sql:
                yield _sse({"type": "sql", "query": sql})
            
            yield _sse({"type": "token", "token": resposta})
            memory.chat_memory.add_user_message(pergunta)
            memory.chat_memory.add_ai_message(resposta)
            
            tid = getattr(sql_agent, "_langfuse_handler", None)
            tid = getattr(tid, "last_trace_id", None) if tid else None
            _auto_score(tid, resposta)
            yield _sse({"type": "done", "elapsed": time.time() - start, "trace_id": tid})
            return

        if categoria == "BOTH":
            compressed_history = _compress_history(memory.chat_memory.messages)
            resposta_sql, sql, _ = await asyncio.to_thread(
                sql_agent.answer, pergunta_standalone, compressed_history, entidade_id, ano_agricola_id
            )
            if sql:
                yield _sse({"type": "sql", "query": sql})
                
            prompt_text, num_docs = await asyncio.to_thread(_retrieve_and_build_prompt, pergunta_standalone)
            resposta_rag = await asyncio.to_thread(llm.invoke, prompt_text, config={"callbacks": callbacks})
            
            merge_prompt = f"""Combina as seguintes duas respostas numa resposta única e coerente em português de Portugal.
Se uma das respostas disser 'Não tenho essa informação', usa apenas a outra.

Resposta da documentação:
{resposta_rag}

Dados da base de dados:
{resposta_sql}

Pergunta original: {pergunta_standalone}

Resposta combinada:"""
            
            full_response = ""
            for chunk in llm.stream(merge_prompt, config={"callbacks": callbacks}):
                full_response += chunk
                yield _sse({"type": "token", "token": chunk})
                await asyncio.sleep(0)
                
            memory.chat_memory.add_user_message(pergunta)
            memory.chat_memory.add_ai_message(full_response)
            tid = _trace_id(lf_handler)
            _auto_score(tid, full_response)
            yield _sse({"type": "done", "elapsed": time.time() - start, "num_docs": num_docs, "trace_id": tid})
            return

        prompt_text, num_docs = await asyncio.to_thread(_retrieve_and_build_prompt, pergunta_standalone)
        full_response = ""
        
        for chunk in llm.stream(prompt_text, config={"callbacks": callbacks}):
            full_response += chunk
            yield _sse({"type": "token", "token": chunk})
            await asyncio.sleep(0)
            
        memory.chat_memory.add_user_message(pergunta)
        memory.chat_memory.add_ai_message(full_response)
        tid = _trace_id(lf_handler)
        _auto_score(tid, full_response)
        yield _sse({"type": "done", "elapsed": time.time() - start, "num_docs": num_docs, "trace_id": tid})
    return StreamingResponse(stream_gen(), media_type="text/event-stream")


@router.post("/feedback")
async def feedback(trace_id: str = Form(...), score: int = Form(...), comment: str = Form("")):
    lf = get_langfuse_client()
    if not lf:
        return {"status": "langfuse_disabled"}
    lf.create_score(
        trace_id=trace_id,
        name="user-feedback",
        value=float(score),
        data_type="NUMERIC",
        comment=comment if comment else ("like" if score == 1 else "dislike"),
    )
    return {"status": "ok"}


def _process_file_for_index(filepath: str) -> list:
    ext = os.path.splitext(filepath)[1].lower()
    module_name = (
        os.path.basename(filepath)
        .rsplit(".", 1)[0]
        .replace("_notes", "").replace("_notas", "").replace("_", " ")
        .title()
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    if ext == ".md":
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        loader = TextLoader(filepath, encoding="utf-8")
        raw_docs = loader.load()
        documents = []
        for raw_doc in raw_docs:
            splits = md_splitter.split_text(raw_doc.page_content)
            for split in splits:
                split.metadata["source"] = filepath
                split.metadata["module"] = module_name
                documents.append(split)
        chunks = text_splitter.split_documents(documents)
    elif ext == ".txt":
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["module"] = module_name
        chunks = text_splitter.split_documents(docs)
    elif ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata["module"] = module_name
        chunks = text_splitter.split_documents(docs)
    else:
        return []
    fname = os.path.basename(filepath)
    for chunk in chunks:
        chunk.metadata["filename"] = fname
        module = chunk.metadata.get("module", "Desconhecido")
        header_context = " > ".join(
            [chunk.metadata[k] for k in HEADER_KEYS if k in chunk.metadata]
        )
        prefix = f"[Módulo: {module}]"
        if header_context:
            prefix += f" [Secção: {header_context}]"
        chunk.page_content = f"{prefix}\n{chunk.page_content}"
    return chunks


def _rebuild_retriever():
    """Reconstrói BM25 + MMR após upload/deleção para que novos chunks sejam encontrados."""
    if vectorstore is None or qa_chain is None:
        return
    raw = vectorstore.get(include=["documents", "metadatas"])
    bm25_docs = [Document(page_content=t, metadata=m or {})
                 for t, m in zip(raw["documents"], raw["metadatas"])]
    new_bm25 = BM25Retriever.from_documents(bm25_docs)
    new_bm25.k = SEARCH_K
    new_mmr = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": SEARCH_K, "fetch_k": SEARCH_FETCH_K, "lambda_mult": SEARCH_LAMBDA_MULT},
    )
    qa_chain.retriever = EnsembleRetriever(
        retrievers=[new_bm25, new_mmr],
        weights=[HYBRID_BM25_WEIGHT, HYBRID_MMR_WEIGHT],
    )


@router.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...)):
    safe_name = os.path.basename(file.filename or "")
    if not safe_name or "/" in safe_name or "\\" in safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Nome de ficheiro inválido.")
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in [".md", ".txt", ".pdf"]:
        raise HTTPException(status_code=400, detail="Formato não suportado. Use .md, .txt ou .pdf.")
    dest = os.path.join(DOCS_DIR, safe_name)
    if not os.path.abspath(dest).startswith(os.path.abspath(DOCS_DIR)):
        raise HTTPException(status_code=400, detail="Acesso negado.")
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    chunks = _process_file_for_index(dest)
    before = vectorstore._collection.count() if vectorstore is not None else 0
    if chunks and vectorstore is not None:
        vectorstore.add_documents(chunks)
    after = vectorstore._collection.count() if vectorstore is not None else 0
    asyncio.get_event_loop().run_in_executor(None, _rebuild_retriever)
    return {"filename": file.filename, "chunks_indexed": len(chunks), "total_in_index": after, "added_to_index": after - before}


@router.get("/admin/documents")
async def admin_documents():
    files = []
    for f in glob_module.glob(os.path.join(DOCS_DIR, "*")):
        name = os.path.basename(f)
        if not name.startswith("_"):
            ext = os.path.splitext(name)[1].lower()
            if ext in [".md", ".txt", ".pdf"]:
                files.append({"name": name, "size_kb": round(os.path.getsize(f) / 1024, 1)})
    return {"documents": sorted(files, key=lambda x: x["name"])}


@router.get("/admin/debug/search")
async def admin_debug_search(q: str = Query(...), filename: str = Query(None)):
    total = vectorstore._collection.count()
    by_file = []
    if filename:
        r = vectorstore._collection.get(where={"filename": filename}, include=["metadatas", "documents"])
        by_file = [{"content": r["documents"][i][:200], "metadata": r["metadatas"][i]} for i in range(len(r["ids"]))]
    docs = vectorstore.similarity_search(q, k=3)
    return {
        "total_in_index": total,
        "chunks_for_file": len(by_file),
        "semantic_results": [{"content": d.page_content[:300], "metadata": d.metadata} for d in docs],
        "file_chunks_sample": by_file[:2],
    }


@router.delete("/admin/documents/{filename}")
async def admin_delete(filename: str):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Nome de ficheiro inválido.")
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.abspath(filepath).startswith(os.path.abspath(DOCS_DIR)):
        raise HTTPException(status_code=400, detail="Acesso negado.")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Ficheiro não encontrado.")
    chunks_removed = 0
    try:
        results = vectorstore._collection.get(where={"filename": filename})
        ids = results.get("ids", [])
        if ids:
            vectorstore._collection.delete(ids=ids)
            chunks_removed = len(ids)
    except Exception:
        pass
    os.remove(filepath)
    asyncio.get_event_loop().run_in_executor(None, _rebuild_retriever)
    return {"deleted": filename, "chunks_removed": chunks_removed}
