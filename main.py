import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import api.chat as chat_module
from api.chat import router as chat_router
from agents.rag_agent import build_rag_chain
from agents.sql_agent import SQLAgentTools
from core.vectorstore import get_embeddings, init_vectorstore
import core.config as cfg

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("A inicializar embeddings...")
    embeddings = get_embeddings()
    print("A carregar vectorstore...")
    vectorstore = init_vectorstore(embeddings)
    lf_handler = cfg.get_langfuse_handler() if hasattr(cfg, "get_langfuse_handler") else None
    callbacks = [lf_handler] if lf_handler else []
    print("A construir RAG chain...")
    chain, llm, llm_condense, retriever = build_rag_chain(vectorstore, callbacks=callbacks)
    print("A inicializar SQL Agent...")
    sql_agent = SQLAgentTools(embeddings, llm_model=cfg.SQL_MODEL)
    for db_name, db_url in cfg.DATABASES.items():
        if db_url:
            sql_agent.add_database(db_name, db_url)
    sql_agent.setup()
    chat_module.qa_chain     = chain
    chat_module.sql_agent    = sql_agent
    chat_module.llm          = llm
    chat_module.llm_condense = llm_condense
    chat_module.retriever    = retriever
    chat_module.lf_callbacks = callbacks
    chat_module.vectorstore  = vectorstore
    print("Pronto.")
    yield
    print("A encerrar servidor.")


app = FastAPI(title="AgriSystem AI Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
