from langchain_ollama import OllamaLLM
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from core.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_NUM_CTX,
    CONDENSE_MODEL, CONDENSE_TEMPERATURE, CONDENSE_NUM_CTX,
    SEARCH_TYPE, SEARCH_K, SEARCH_FETCH_K, SEARCH_LAMBDA_MULT,
    HYBRID_BM25_WEIGHT, HYBRID_MMR_WEIGHT,
    CONDENSE_PROMPT_TEMPLATE, QA_PROMPT_TEMPLATE,
)

CONDENSE_PROMPT = PromptTemplate(
    template=CONDENSE_PROMPT_TEMPLATE,
    input_variables=["chat_history", "question"],
)

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_rag_chain(vectorstore, callbacks=None):
    callbacks = callbacks or []
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
    )
    llm_condense = OllamaLLM(
        model=CONDENSE_MODEL,
        temperature=CONDENSE_TEMPERATURE,
        num_ctx=CONDENSE_NUM_CTX,
    )
    mmr = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "fetch_k": SEARCH_FETCH_K,
            "lambda_mult": SEARCH_LAMBDA_MULT,
        },
    )
    raw = vectorstore.get(include=["documents", "metadatas"])
    bm25_docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(raw["documents"], raw["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = SEARCH_K
    retriever = EnsembleRetriever(
        retrievers=[bm25, mmr],
        weights=[HYBRID_BM25_WEIGHT, HYBRID_MMR_WEIGHT],
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=llm_condense,
        retriever=retriever,
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
        callbacks=callbacks,
    )
    return chain, llm, llm_condense, retriever
