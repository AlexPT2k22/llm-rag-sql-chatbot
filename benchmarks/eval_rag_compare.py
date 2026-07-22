"""
  python benchmarks/eval_rag_compare.py                  # corre as 4
  python benchmarks/eval_rag_compare.py --only baseline  # só uma
"""
import os
import sys
import time
import json
import argparse
import unicodedata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains import HypotheticalDocumentEmbedder
from core.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_NUM_CTX,
    SEARCH_K, SEARCH_FETCH_K, SEARCH_LAMBDA_MULT,
    QA_PROMPT_TEMPLATE, CHROMA_DIR,
)
from core.vectorstore import get_embeddings
from langchain_chroma import Chroma
from benchmarks.eval_rag import GOLDEN_SET, _norm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

QA_PROMPT = PromptTemplate(template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"])


def format_context(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_baseline(vs, embeddings):
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": SEARCH_K, "fetch_k": SEARCH_FETCH_K, "lambda_mult": SEARCH_LAMBDA_MULT},
    )


def build_hyde(vs, embeddings):
    """HyDE: o LLM gera um doc hipotético; embed desse doc usado para retrieval."""
    llm = OllamaLLM(model=LLM_MODEL, temperature=0, num_ctx=LLM_NUM_CTX)
    hyde_emb = HypotheticalDocumentEmbedder.from_llm(
        llm=llm,
        base_embeddings=embeddings,
        prompt_key="web_search",
    )
    vs_hyde = Chroma(persist_directory=CHROMA_DIR, embedding_function=hyde_emb)
    return vs_hyde.as_retriever(
        search_type="mmr",
        search_kwargs={"k": SEARCH_K, "fetch_k": SEARCH_FETCH_K, "lambda_mult": SEARCH_LAMBDA_MULT},
    )


def _build_hybrid(vs, w_bm25: float, w_mmr: float):
    raw = vs.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(raw["documents"], raw["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = SEARCH_K
    mmr = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": SEARCH_K, "fetch_k": SEARCH_FETCH_K, "lambda_mult": SEARCH_LAMBDA_MULT},
    )
    return EnsembleRetriever(retrievers=[bm25, mmr], weights=[w_bm25, w_mmr])


def build_hybrid(vs, embeddings):
    return _build_hybrid(vs, 0.5, 0.5)


def build_hybrid_30_70(vs, embeddings):
    return _build_hybrid(vs, 0.3, 0.7)


def build_hybrid_70_30(vs, embeddings):
    return _build_hybrid(vs, 0.7, 0.3)


def build_rerank(vs, embeddings):
    """MMR (fetch_k alto) + cross-encoder reranker."""
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    base = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": SEARCH_LAMBDA_MULT},
    )
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=SEARCH_K)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)


STRATEGIES = {
    "baseline":     build_baseline,
    "hyde":         build_hyde,
    "hybrid":       build_hybrid,        # 0.5 / 0.5
    "hybrid_30_70": build_hybrid_30_70,  # + peso MMR
    "hybrid_70_30": build_hybrid_70_30,  # + peso BM25
    "rerank":       build_rerank,
}


def run_strategy(name: str, retriever, llm) -> dict:
    print(f"\n=== {name.upper()} — {len(GOLDEN_SET)} perguntas ===")
    items = []
    answered = 0
    correct = 0
    no_info = 0
    times = []

    for i, item in enumerate(GOLDEN_SET, 1):
        q = item["q"]
        must = [_norm(k) for k in item["must"]]

        t0 = time.time()
        try:
            docs = retriever.invoke(q)
            context = format_context(docs)
            prompt = QA_PROMPT.format(context=context, question=q)
            raw = llm.invoke(prompt)
            elapsed = time.time() - t0
            times.append(elapsed)

            answer = _norm(raw)
            if "nao tenho informacao suficiente" in answer:
                status = "NO_INFO"
                no_info += 1
            else:
                answered += 1
                if all(k in answer for k in must):
                    correct += 1
                    status = "OK"
                else:
                    missing = [k for k in must if k not in answer]
                    status = f"MISSING:{','.join(missing)}"

            items.append({
                "i": i, "q": q, "status": status,
                "elapsed": round(elapsed, 2), "n_docs": len(docs),
            })
            print(f"[{i:02d}/{len(GOLDEN_SET)}] {status:25s} {elapsed:5.1f}s — {q[:55]}")
        except Exception as e:
            elapsed = time.time() - t0
            times.append(elapsed)
            items.append({"i": i, "q": q, "status": f"ERROR:{e}", "elapsed": round(elapsed, 2)})
            print(f"[{i:02d}/{len(GOLDEN_SET)}] ERROR: {e}")

    total = len(GOLDEN_SET)
    summary = {
        "strategy": name,
        "total": total,
        "answered": answered,
        "correct": correct,
        "no_info": no_info,
        "accuracy_pct": round(100 * correct / total, 1),
        "answered_pct": round(100 * answered / total, 1),
        "no_info_pct": round(100 * no_info / total, 1),
        "avg_time": round(sum(times) / len(times), 2) if times else 0,
        "max_time": round(max(times), 2) if times else 0,
        "items": items,
    }

    out = os.path.join(RESULTS_DIR, f"rag_compare_{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[{name}] correct={correct}/{total} ({summary['accuracy_pct']}%)  "
          f"no_info={no_info}  avg={summary['avg_time']}s  → {out}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(STRATEGIES.keys()),
                        help="Corre só uma estratégia")
    args = parser.parse_args()

    print("[setup] A carregar embeddings + vectorstore...")
    embeddings = get_embeddings()
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    print(f"[setup] Vectorstore com {vs._collection.count()} chunks")

    llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_ctx=LLM_NUM_CTX)

    targets = [args.only] if args.only else list(STRATEGIES.keys())
    summaries = []
    for name in targets:
        try:
            print(f"\n[setup] A construir retriever: {name}")
            retriever = STRATEGIES[name](vs, embeddings)
            summaries.append(run_strategy(name, retriever, llm))
        except Exception as e:
            import traceback
            print(f"[{name}] FALHOU no setup: {e}")
            traceback.print_exc()

    if len(summaries) > 1:
        agg = {s["strategy"]: {k: v for k, v in s.items() if k != "items"} for s in summaries}
        out = os.path.join(RESULTS_DIR, "rag_compare_summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        print(f"\n=== Resumo agregado → {out} ===")
        for s in summaries:
            print(f"  {s['strategy']:10s}  acc={s['accuracy_pct']:5.1f}%  "
                  f"no_info={s['no_info_pct']:5.1f}%  avg={s['avg_time']:5.1f}s")


if __name__ == "__main__":
    main()
