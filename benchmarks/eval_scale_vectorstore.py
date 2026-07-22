"""
Teste de escalabilidade do vectorstore
Mede o tempo de recuperação semântica à medida que
o número de chunks indexados cresce.
Tamanhos testados: 200, 500, 1000, 1557 (real), 3000, 6000, 12000
Para tamanhos > corpus real, são gerados chunks sintéticos com estrutura idêntica.

Execução:
    cd entregaEstagio
    python -m benchmarks.eval_scale_vectorstore

Resultados guardados em: benchmarks/results/scale_vectorstore.json
Gráfico guardado em:     benchmarks/results/scale_vectorstore.png
"""

import json
import os
import random
import shutil
import string
import sys
import tempfile
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.vectorstore import get_embeddings, HEADER_KEYS
import core.config as cfg

#Configuração
SIZES_TO_TEST = [200, 500, 1000, 1557, 3000, 6000, 12000]
QUERIES = [
    "Como registo uma prática agrícola?",
    "Quais são os tipos de solo disponíveis?",
    "O que é a monitorização de pragas?",
    "Como faço a gestão de fitofármacos?",
    "O que são os índices de vegetação NDVI?",
]
REPEATS = 5
K_RESULTS = 5
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "scale_vectorstore.json")
OUTPUT_PNG  = os.path.join(RESULTS_DIR, "scale_vectorstore.png")

MODULES = ["Field Monitoring", "Field Ops Agrícolas", "Cellar", "Sensores", "AgriSystem"]
SECTIONS = ["Introdução", "Procedimentos", "Configuração", "Relatórios", "Consultas"]


def _random_word(n=8):
    return "".join(random.choices(string.ascii_lowercase, k=n))


def generate_synthetic_chunk(idx: int) -> Document:
    module  = random.choice(MODULES)
    section = random.choice(SECTIONS)
    words   = " ".join(_random_word(random.randint(3, 10)) for _ in range(60))
    content = f"[Módulo: {module}] [Secção: {section}]\n{words}"
    return Document(
        page_content=content,
        metadata={"module": module, "filename": f"synthetic_{idx}.txt"},
    )


def load_real_chunks(embeddings) -> list[Document]:
    """Carrega os chunks reais do vectorstore de produção."""
    print("A carregar chunks reais do vectorstore...")
    vs = Chroma(persist_directory=cfg.CHROMA_DIR, embedding_function=embeddings)
    raw = vs.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(raw["documents"], raw["metadatas"])
    ]
    print(f"  {len(docs)} chunks reais carregados.")
    return docs


def build_corpus(real_chunks: list[Document], target_size: int) -> list[Document]:
    """Constrói um corpus do tamanho pretendido (completa com sintéticos se necessário)."""
    if target_size <= len(real_chunks):
        return random.sample(real_chunks, target_size)
    synthetic_needed = target_size - len(real_chunks)
    synthetic = [generate_synthetic_chunk(i) for i in range(synthetic_needed)]
    return real_chunks + synthetic


def measure_retrieval(vs: Chroma, queries: list[str], k: int, repeats: int) -> dict:
    times = []
    for query in queries:
        for _ in range(repeats):
            t0 = time.perf_counter()
            vs.similarity_search(query, k=k)
            times.append(time.perf_counter() - t0)
    return {
        "avg_s":  round(sum(times) / len(times), 4),
        "min_s":  round(min(times), 4),
        "max_s":  round(max(times), 4),
        "p95_s":  round(sorted(times)[int(len(times) * 0.95)], 4),
        "n_measurements": len(times),
    }


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("A inicializar embeddings (pode demorar)...")
    embeddings = get_embeddings()

    real_chunks = load_real_chunks(embeddings)
    results = []

    for size in SIZES_TO_TEST:
        print(f"\n[{size} chunks] A construir corpus...")
        corpus = build_corpus(real_chunks, size)

        tmp_dir = tempfile.mkdtemp(prefix="chroma_scale_")
        try:
            print(f"[{size} chunks] A indexar no vectorstore temporário...")
            t_index_start = time.perf_counter()
            vs = Chroma.from_documents(
                documents=corpus,
                embedding=embeddings,
                persist_directory=tmp_dir,
            )
            index_time = round(time.perf_counter() - t_index_start, 2)
            print(f"[{size} chunks] Indexação: {index_time}s. A medir retrieval...")

            metrics = measure_retrieval(vs, QUERIES, K_RESULTS, REPEATS)
            entry = {
                "size":       size,
                "real":       min(size, len(real_chunks)),
                "synthetic":  max(0, size - len(real_chunks)),
                "index_time_s": index_time,
                **metrics,
            }
            results.append(entry)
            print(f"[{size} chunks] avg={metrics['avg_s']}s  p95={metrics['p95_s']}s")

        finally:
            del vs
            shutil.rmtree(tmp_dir, ignore_errors=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados em {OUTPUT_JSON}")
    _plot(results)


def _plot(results: list[dict]):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        sizes   = [r["size"]   for r in results]
        avg     = [r["avg_s"]  for r in results]
        p95     = [r["p95_s"]  for r in results]
        idx_t   = [r["index_time_s"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Escalonamento do Vectorstore ChromaDB", fontsize=14, fontweight="bold")

        ax1.plot(sizes, avg, "o-", color="#0f4c5c", label="Média")
        ax1.fill_between(sizes, avg, p95, alpha=0.15, color="#0f4c5c", label="Avg–P95")
        ax1.plot(sizes, p95, "s--", color="#e67e22", label="P95")
        ax1.set_xlabel("Número de chunks no índice")
        ax1.set_ylabel("Tempo de retrieval (s)")
        ax1.set_title("Latência de Retrieval (similarity_search, k=5)")
        ax1.legend()
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2.bar([str(s) for s in sizes], idx_t, color="#0f4c5c", alpha=0.8)
        ax2.set_xlabel("Número de chunks")
        ax2.set_ylabel("Tempo de indexação (s)")
        ax2.set_title("Tempo de Indexação Inicial")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado em {OUTPUT_PNG}")
        plt.close()
    except ImportError:
        print("matplotlib não disponível — gráfico não gerado.")


if __name__ == "__main__":
    run()
