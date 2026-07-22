import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.vectorstore import get_embeddings, load_vectorstore
TEST_CASES = [
    ("Como adicionar uma parcela?", "parcela"),
    ("Como registar um fitofármaco?", "fitofármaco"),
    ("Como funciona a gestão de rega?", "rega"),
    ("Como criar uma prática cultural?", "prática cultural"),
    ("Como exportar dados do AgriSystem?", "exportar"),
    ("O que é uma unidade de gestão?", "unidade de gestão"),
    ("Como registar uma colheita?", "colheita"),
    ("Como funciona o módulo de cellar?", "cellar"),
]

K = 4

def evaluate(vectorstore, test_cases, k):
    hits = 0
    misses = []
    for question, keyword in test_cases:
        results = vectorstore.similarity_search(question, k=k)
        found = any(keyword.lower() in r.page_content.lower() for r in results)
        if found:
            hits += 1
        else:
            misses.append((question, keyword))
    return hits, misses


def main():
    print("A carregar vectorstore...")
    embeddings = get_embeddings()
    vs = load_vectorstore(embeddings)

    print(f"A avaliar {len(TEST_CASES)} perguntas (top-{K})...\n")
    hits, misses = evaluate(vs, TEST_CASES, K)
    total = len(TEST_CASES)
    pct = 100 * hits / total

    print("=" * 50)
    print(f"Resultado: {hits}/{total} ({pct:.1f}%)")
    print("=" * 50)

    if misses:
        print(f"\nMisses ({len(misses)}):")
        for q, kw in misses:
            print(f"  [{kw}] <- \"{q}\"")

    sys.exit(0 if hits == total else 1)


if __name__ == "__main__":
    main()
