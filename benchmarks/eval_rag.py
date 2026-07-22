"""
Critérios:
- "Não tenho informação suficiente" → answered=False
- Caso contrário, verifica se contém keywords esperadas
"""
import sys
import os
import time
import json
import unicodedata
import requests

def _norm(text: str) -> str:
    """Lowercase + remove acentos para comparação tolerante."""
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII").lower()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

GOLDEN_SET = [
    # Cellar
    {"q": "O que faz o módulo Cellar?",                       "must": ["cellar"]},
    {"q": "Como funciona a gestão de lotes na cellar?",       "must": ["lote"]},
    {"q": "O que são as cubas na cellar?",                    "must": ["cuba"]},
    {"q": "Como registo uma vinificação?",                   "must": ["vinifica"]},
    {"q": "O que é o engarrafamento na cellar?",              "must": ["engarrafa"]},
    {"q": "Como faço uma análise de mosto?",                 "must": ["mosto"]},
    {"q": "Como registo uma fermentação?",                   "must": ["ferment"]},
    {"q": "Como faço a gestão de barricas?",                 "must": ["barric"]},
    {"q": "Como registo a entrada de uvas na cellar?",        "must": ["uva"]},
    {"q": "O que é a representação estrutural da cellar?",    "must": ["estrutur"]},
    {"q": "Como registo uma trasfega?",                      "must": ["trasfeg"]},
    {"q": "Como funciona a rotulagem de garrafas?",          "must": ["rotul"]},
    {"q": "O que é uma vindima?",                            "must": ["vindima"]},
    {"q": "Como faço uma certificação de vinho?",            "must": ["certific"]},
    {"q": "Como registo um lote a granel?",                  "must": ["granel"]},
    {"q": "Como configuro um produto enológico?",            "must": ["enolog"]},

    # Field Monitoring de Parcelas
    {"q": "O que faz o módulo de Field Monitoring de Parcelas?","must": ["parcela"]},
    {"q": "Como adiciono uma parcela?",                      "must": ["parcela"]},
    {"q": "Como crio uma propriedade?",                      "must": ["propriedade"]},
    {"q": "Como registo uma visita técnica?",                "must": ["visita"]},
    {"q": "Como agendo uma colheita?",                       "must": ["colheita"]},
    {"q": "Como registo um acidente climático?",             "must": ["acidente", "clim"]},
    {"q": "Como funciona o cadastro das explorações?",       "must": ["cadastro", "exploraç"]},
    {"q": "Como adiciono uma rota de monitorização?",        "must": ["rota"]},
    {"q": "Que tipos de exposição solar existem?",           "must": ["exposiç"]},
    {"q": "Como vejo o perfil de solo de uma parcela?",      "must": ["solo"]},
    {"q": "Como edito as coordenadas de uma propriedade?",   "must": ["coorden"]},
    {"q": "Como apago uma parcela?",                         "must": ["apag", "elimin"]},
    {"q": "Como funciona a gestão de pragas?",               "must": ["prag"]},
    {"q": "Que castas estão registadas no sistema?",         "must": ["cast"]},
    {"q": "Como vejo a área total de uma propriedade?",      "must": ["area", "área"]},
    {"q": "Como funciona o módulo de georreferenciação?",    "must": ["georref"]},

    # Field Ops Agrícolas
    {"q": "O que faz o módulo de Field Ops Agrícolas?",       "must": ["pratic"]},
    {"q": "Como registo uma prática cultural?",              "must": ["pratic"]},
    {"q": "Como faço o registo de fitofármacos?",            "must": ["fitofarmac"]},
    {"q": "Como vejo o stock de fitofármacos?",              "must": ["stock"]},
    {"q": "Como registo uma compra?",                        "must": ["compra"]},
    {"q": "Como gero um relatório de produção?",             "must": ["relator"]},
    {"q": "Como adiciono um equipamento?",                   "must": ["equipament"]},
    {"q": "Como faço a gestão de recursos humanos?",         "must": ["recurso", "human"]},
    {"q": "Como vejo os custos por parcela?",                "must": ["custo"]},
    {"q": "Como registo uma rota diária?",                   "must": ["rota"]},
    {"q": "Como funciona a gestão de orçamentos?",           "must": ["orcament", "orçament"]},
    {"q": "Como adiciono um empreiteiro?",                   "must": ["empreiteir"]},
    {"q": "Como registo uma operação de rega?",              "must": ["rega"]},
    {"q": "Como funciona a gestão de stocks de adubos?",     "must": ["adubo", "stock"]},
    {"q": "Como gero um relatório de tratamentos?",          "must": ["tratament"]},

    # Comparações entre módulos
    {"q": "Qual a diferença entre Cellar e Field Monitoring de Parcelas?", "must": ["cellar", "parcela"]},
    {"q": "Qual a diferença entre Cellar e Field Ops Agrícolas?",        "must": ["cellar", "pratic"]},
    {"q": "Qual a diferença entre Field Monitoring e Field Ops Agrícolas?","must": ["monitoriz", "pratic"]},
]


def run():
    total = len(GOLDEN_SET)
    answered = 0
    correct = 0
    times = []

    print(f"\n=== RAG Benchmark — {total} perguntas ===\n")

    for i, item in enumerate(GOLDEN_SET, 1):
        q = item["q"]
        must = [_norm(k) for k in item["must"]]

        t0 = time.time()
        try:
            r = requests.post(API_URL, data={"message": q, "session_id": f"bench-{i}"}, timeout=120)
            r.raise_for_status()
            data = r.json()
            raw_answer = data.get("answer", "")
            answer = _norm(raw_answer)
            elapsed = time.time() - t0
            times.append(elapsed)

            if "nao tenho informacao suficiente" in answer:
                status = "NO_INFO"
            else:
                answered += 1
                if all(k in answer for k in must):
                    correct += 1
                    status = "OK"
                else:
                    missing = [k for k in must if k not in answer]
                    status = f"MISSING {missing}"

            print(f"[{i:02d}/{total}] {status:20s} {elapsed:5.1f}s — {q[:60]}")
        except Exception as e:
            print(f"[{i:02d}/{total}] ERROR: {e}")

    print("\n=== Resultados ===")
    print(f"Total:           {total}")
    print(f"Respondidas:     {answered}/{total} ({100*answered/total:.0f}%)")
    print(f"Correctas:       {correct}/{total} ({100*correct/total:.0f}%)")
    if times:
        print(f"Tempo médio:     {sum(times)/len(times):.1f}s")
        print(f"Tempo máximo:    {max(times):.1f}s")


if __name__ == "__main__":
    run()
