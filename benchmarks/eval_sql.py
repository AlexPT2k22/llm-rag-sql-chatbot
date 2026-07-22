import sys
import os
import time
import unicodedata
import requests


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII").lower()


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

GOLDEN_SET = [
    {"q": "Quantas parcelas existem?",                   "must": ["parcela"], "must_sql": ["parcela"]},
    {"q": "Quantas propriedades estão registadas?",      "must": ["propriedade"], "must_sql": ["propriedade"]},
    {"q": "Quantas cubas existem na cellar?",             "must": ["cuba"], "must_sql": ["cuba"]},
    {"q": "Quantos lotes de vinho temos?",               "must": ["lote"], "must_sql": ["lote"]},
    {"q": "Quantas cellars estão registadas?",            "must": ["cellar"], "must_sql": ["cellar"]},
    {"q": "Quantos distritos existem?",                  "must": ["distrito"], "must_sql": ["distrito"]},
    {"q": "Quantos concelhos existem?",                  "must": ["concelho"], "must_sql": ["concelho"]},
    {"q": "Quantas variedades de uva estão registadas?", "must": ["variedade", "casta"], "must_sql": ["variedade"]},
    {"q": "Quantos fitofármacos temos no sistema?",      "must": ["fitofarmac"], "must_sql": ["fitofarmaco"]},
    {"q": "Quantos equipamentos existem?",               "must": ["equipament"], "must_sql": ["equipamento"]},
    {"q": "Lista as freguesias registadas",              "must": ["freguesia"], "must_sql": ["freguesia"]},
    {"q": "Quantos tipos de cultura existem?",           "must": ["cultura"], "must_sql": ["cultura"]},
    {"q": "Lista os tanoeiros",                          "must": ["tanoeir"], "must_sql": ["tanoeiro"]},
    {"q": "Quantas barricas existem na cellar?",          "must": ["barric"], "must_sql": ["barrica"]},
    {"q": "Quantos perfis de solo estão definidos?",     "must": ["solo"], "must_sql": ["solo"]},

    #JOIN
    {"q": "Quantas parcelas existem em cada distrito?",          "must": ["distrito"], "must_sql": ["join", "distrito", "parcela"]},
    {"q": "Lista as cubas com o nome da cellar a que pertencem",  "must": ["cuba", "cellar"], "must_sql": ["join", "cuba", "cellar"]},
    {"q": "Quantas parcelas tem cada propriedade?",              "must": ["parcela", "propriedade"], "must_sql": ["join", "parcela", "propriedade"]},
    {"q": "Lista as variedades de uva por categoria de vinho",   "must": ["variedade", "categoria"], "must_sql": ["join", "variedade"]},
    {"q": "Quantos lotes existem por cellar?",                    "must": ["lote", "cellar"], "must_sql": ["join", "lote", "cellar"]},
    {"q": "Lista as propriedades com o seu concelho",            "must": ["propriedade", "concelho"], "must_sql": ["join", "propriedade", "concelho"]},
    {"q": "Quantas operações de rega tem cada parcela?",         "must": ["rega", "parcela"], "must_sql": ["join", "rega", "parcela"]},
    {"q": "Quantos equipamentos tem cada propriedade?",          "must": ["equipament", "propriedade"], "must_sql": ["join", "equipamento", "propriedade"]},
    {"q": "Lista as fermentações com o lote correspondente",     "must": ["ferment", "lote"], "must_sql": ["join", "fermentacao", "lote"]},
    {"q": "Quantas castas tem cada parcela?",                    "must": ["casta", "variedade", "parcela"], "must_sql": ["join", "parcela"]},
    {"q": "Qual a área total das parcelas?",                       "must": ["area", "área"], "must_sql": ["sum", "area"]},
    {"q": "Qual a média de capacidade das cubas?",                 "must": ["capacidade", "media", "média"], "must_sql": ["avg", "capacidade"]},
    {"q": "Qual a soma do volume de vinho tinto?",                 "must": ["tinto", "litro", "volume"], "must_sql": ["sum", "litragem", "tinto"]},
    {"q": "Qual a parcela com maior área?",                        "must": ["parcela", "area", "área"], "must_sql": ["max", "area"]},
    {"q": "Qual a cuba com maior capacidade?",                     "must": ["cuba", "capacidade"], "must_sql": ["max", "capacidade"]},
    {"q": "Qual o distrito com mais parcelas?",                    "must": ["distrito"], "must_sql": ["count", "distrito"]},
    {"q": "Qual o concelho com mais propriedades?",                "must": ["concelho"], "must_sql": ["count", "concelho"]},
    {"q": "Quantas parcelas há por tipo de cultura?",              "must": ["parcela", "cultura"], "must_sql": ["group by", "cultura"]},
    {"q": "Quantas cubas estão em cada estado?",                   "must": ["cuba", "estado"], "must_sql": ["group by", "estado"]},
    {"q": "Qual o total de equipamentos por tipo?",                "must": ["equipament", "tipo"], "must_sql": ["group by"]},

    #Multi-BD
    {"q": "Quantas parcelas e quantas cubas existem no total?",    "must": ["parcela", "cuba"], "must_sql": ["parcela", "cuba"]},
    {"q": "Compara o número de propriedades com o número de cellars", "must": ["propriedade", "cellar"], "must_sql": ["propriedade", "cellar"]},
    {"q": "Quantos fitofármacos e quantos produtos enológicos temos?", "must": ["fitofarmac", "enolog"], "must_sql": ["fitofarmaco"]},
    {"q": "Quantos equipamentos existem versus quantas barricas?", "must": ["equipament", "barric"], "must_sql": ["equipamento", "barrica"]},
    {"q": "Lista o total de parcelas, cubas e equipamentos",       "must": ["parcela", "cuba", "equipament"], "must_sql": ["parcela"]},

    #Linguagem difícil
    {"q": "Diz-me quantas parcelas é que existem afinal",           "must": ["parcela"], "must_sql": ["parcela"]},
    {"q": "Sabes me dizer o número de cubas?",                      "must": ["cuba"], "must_sql": ["cuba"]},
    {"q": "Quantas mesmo? Refiro-me a propriedades",                "must": ["propriedade"], "must_sql": ["propriedade"]},
    {"q": "Qual o total exato de lotes registados?",                "must": ["lote"], "must_sql": ["lote"]},
    {"q": "Preciso de saber quantos equipamentos temos em uso",     "must": ["equipament"], "must_sql": ["equipamento"]},
    {"q": "Listagem completa das cellars, por favor",                "must": ["cellar"], "must_sql": ["cellar"]},
    {"q": "Mostra-me todas as variedades de uva existentes",        "must": ["variedade", "casta", "uva"], "must_sql": ["variedade"]},
    {"q": "Tenho curiosidade: quantos distritos há?",               "must": ["distrito"], "must_sql": ["distrito"]},
    {"q": "Podes contar quantas parcelas tenho no sistema?",        "must": ["parcela"], "must_sql": ["parcela"]},
    {"q": "Qual a quantidade total de fitofármacos disponíveis?",   "must": ["fitofarmac"], "must_sql": ["fitofarmaco"]},
]


def run():
    total = len(GOLDEN_SET)
    answered = 0
    correct = 0
    sql_correct = 0
    times = []

    print(f"\n=== SQL Benchmark — {total} perguntas ===\n")

    for i, item in enumerate(GOLDEN_SET, 1):
        q = item["q"]
        must = [_norm(k) for k in item["must"]]
        must_sql = [_norm(k) for k in item.get("must_sql", [])]

        t0 = time.time()
        try:
            r = requests.post(API_URL, data={"message": q, "session_id": f"sqlbench-{i}"}, timeout=180)
            r.raise_for_status()
            data = r.json()
            raw_answer = data.get("answer", "")
            sql_query = data.get("sql_query") or ""
            answer = _norm(raw_answer)
            sql_norm = _norm(sql_query)
            elapsed = time.time() - t0
            times.append(elapsed)

            no_info = any(s in answer for s in [
                "nao tenho informacao", "nao consegui", "nao foi possivel",
                "erro", "nao encontrei",
            ])

            if no_info or not raw_answer.strip():
                status = "NO_INFO"
            else:
                answered += 1
                hit = any(k in answer for k in must)
                if hit:
                    correct += 1
                    if must_sql and any(k in sql_norm for k in must_sql):
                        sql_correct += 1
                        status = "OK+SQL"
                    elif must_sql:
                        status = "OK (sql?)"
                    else:
                        status = "OK"
                else:
                    status = f"MISS {must[:2]}"

            print(f"[{i:02d}/{total}] {status:12s} {elapsed:5.1f}s — {q[:60]}")
        except Exception as e:
            print(f"[{i:02d}/{total}] ERROR: {e}")

    print("\n=== Resultados ===")
    print(f"Total:               {total}")
    print(f"Respondidas:         {answered}/{total} ({100*answered/total:.0f}%)")
    print(f"Correctas (resposta):{correct}/{total} ({100*correct/total:.0f}%)")
    print(f"SQL correcto:        {sql_correct}/{total} ({100*sql_correct/total:.0f}%)")
    if times:
        print(f"Tempo médio:         {sum(times)/len(times):.1f}s")
        print(f"Tempo máximo:        {max(times):.1f}s")


if __name__ == "__main__":
    run()
