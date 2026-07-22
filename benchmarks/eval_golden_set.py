"""
Golden set benchmark for the SQL agent.
Each question is tested in isolation, without session history.

Usage:
  python benchmarks/eval_golden_set.py
  python benchmarks/eval_golden_set.py --entity 1 --year 2025

Default entity 1, year 2025 — use for reference.
Replace mock values with your actual entity:
  - Property: Quinta do Vale
  - Plot: Plot A1
  - HR: Worker Silva
  - Equipment: Crawler Tractor X100
  - Phytosanitary: AGRO-STAR
  - Fertilizer: FertiGro 7-14-14
"""
import sys, os, time, argparse, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.vectorstore import get_embeddings
from agents.sql_agent import SQLAgentTools
from langchain_ollama import ChatOllama
import core.config as cfg

JUDGE_PROMPT = """És um avaliador especialista em sistemas de chatbot agrícola.
Avalia se a resposta abaixo é CORRECTA para a pergunta do utilizador.

Pergunta: {question}
SQL gerado: {sql}
Resposta do chatbot: {answer}

REGRAS DE AVALIAÇÃO:
score=1 (CORRECTO) quando:
- A resposta contém dados relevantes (tabela, lista ou valores numéricos)
- A resposta diz "Não foram encontrados registos" E o SQL parece correcto para a pergunta (os dados podem genuinamente não existir)
- A resposta indica claramente a ausência de dados sem expor SQL

score=0 (ERRADO) quando:
- A resposta expõe código SQL visível ao utilizador
- A resposta contém mensagens de erro técnico (ex: "ERRO SQL", "column does not exist")
- O SQL usa claramente a tabela ERRADA para a pergunta (ex: tabela de manutenção para responder sobre horas de trabalho)
- A resposta pede ao utilizador para reformular em vez de responder

IMPORTANTE: "Não foram encontrados registos" NÃO é automaticamente score=0.
Se o SQL está correcto para a pergunta, pode ser que os dados realmente não existam — isso é score=1.

Responde APENAS com JSON:
{{"score": 1, "reason": "breve justificação"}}
{{"score": 0, "reason": "breve justificação"}}

JSON:"""

def llm_judge(question: str, answer: str, sql: str, judge_model: str) -> dict:
    """Avalia a resposta com LLM-as-Judge via Ollama (local ou cloud)."""
    if not answer or len(answer) < 15:
        return {"score": 0, "reason": "resposta vazia"}
    try:
        llm = ChatOllama(model=judge_model, temperature=0, num_ctx=2048, num_predict=512)
        prompt = JUDGE_PROMPT.format(
            question=question,
            sql=(sql[:300] if sql else "nenhum"),
            answer=answer[:600]
        )
        raw = llm.invoke(prompt).content.strip()
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        return {"score": -1, "reason": f"erro judge: {e}"}
    return {"score": -1, "reason": "parse falhou"}

# Valores reais para substituir XX
DEFAULTS = {
    "entidade":    1,
    "ano":         1,
    "propriedade": "Quinta do Vale",
    "parcela":     "Parcela A1",
    "rh":          "Worker Silva",
    "empreiteiro": "Contractor A",
    "equipamento": "Crawler Tractor X100",
    "fitofarmaco": "AGRO-STAR",
    "adubo":       "FertiGro 7-14-14",
    # Cellar — entity 1, year 2025
    "cellar_entidade":   1,
    "cellar_ano":        2025,
    "lote":             "DOC Red",
    "cuba":             "Tank 01",
    "rh_cellar":         "Worker Maria",
    "consumivel":       "Bottle 0.75 Standard",
    "consumivel_enol":  "Sulfur Dioxide Solution",
}

def build_golden_set(d):
    return [
        # Processo 1
        ("P1-01", f"Quais as culturas existentes e quais as respetivas áreas na {d['propriedade']}"),
        ("P1-02",  "Qual a area da cultura vinha existente no concelho de Vila Real"),
        ("P1-03",  "Qual é a área das parcelas da cultura vinha e da variedade Touriga-Franca"),
        ("P1-04",  "Qual é a área de vinha com modo de produção Produção Integrada"),
        ("P1-05",  "Listar por parcela os sistemas de condução e as respetivas áreas das parcelas"),
        ("P1-06", f"Quais os porta-enxertos associados nas parcelas da {d['propriedade']}"),
        ("P1-07",  "Listar os tipos de rega existentes nas propriedades da entidade organizado por propriedade parcela área e tipo de rega"),
        ("P1-08", f"Listar parcelas da {d['propriedade']} com a informação de armação tipo de embardamento estado embardamento e ano de plantação"),
        ("P1-09", f"Com base na área e do compasso da linha e entrelinha da {d['parcela']} calcula o número de plantas estimado"),
        ("P1-10",  "Apresenta média de altitudes e declives para cada propriedade de acordo com a informação registada nas parcelas"),

        # Processo 2
        ("P2-01",  "Apresenta as informações registadas nas práticas do tipo aplicação de fitofármacos durante o ano de 2025"),
        ("P2-02", f"Qual o período custo e tempo despendido na prática do tipo PODA na {d['propriedade']}"),
        ("P2-03", f"Onde foi aplicado o produto fitofármaco {d['fitofarmaco']} durante o ano de 2025"),
        ("P2-04", f"O {d['adubo']} foi aplicado em que parcelas e quais as quantidades aplicadas durante o ano 2025"),
        ("P2-05", f"Listar dias trabalhados do {d['empreiteiro']} durante o ano de 2025 agrupado por mês"),
        ("P2-06", f"Apresenta o histórico de todas as práticas realizadas pelo Recurso Humano {d['rh']}"),
        ("P2-07", f"Quais as práticas culturais realizadas em 2025 pelo equipamento {d['equipamento']}"),
        ("P2-08",  "Apresenta as horas trabalhadas e custos associados por tipo de prática cultural para o ano 2025"),
        ("P2-09",  "Apresenta as horas trabalhadas e custos associados por tipo de recurso para o ano 2025"),
        ("P2-10",  "Apresenta custos de produção e total de horas trabalhadas para 2025 agrupado por mês"),

        # Processo 3
        ("P3-01", f"Lista as práticas culturais com custos e tempos de trabalho da {d['propriedade']}"),
        ("P3-02", f"Lista as práticas culturais com custos e tempos de trabalho da {d['parcela']}"),
        ("P3-03", f"Lista as práticas culturais do tipo PODA com custos e tempos de trabalho da {d['parcela']} agrupado por Variedade para a {d['propriedade']}"),
        ("P3-04", f"Compara custos e horas trabalhadas das parcelas por Tipo de Cultura para a {d['propriedade']}"),
        ("P3-05",  "Compara custos e horas trabalhadas das parcelas por Armação para a cultura da vinha"),
        ("P3-06",  "Apresenta os custos e horas trabalhadas por tipo de recurso para cada propriedade da Entidade"),
        ("P3-07",  "Apresenta os custos e horas trabalhadas por tipo de recurso para cada parcela de todas as propriedades da Entidade"),
        ("P3-08",  "Cria um calendário com as horas trabalhadas nas práticas culturais para os Recursos Humanos"),
        ("P3-09",  "Cria um calendário com as horas trabalhadas nas práticas culturais para os Equipamentos"),
        ("P3-10", f"Cria um relatório com total de horas trabalhadas pelo {d['empreiteiro']} agrupado por cargo e função com as horas e dias trabalhados durante o ano de 2025"),

        # Módulo Cellar (entidade {d['cellar_entidade']}, ano {d['cellar_ano']})
        ("A-01",  "Lista todas as vinificações de tintos para o ano 2025"),
        ("A-02",  "Lista todas as vinificações de brancos agrupado por variedades para o ano 2025"),
        ("A-03", f"Lista todas as operações associadas ao lote {d['lote']}"),
        ("A-04", f"Lista o custo e horas trabalhadas no lote {d['lote']}"),
        ("A-05",  "Lista as operações onde existiu uso de consumíveis e apresenta a quantidade gasta em cada operação"),
        ("A-06",  "Lista as quantidades de vinho existentes agrupadas por tipo de vinho"),
        ("A-07",  "Lista as quantidades de vinho existentes em cada fase ou estado de vinificação"),
        ("A-08",  "Lista a quantidade de vinho existente por tipo de armazenamento: cubas barricas e garrafas"),
        ("A-09",  "Lista a quantidade de perdas de vinho apresentadas por cada uma das fases da cellar"),
        ("A-10", f"Calcula o custo de produção do litro de vinho do lote {d['lote']} que saiu para comercialização"),
        ("A-11",  "Lista a quantidade de vinho vendida por cliente"),
        ("A-12",  "Lista as operações e quantidades consumidas do produto enológico Anidrido Sulfuroso"),
        ("A-13", f"Lista o histórico de utilização do consumível {d['consumivel']}"),
        ("A-14",  "Lista o histórico de utilização dos consumíveis usados no ano 2025"),
        ("A-15", f"Lista as operações e horas trabalhadas do Recurso Humano {d['rh_cellar']}"),
        ("A-16",  "Cria um calendário com as horas trabalhadas nas operações para os Recursos Humanos"),
        ("A-17", f"Lista o histórico de operações da Cuba {d['cuba']} durante o ano 2025"),
        ("A-18", f"Valida se no Lote {d['lote']} foi efetuada a operação de Registo de Amostra"),
        ("A-19",  "Calcula os custos e horas trabalhadas de todos os lotes registados em 2025"),
        ("A-20",  "Calcula custos e horas trabalhadas em 2025 por tipo de vinho"),
    ]


def run_benchmark(entidade_id: int, ano_agricola_id: int, defaults: dict, model: str = None, judge_model: str = None, filter_prefix: str = None):
    model = model or cfg.SQL_MODEL
    use_judge = judge_model is not None
    print(f"\n{'='*70}")
    print(f"GOLDEN SET — entidade={entidade_id}, ano={ano_agricola_id}, modelo={model}")
    if use_judge:
        print(f"LLM-as-Judge: {judge_model}")
    print(f"{'='*70}\n")

    embeddings = get_embeddings()
    agent = SQLAgentTools(embeddings=embeddings, llm_model=model)
    for name, url in cfg.DATABASES.items():
        agent.add_database(name, url)
    agent.setup()

    golden_set = build_golden_set(defaults)
    if filter_prefix:
        golden_set = [(qid, q) for qid, q in golden_set if qid.startswith(filter_prefix)]
        print(f"Filtro activo: '{filter_prefix}' — {len(golden_set)} perguntas")
    results = []
    pipeline_ok = pipeline_fail = fallback_ok = fallback_fail = 0

    for qid, question in golden_set:
        print(f"\n[{qid}] {question[:90]}")

        SQLAgentTools._metrics = {k: 0 for k in SQLAgentTools._metrics}
        eid = defaults.get("cellar_entidade", entidade_id) if qid.startswith("A-") else entidade_id
        aid = defaults.get("cellar_ano", ano_agricola_id) if qid.startswith("A-") else ano_agricola_id

        try:
            answer, sql, elapsed = agent.answer(
                question=question,
                chat_history=None,
                entidade_id=eid,
                ano_agricola_id=aid,
            )
            m = SQLAgentTools._metrics
            used_pipeline = m["pipeline_ok"] > 0
            has_data = bool(answer) and len(answer) > 30 and "não foram encontrados" not in answer.lower()

            status = "✓" if has_data else "∅"
            path = "PIPELINE" if used_pipeline else "FALLBACK"

            # LLM-as-Judge
            judge_result = None
            if use_judge:
                judge_result = llm_judge(question, answer, sql, judge_model)
                judge_icon = "J✓" if judge_result["score"] == 1 else ("J∅" if judge_result["score"] == 0 else "J?")
                print(f"  [{status}] {judge_icon} {path} | {elapsed:.1f}s | {judge_result['reason'][:60]}")
            else:
                print(f"  [{status}] {path} | {elapsed:.1f}s")

            if sql:
                print(f"  SQL: {sql[:100]}")
            print(f"  Resp: {answer[:120]}")

            if used_pipeline:
                pipeline_ok += 1
            elif m["fallback_ok"] > 0:
                fallback_ok += 1
            else:
                fallback_fail += 1

            results.append({
                "id": qid, "question": question, "status": status,
                "path": path, "elapsed": round(elapsed, 1),
                "sql": sql, "answer": answer[:400],
                "judge": judge_result,
            })

        except Exception as e:
            print(f"  [✗] ERRO: {e}")
            fallback_fail += 1
            results.append({
                "id": qid, "question": question, "status": "✗",
                "path": "ERRO", "elapsed": 0.0, "sql": "", "answer": str(e)[:200],
            })

    total = len(golden_set)
    print(f"\n{'='*70}")
    print(f"RESULTADOS ({total} perguntas)")
    print(f"  Pipeline OK:   {pipeline_ok}")
    print(f"  Pipeline fail: {pipeline_fail}")
    print(f"  Fallback OK:   {fallback_ok}")
    print(f"  Fallback fail: {fallback_fail}")
    print(f"  Com dados (✓): {sum(1 for r in results if r['status']=='✓')}")
    print(f"  Sem dados (∅): {sum(1 for r in results if r['status']=='∅')}")
    print(f"  Erros    (✗): {sum(1 for r in results if r['status']=='✗')}")
    if use_judge:
        j_ok  = sum(1 for r in results if r.get("judge") and r["judge"].get("score") == 1)
        j_nok = sum(1 for r in results if r.get("judge") and r["judge"].get("score") == 0)
        print(f"  Judge ✓:       {j_ok}/{total} ({round(j_ok/total*100)}%)")
        print(f"  Judge ✗:       {j_nok}/{total} ({round(j_nok/total*100)}%)")
    print(f"{'='*70}\n")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    model_tag = model.replace(":", "_").replace("/", "_")
    out_file = os.path.join(out_dir, f"golden_set_ent{entidade_id}_{model_tag}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"entidade": entidade_id, "ano": ano_agricola_id,
                   "defaults": defaults, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Guardado em {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entidade", type=int, default=DEFAULTS["entidade"])
    parser.add_argument("--ano",      type=int, default=DEFAULTS["ano"])
    parser.add_argument("--model",       type=str, default=None,
                        help="Modelo SQL (ex: qwen2.5:14b). Default: SQL_MODEL do config.")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Modelo judge (ex: qwen3-coder:480b-cloud). Omitir para não usar judge.")
    parser.add_argument("--filter", type=str, default=None,
                        help="Prefixo para filtrar perguntas (ex: A- para só Cellar, P1 para só P1).")
    args = parser.parse_args()

    d = dict(DEFAULTS)
    d["entidade"] = args.entidade
    d["ano"]      = args.ano

    run_benchmark(args.entidade, args.ano, d, model=args.model, judge_model=args.judge_model, filter_prefix=args.filter)
