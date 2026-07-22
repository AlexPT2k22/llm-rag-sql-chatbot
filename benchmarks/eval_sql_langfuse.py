"""
Experiment runner para testar prompt versions do SQL Agent via Langfuse.

Usa o dataset "sql-agent-golden-set" e lf.run_experiment() para registar resultados.

Uso:
    cd entregaEstagio
    python -m benchmarks.eval_sql_langfuse                      # usa prompt production
    python -m benchmarks.eval_sql_langfuse --run-name "v2-test" # nome custom
"""

import argparse
import json
import os
import sys
import time
import unicodedata

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from langfuse import Langfuse, Evaluation

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DATASET_NAME = "sql-agent-golden-set"


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII").lower()


def task(item):
    """Envia a pergunta ao SQL Agent e devolve a resposta."""
    question = item.input["question"]
    params = {"message": question, "session_id": f"exp-{int(time.time())}-{id(item)}"}

    resp = requests.get(f"{API_URL}/chat/stream", params=params, stream=True, timeout=180)
    resp.raise_for_status()

    answer = ""
    sql_query = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            evt = json.loads(line[6:])
            if evt.get("type") == "token":
                answer += evt.get("token", "")
            elif evt.get("type") == "sql":
                sql_query = evt.get("query", "")
        except (json.JSONDecodeError, KeyError):
            pass

    return {"answer": answer, "sql_query": sql_query}


def correctness_evaluator(*, output, expected_output, **kwargs):
    """Avalia se a resposta contém as keywords esperadas."""
    answer = _norm(output.get("answer", ""))
    sql = _norm(output.get("sql_query", ""))
    keywords_str = expected_output.get("keywords", "")
    keywords = [_norm(k.strip()) for k in keywords_str.split(",")]

    no_info = any(s in answer for s in [
        "nao tenho informacao", "nao consegui", "nao foi possivel", "nao encontrei"
    ])

    if no_info or not answer.strip():
        return Evaluation(name="correctness", value=0.0, comment="NO_INFO")

    hit = any(k in answer or k in sql for k in keywords)
    return Evaluation(
        name="correctness",
        value=1.0 if hit else 0.0,
        comment="OK" if hit else f"MISS: {keywords_str}",
    )


def run(run_name: str):
    lf = Langfuse()
    dataset = lf.get_dataset(DATASET_NAME)

    print(f"\n=== Experiment: {run_name} — {len(dataset.items)} perguntas ===\n")

    result = lf.run_experiment(
        name=DATASET_NAME,
        run_name=run_name,
        data=dataset.items,
        task=task,
        evaluators=[correctness_evaluator],
        max_concurrency=1,
    )

    print(f"\nResultados visíveis em: Langfuse → Datasets → {DATASET_NAME} → Runs → {run_name}")
    lf.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None, help="Nome do experiment run")
    args = parser.parse_args()

    if not args.run_name:
        try:
            lf = Langfuse()
            p = lf.get_prompt("sql-agent-system", type="text")
            args.run_name = f"prompt-v{p.version}"
        except Exception:
            args.run_name = f"run-{int(time.time())}"

    run(args.run_name)
