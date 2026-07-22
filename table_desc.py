"""
Gera descricoes automaticas para todas as tabelas/views das BDs com LLM local (Ollama).
Resultado guardado em documents/table_descriptions.json
"""
import json
import psycopg2
import requests
import time
import os
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"
BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "documents" / "table_descriptions.json"

DATABASES = {
    "Operations": os.getenv("DB_OPERATIONS_URL"),
    "Plots":      os.getenv("DB_PLOTS_URL"),
    "Cellar":     os.getenv("DB_CELLAR_URL"),
}

SKIP_TABLES = {
    "spatial_ref_sys", "geography_columns", "geometry_columns", "sync_log",
    "report", "report_aux", "report_content_queries", "report_fields",
    "configuracao_dynagrid", "funcionalidade", "funcionalidade_gr",
    "funcionalidade_grfc", "modulo_menu", "tipo_menu",
    "alert_notification_control", "alerta", "alerta_op", "alerta_utilizador",
    "utilizador", "utilizador_funcionalidade_acao", "adega_utilizador",
}

PROMPT_TEMPLATE = """Es um arquiteto de dados especialista em sistemas agricolas e vitivinicolas.

Analisa a seguinte tabela da base de dados "{db_name}" do sistema AgriSystem:

Tabela: {table_name}
Tipo: {table_type}

Colunas e Tipos:
{columns}

Amostra de dados (ate 3 linhas):
{sample}

A tua tarefa e catalogar esta tabela para alimentar um sistema de Busca Semantica (RAG) que ajuda um Chatbot a traduzir perguntas em linguagem natural para SQL.

Responde APENAS em JSON valido com este formato exacto, sem texto adicional:
{{
  "descricao": "Explica o que a tabela guarda e INCLUI OBRIGATORIAMENTE 2 a 3 exemplos de valores reais que leste na amostra.",
  "uso_chatbot": "sim | parcial | nao",
  "justificacao_uso": "Explicacao tecnica curta do porque desta classificacao."
}}

Criterios estritos para 'uso_chatbot':
- "sim": Tabela de negocio principal, metricas ou factos (ex: parcelas, tarefas, colheitas, lotes, compras).
- "parcial": Tabela de referencia, dicionario, dimensao ou categorias (ex: listas de concelhos, castas, estados, unidades). Estas tabelas sao VITAIS para filtrar dados e fazer JOINs.
- "nao": EXCLUSIVAMENTE tabelas de sistema tecnico (ex: logs, auditoria, passwords, gestao de menus, geometria PostGIS).
"""

def get_tables(cur, skip):
    cur.execute("""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type IN ('BASE TABLE', 'VIEW')
        UNION ALL
        SELECT matviewname AS table_name, 'MATERIALIZED VIEW' AS table_type
        FROM pg_matviews
        WHERE schemaname = 'public'
        ORDER BY table_type, table_name
    """)
    return [(r[0], r[1]) for r in cur.fetchall() if r[0] not in skip]


def get_columns(cur, table_name):
    cur.execute("""
        SELECT column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        UNION ALL
        SELECT a.attname AS column_name, format_type(a.atttypid, a.atttypmod) AS data_type, a.attnum::integer AS ordinal_position
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        JOIN pg_matviews m ON m.matviewname = c.relname AND m.schemaname = n.nspname
        WHERE n.nspname = 'public' AND c.relname = %s
          AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY ordinal_position
    """, (table_name, table_name))
    return [(r[0], r[1]) for r in cur.fetchall()]


def get_sample(cur, table_name):
    try:
        cur.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        if not rows:
            return "(tabela vazia)"
        lines = ["  |  ".join(cols)]
        for row in rows:
            lines.append("  |  ".join(str(v)[:50] if v is not None else "NULL" for v in row))
        return "\n".join(lines)
    except Exception as e:
        return f"(erro ao ler dados: {e})"


def ask_llm(prompt):
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 4096}
        }, timeout=500)
        resp.raise_for_status()
        text = resp.json()["response"].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"descricao": text, "uso_chatbot": "?", "justificacao_uso": "erro ao parsear JSON"}
    except Exception as e:
        return {"descricao": f"erro: {e}", "uso_chatbot": "?", "justificacao_uso": ""}


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Retomando — {sum(len(v) for v in results.values())} tabelas ja processadas")
    else:
        results = {db: {} for db in DATABASES}

    total_tables = 0
    for db_name, db_url in DATABASES.items():
        if not db_url:
            print(f"A saltar {db_name} — URL nao configurada")
            continue
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        tables = get_tables(cur, SKIP_TABLES)
        total_tables += len(tables)
        print(f"\n=== {db_name} — {len(tables)} tabelas/views ===")

        for i, (table_name, table_type) in enumerate(tables, 1):
            if table_name in results.get(db_name, {}):
                print(f"  [{i}/{len(tables)}] {table_name} — ja processada, a saltar")
                continue

            columns = get_columns(cur, table_name)
            cols_str = "\n".join(f"  - {col} ({dtype})" for col, dtype in columns)
            sample_str = get_sample(cur, table_name)
            type_label = "View" if table_type == "VIEW" else "Tabela"

            prompt = PROMPT_TEMPLATE.format(
                db_name=db_name,
                table_name=table_name,
                table_type=type_label,
                columns=cols_str,
                sample=sample_str,
            )

            print(f"  [{i}/{len(tables)}] {table_name}...", end=" ", flush=True)
            t0 = time.time()
            desc = ask_llm(prompt)
            elapsed = time.time() - t0
            print(f"{desc.get('uso_chatbot','?')} ({elapsed:.1f}s)")

            if db_name not in results:
                results[db_name] = {}

            results[db_name][table_name] = {
                "tipo": type_label,
                "colunas": [col[0] for col in columns],
                "schema_ddl": cols_str,
                "dados_amostra": sample_str,
                **desc
            }

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        cur.close()
        conn.close()

    print(f"\nConcluido — {total_tables} tabelas processadas")
    print(f"Resultado em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
