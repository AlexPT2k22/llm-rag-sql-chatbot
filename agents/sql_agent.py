import psycopg2
import os
import re
import time
import unicodedata
import networkx as nx
import core.config as cfg
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAIN_CHROMA_DIR = os.path.join(BASE_DIR, "chroma_domains")
TABLE_CHROMA_DIR = os.path.join(BASE_DIR, "chroma_tables")
_descriptions_file = os.path.join(BASE_DIR, "documents", "table_descriptions.json")

try:
    with open(_descriptions_file, encoding="utf-8") as _f:
        TABLE_DESCRIPTIONS = json.load(_f)
except FileNotFoundError:
    TABLE_DESCRIPTIONS = {}


def sanitize(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).encode("utf-8", "replace").decode("utf-8")


SKIP_TABLES = {"spatial_ref_sys", "sync_log", "geography_columns", "geometry_columns"}
DOMAIN_GROUPS = {
    "plots_and_fields": {
        "db": "Operations",
        "desc": "Parcelas agrícolas, propriedades, geodados, variedades plantadas, culturas, áreas, armação, embardamento, rega, porta-enxerto, sistema de condução, modo de produção, unidades de gestão",
        "prefixes": ["parcela", "propriedade", "unidade_gestao", "variedade", "v_field"],
    },
    "phytopharmaceuticals": {
        "db": "Operations",
        "desc": "Produtos fitofarmacêuticos, caldas, registos de uso, empresas, histórico de aplicações",
        "prefixes": ["fitofarmaco", "fito"],
    },
    "operations_ref": {
        "db": "Operations",
        "desc": "Tabelas de referência da monitorização: tipos de cultura, solo, rega, risco, espécie, produção, região, exposição, gama",
        "prefixes": ["tipo_"],
    },
    "operations_geo": {
        "db": "Operations",
        "desc": "Geographic and administrative data: districts, municipalities, parishes, geological soil profile, agricultural year, entities, risks",
        "prefixes": ["distrito", "concelho", "freguesia", "perfil_solo", "ano_agricola", "entidade", "risco", "fase_parcelar", "outro_detentor"],
    },
    "plots_operations": {
        "db": "Plots",
        "desc": "Práticas culturais agrícolas: operações, caldas, fitofármacos, colheitas, rega, fertilização, monitorização, recursos humanos, equipamentos, riscos, custos, horas trabalhadas, empreiteiros, adubos",
        "prefixes": ["pratica_cultural", "pratica_agricola", "grupo_pratica", "pc_", "view_", "v_plot"],
    },
    "resources": {
        "db": "Plots",
        "desc": "Recursos da exploração: equipamentos, manutenção, recursos humanos, avaliações, produtos, adubos, fitofármacos, formação e licenças",
        "prefixes": ["recurso", "rh_formacao"],
    },
    "routes_planning": {
        "db": "Plots",
        "desc": "Rotas de trabalho agrícola, planeamento, intervalos, consumíveis, mapas de rota, parcelas por rota, pontos GPS, coordenadas geográficas, latitude, longitude, rota_map",
        "prefixes": ["rota"],
    },
    "purchases_stock": {
        "db": "Plots",
        "desc": "Compras, requisições de compra, fornecedores de produtos, stock, baixas de stock, produtos por lote, orçamentos",
        "prefixes": ["compra", "requisicao", "fornecedor", "baixa_stock", "produto_lote", "orcamento"],
    },
    "reports": {
        "db": "Plots",
        "desc": "Relatórios configuráveis, campos de relatório, queries de conteúdo, tipos de relatório",
        "prefixes": ["report"],
    },
    "plots_ref": {
        "db": "Plots",
        "desc": "Tabelas de referência para práticas: tipos de prática cultural, equipamento, nutriente, risco, manutenção, formação, superfície tratada",
        "prefixes": ["tipo_", "valor_nutriente"],
    },
    "plots_management": {
        "db": "Plots",
        "desc": "Gestão geral: anos agrícolas, entidades, utilizadores, contadores, rebanhos, parcelas, propriedades, variedades, funcionalidades",
        "prefixes": ["ano_agricola", "entidade", "utilizador", "contador", "rebanho", "parcela", "propriedade", "variedade", "funcionalidade", "risco", "operacao_grupo"],
    },
    "cellar_reception": {
        "db": "Cellar",
        "desc": "Receção de uva e vindima: entradas de uva, pesagens, prensagem, desencuba, fermentação, análises durante fermentação, remontagem, delestage",
        "prefixes": ["rececao", "pesagem", "op_rececao", "entrada_granel"],
    },
    "cellar_lots_tanks": {
        "db": "Cellar",
        "desc": "Lotes de vinho e cubas/depósitos: lote, cuba, composição de lotes, fase do lote, histórico de lote, lote composto, lote de armazenamento, barrica",
        "prefixes": ["lote", "cuba", "adega_lote", "adega_cuba", "barrica"],
    },
    "cellar_operations": {
        "db": "Cellar",
        "desc": "Operações de adega: trasfega, filtragem, estabilização, engarrafamento, rotulagem, embalamento, venda a granel, amostras, acertos, perdas, limpeza",
        "prefixes": ["op_armazenamento", "op_embalamento", "op_admin", "op_mac", "op_relocaliza", "operacao_geral_adega", "opga"],
    },
    "cellar_bottling": {
        "db": "Cellar",
        "desc": "Engarrafamento e embalamento: operações de engarrafar, rotular, embalar, certificações, submissões de certificação, composição de certificados",
        "prefixes": ["lote_certificacao", "lote_op_embalamento", "lote_op_armazenamento"],
    },
    "cellar_management": {
        "db": "Cellar",
        "desc": "Gestão da adega: adegas vinícolas, utilizadores, anos agrícolas, entidades, produtos enológicos, consumíveis, fornecedores, stock, compras, equipamentos de adega",
        "prefixes": ["adega", "utilizador", "ano_agricola", "entidade", "produto", "consumivel", "fornecedor", "stock", "equipamento_adg"],
    },
    "cellar_ref": {
        "db": "Cellar",
        "desc": "Tabelas de referência da adega: tipos de vinho, categorias, derivação, fases, tipos de cuba, materiais de barrica, tanoeiros, variedades, tipo de cultura",
        "prefixes": ["vinho", "tipo_", "fase", "lote_tipo", "tanoeiro", "variedade", "barrica_tipo", "barrica_modelo"],
    },
    "cellar_views": {
        "db": "Cellar",
        "desc": "Views de operações de vinificação, volumes de vinho, perdas, consumíveis e calendário RH da adega: operações por lote e cuba, custos e horas, volumes por tipo de vinho e fase, perdas por fase, calendário de horas RH",
        "prefixes": ["v_cellar", "view_operacao", "view_historico"],
    },

}


_few_shot_file = os.path.join(BASE_DIR, "documents", "few_shot_sql.json")
try:
    with open(_few_shot_file, encoding="utf-8") as _f:
        _FEW_SHOT_EXAMPLES = json.load(_f)
except FileNotFoundError:
    _FEW_SHOT_EXAMPLES = []


class SQLAgentTools:
    CACHE_TTL = 300  # segundos (5 min)
    def __init__(self, embeddings, llm_model="qwen2.5:7b"):
        self.embeddings = embeddings
        self.llm_model = llm_model
        self.db_urls = {}
        self.table_schemas = {}
        self.table_fks = {}
        self.fk_graphs = {}
        self.domain_vectorstore = None
        self.table_vectorstore = None
        self.bm25_index = None
        self.bm25_docs = []
        self._few_shot_vectors = None
        self.agent = None
        self._cache = {}
        self._langfuse_handler = cfg.get_langfuse_handler()
    def add_database(self, name, url):
        self.db_urls[name] = url
    def setup(self):
        """Extrai schemas e cria vectorstores + agent com tools."""
        print("  A extrair schemas das bases de dados...")
        table_docs = []
        for db_name, url in self.db_urls.items():
            table_docs.extend(self._extract_schemas(db_name, url))
        print(f"  {len(self.table_schemas)} tabelas extraídas.")
        self._build_fk_graphs()
        print(f"  Grafos FK construídos: {list(self.fk_graphs.keys())}")
        if os.path.exists(DOMAIN_CHROMA_DIR):
            print("  Domain vectorstore já existe, a carregar...")
            self.domain_vectorstore = Chroma(
                persist_directory=DOMAIN_CHROMA_DIR,
                embedding_function=self.embeddings,
                collection_name="domains"
            )
        else:
            print("  A criar domain vectorstore...")
            domain_docs = self._build_domain_docs()
            self.domain_vectorstore = Chroma.from_documents(
                documents=domain_docs,
                embedding=self.embeddings,
                persist_directory=DOMAIN_CHROMA_DIR,
                collection_name="domains"
            )
            print(f"  {len(domain_docs)} domínios criados.")
        if os.path.exists(TABLE_CHROMA_DIR):
            print("  Table vectorstore já existe, a carregar...")
            self.table_vectorstore = Chroma(
                persist_directory=TABLE_CHROMA_DIR,
                embedding_function=self.embeddings,
                collection_name="tables"
            )
        else:
            print("  A criar table vectorstore...")
            self.table_vectorstore = Chroma.from_documents(
                documents=table_docs,
                embedding=self.embeddings,
                persist_directory=TABLE_CHROMA_DIR,
                collection_name="tables"
            )
            print(f"  {len(table_docs)} tabelas indexadas.")

        # Índice BM25
        from rank_bm25 import BM25Okapi
        self.bm25_docs = table_docs
        if table_docs:
            self.bm25_index = BM25Okapi([doc.page_content.lower().split() for doc in table_docs])
            print(f"  BM25 indexado: {len(table_docs)} documentos.")
        else:
            self.bm25_index = None
            print("  BM25 nao indexado: 0 tabelas disponiveis.")

        # Few-shot
        if _FEW_SHOT_EXAMPLES:
            questions = [ex["question"] for ex in _FEW_SHOT_EXAMPLES]
            self._few_shot_vectors = self.embeddings.embed_documents(questions)
            print(f"  Few-shot indexado: {len(_FEW_SHOT_EXAMPLES)} exemplos.")

        self._create_agent()
    def _extract_schemas(self, db_name, url):
        conn = psycopg2.connect(url)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type IN ('BASE TABLE', 'VIEW')
                UNION
                SELECT matviewname AS table_name
                FROM pg_matviews
                WHERE schemaname = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall() if row[0] not in SKIP_TABLES]
            cur.execute("""
                SELECT table_name, column_name, data_type, is_nullable, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = 'public'
                UNION ALL
                SELECT
                    c.relname AS table_name,
                    a.attname AS column_name,
                    format_type(a.atttypid, a.atttypmod) AS data_type,
                    CASE WHEN a.attnotnull THEN 'NO' ELSE 'YES' END AS is_nullable,
                    a.attnum::integer AS ordinal_position
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                JOIN pg_matviews m ON m.matviewname = c.relname AND m.schemaname = n.nspname
                WHERE n.nspname = 'public'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                ORDER BY table_name, ordinal_position
            """)
            columns_by_table = {}
            for table, col, dtype, nullable, _pos in cur.fetchall():
                columns_by_table.setdefault(table, []).append((col, dtype, nullable))
            cur.execute("""
                SELECT DISTINCT tc.table_name, kcu.column_name, ccu.table_name, ccu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
            """)
            for table, col, ref_table, ref_col in cur.fetchall():
                self.table_fks.setdefault((db_name, table), []).append((col, ref_table, ref_col))
            cur.close()
        finally:
            conn.close()
        docs = []
        for table in tables:
            cols = columns_by_table.get(table, [])
            col_lines = [f"  {c[0]} ({c[1]})" for c in cols]
            fk_lines = [f"  {fk[0]} -> {fk[1]}.{fk[2]}" for fk in self.table_fks.get((db_name, table), [])]
            schema_text = f"Base de dados: {db_name}\nTabela: {table}\nColunas:\n" + "\n".join(col_lines)
            if fk_lines:
                schema_text += "\nChaves estrangeiras:\n" + "\n".join(fk_lines)
            self.table_schemas[(db_name, table)] = schema_text
            col_names_readable = ", ".join([c[0].replace("_", " ") for c in cols])
            search_text = f"Tabela {table.replace('_', ' ')} na base de dados {db_name}. Colunas: {col_names_readable}"
            desc = TABLE_DESCRIPTIONS.get(db_name, {}).get(table, {}).get("descricao", "")
            if desc:
                search_text += f"\nDescricao: {desc}"
            docs.append(Document(
                page_content=search_text,
                metadata={"db_name": db_name, "table_name": table}
            ))
        return docs
    def _build_fk_graphs(self):
        """Constrói um grafo dirigido de FKs por BD para pathfinding entre tabelas."""
        for db_name in self.db_urls:
            G = nx.DiGraph()
            for (db, table) in self.table_schemas:
                if db == db_name:
                    G.add_node(table)
            for (db, table), fks in self.table_fks.items():
                if db == db_name:
                    for col, ref_table, ref_col in fks:
                        if ref_table in G:
                            G.add_edge(table, ref_table, col=col, ref_col=ref_col)
                            G.add_edge(ref_table, table, col=ref_col, ref_col=col)
            self.fk_graphs[db_name] = G
    def _build_domain_docs(self):
        docs = []
        for domain_name, group in DOMAIN_GROUPS.items():
            db = group["db"]
            desc = group["desc"]
            prefixes = group["prefixes"]
            domain_tables = []
            for (db_name, table_name) in self.table_schemas:
                if db_name == db and any(table_name.startswith(p) for p in prefixes):
                    domain_tables.append(table_name)
            table_list = ", ".join(sorted(domain_tables))
            search_text = f"Domínio: {domain_name.replace('_', ' ')}. {desc}. Tabelas: {table_list}"
            docs.append(Document(
                page_content=search_text,
                metadata={"domain_name": domain_name, "db_name": db, "num_tables": len(domain_tables)}
            ))
        return docs
    def _create_agent(self):
        """Cria o ReAct agent com tools."""
        agent_ref = self
        @tool
        def search_tables(question: str) -> str:
            """Procura tabelas relevantes nas bases de dados com base na pergunta. Usa isto PRIMEIRO para descobrir que tabelas existem. Retorna tabelas com o prefixo da BD (ex: [Operations] parcela)."""
            domain_results = agent_ref.domain_vectorstore.similarity_search(question, k=3)
            domain_names = [doc.metadata["domain_name"] for doc in domain_results]
            candidate_tables = []
            for domain_name in domain_names:
                group = DOMAIN_GROUPS[domain_name]
                db = group["db"]
                for (db_name, table_name) in agent_ref.table_schemas:
                    if db_name == db and any(table_name.startswith(p) for p in group["prefixes"]):
                        candidate_tables.append((db_name, table_name))
            if not candidate_tables:
                results = agent_ref.table_vectorstore.similarity_search(question, k=5)
                tables = [(doc.metadata["db_name"], doc.metadata["table_name"]) for doc in results]
            else:
                tables = []
                seen_for_search = set()
                for i, domain_name in enumerate(domain_names):
                    group = DOMAIN_GROUPS[domain_name]
                    db = group["db"]
                    domain_table_names = [
                        t for (db_name, t) in agent_ref.table_schemas
                        if db_name == db and any(t.startswith(p) for p in group["prefixes"])
                    ]
                    if not domain_table_names:
                        continue
                    k_domain = 3 if i == 0 else 2
                    domain_results = agent_ref.table_vectorstore.similarity_search(
                        question, k=k_domain * 3,
                        filter={"table_name": {"$in": domain_table_names}}
                    )
                    added = 0
                    for doc in domain_results:
                        if doc.metadata["db_name"] == db:
                            key = (doc.metadata["db_name"], doc.metadata["table_name"])
                            if key not in seen_for_search:
                                seen_for_search.add(key)
                                tables.append(key)
                                added += 1
                                if added >= k_domain:
                                    break
            MAX_TABLES = 8
            for db_name, table_name in list(tables[:MAX_TABLES]):
                for col, ref_table, ref_col in agent_ref.table_fks.get((db_name, table_name), []):
                    if (db_name, ref_table) not in tables:
                        if (db_name, ref_table) in agent_ref.table_schemas:
                            tables.append((db_name, ref_table))
            seen = set()
            unique_tables = []
            for item in tables:
                if item not in seen:
                    seen.add(item)
                    unique_tables.append(item)
            tables = unique_tables[:MAX_TABLES]
            formatted = [f"[{db}] {t}" for db, t in tables]
            dbs_used = sorted(set(db for db, _ in tables))
            output = f"Databases used: {', '.join(dbs_used)}\nTables found:\n" + "\n".join(formatted)
            return output
        @tool
        def get_table_schema(db_name: str, table_name: str) -> str:
            """Gets the full schema of a table (columns, types, foreign keys).
            Use BEFORE writing SQL to verify which columns exist.
            IMPORTANT: always pass db_name (e.g., 'Operations') and table_name (e.g., 'parcela')."""
            schema = agent_ref.table_schemas.get((db_name, table_name))
            if schema:
                desc_entry = TABLE_DESCRIPTIONS.get(db_name, {}).get(table_name)
                if desc_entry and desc_entry.get("descricao"):
                    schema = f"DESCRIPTION: {desc_entry['descricao']}\n\n{schema}"
                fks = agent_ref.table_fks.get((db_name, table_name), [])
                if len(fks) >= 2:
                    joins = [f"JOIN {ref_t} ON {ref_t}.{ref_col} = {table_name}.{col}" for col, ref_t, ref_col in fks]
                    schema += f"\n\nTHIS IS A JUNCTION TABLE. Typical JOIN pattern:\n  SELECT ... FROM {table_name}\n  " + "\n  ".join(joins)
                return schema
            matches = [(db, t) for (db, t) in agent_ref.table_schemas if t == table_name]
            if matches:
                other_dbs = [db for db, _ in matches]
                results = []
                for db, t in matches:
                    results.append(f"--- {db}.{t} ---\n{agent_ref.table_schemas[(db, t)]}")
                header = f"WARNING: Table '{table_name}' does NOT exist in '{db_name}'. Found in: {', '.join(other_dbs)}.\nUse db_name='{other_dbs[0]}' in the next queries.\n\n"
                return header + "\n\n".join(results)
            return f"Table '{table_name}' not found. Use search_tables to find available tables."
        @tool
        def run_sql(db_name: str, query: str) -> str:
            """Executa uma query SQL SELECT read-only na base de dados PostgreSQL especificada.
            Only SELECT is allowed. Returns the columns and rows of the result.
            IMPORTANT: always specify the db_name (e.g., 'Operations' or 'Plots')."""
            sql_upper = query.strip().upper()
            if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
                return "ERROR: Only SELECT queries are allowed."
            if db_name not in agent_ref.db_urls:
                available = ', '.join(agent_ref.db_urls.keys())
                return f"ERROR: Database '{db_name}' not found. Available databases: {available}"
            def _exec(db, sql):
                url = agent_ref.db_urls[db]
                conn = psycopg2.connect(url, options="-c statement_timeout=10000")
                conn.set_session(readonly=True, autocommit=True)
                cur = conn.cursor()
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                cur.close()
                conn.close()
                return columns, rows
            try:
                columns, rows = _exec(db_name, query)
                is_agg = any(fn in sql_upper for fn in ["COUNT(", "SUM(", "AVG("])
                agg_is_zero = is_agg and rows and len(rows) == 1 and rows[0][0] in (0, None)
                no_rows = not rows
                if agg_is_zero or no_rows:
                    table_match = re.search(r'\bFROM\s+(\w+)', query, re.IGNORECASE)
                    if table_match:
                        table_name = table_match.group(1)
                        other_dbs = [db for db in agent_ref.db_urls if db != db_name]
                        for other_db in other_dbs:
                            if (other_db, table_name) in agent_ref.table_schemas:
                                try:
                                    other_cols, other_rows = _exec(other_db, query)
                                    other_is_zero = is_agg and other_rows and len(other_rows) == 1 and other_rows[0][0] in (0, None)
                                    if other_rows and not other_is_zero:
                                        columns, rows = other_cols, other_rows
                                        db_name = other_db
                                        break
                                except Exception:
                                    pass
                if not rows:
                    return f"[{db_name}] Columns: {columns}\nResult: 0 rows (no data)"
                max_display = 50
                display_rows = rows[:max_display]
                header = "| " + " | ".join([str(c).replace("_", " ").title() for c in columns]) + " |"
                separator = "|" + "|".join(["---" for _ in columns]) + "|"
                table_rows = []
                for row in display_rows:
                    clean_vals = [sanitize(str(val)) if val is not None else "-" for val in row]
                    table_rows.append("| " + " | ".join(clean_vals) + " |")
                
                md_table = "\n".join([header, separator] + table_rows)
                
                text = f"**Results found ({len(rows)} records):**\n\n{md_table}"
                
                if len(rows) > max_display:
                    text += f"\n\n*Showing only the first {max_display} records.*"
                
                return text
            except Exception as e:
                return (
                    f"SQL ERROR [{db_name}]: {e}\n\n"
                    f"SYSTEM HINT: Stop guessing column names! "
                    f"You MUST use the 'get_table_schema' or 'sample_table_data' tool "
                    f"to check the exact table structure before trying run_sql again."
                )
        @tool
        def sample_table_data(db_name: str, table_name: str) -> str:
            """Gets 3 sample rows from a table to understand real column values
            (e.g., whether a column uses 'M/F' or 'Male/Female', has NULLs, date formats, etc.).
            Use when you're unsure about possible values of a column before writing the WHERE clause."""
            if db_name not in agent_ref.db_urls:
                available = ', '.join(agent_ref.db_urls.keys())
                return f"ERROR: Database '{db_name}' not found. Available databases: {available}"
            if (db_name, table_name) not in agent_ref.table_schemas:
                return f"Table '{table_name}' not found in '{db_name}'. Use search_tables first."
            url = agent_ref.db_urls[db_name]
            try:
                conn = psycopg2.connect(url, options="-c statement_timeout=5000")
                conn.set_session(readonly=True, autocommit=True)
                cur = conn.cursor()
                cur.execute(f"SELECT * FROM {table_name} LIMIT 3")
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                cur.close()
                conn.close()
                if not rows:
                    return f"Table '{table_name}' is empty."
                results = [sanitize(str(dict(zip(columns, row)))) for row in rows]
                return f"[{db_name}.{table_name}] Sample data ({len(rows)} rows):\n" + "\n".join(results)
            except Exception as e:
                return f"ERROR getting data from '{table_name}': {e}"
        @tool
        def find_join_path(db_name: str, from_table: str, to_table: str) -> str:
            """Finds the JOIN path between two tables using the Foreign Key graph.
            Use when you need to link two tables that don't have a direct FK (e.g., colheita → tipo_cultura).
            Returns ready-to-use SQL JOINs."""
            G = agent_ref.fk_graphs.get(db_name)
            if G is None:
                return f"ERROR: Database '{db_name}' not found."
            if from_table not in G:
                return f"ERROR: Table '{from_table}' does not exist in '{db_name}'."
            if to_table not in G:
                return f"ERROR: Table '{to_table}' does not exist in '{db_name}'."
            try:
                path = nx.shortest_path(G, from_table, to_table)
            except nx.NetworkXNoPath:
                return f"No FK path exists between '{from_table}' and '{to_table}' in '{db_name}'."
            if len(path) == 1:
                return f"'{from_table}' and '{to_table}' are the same table."
            joins = []
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                edge = G.edges[src, dst]
                joins.append(f"JOIN {dst} ON {dst}.{edge['ref_col']} = {src}.{edge['col']}")
            return (
                f"Path: {' -> '.join(path)}\n"
                f"SQL JOINs:\n" + "\n".join(joins)
            )
        tools_list = [search_tables, get_table_schema, sample_table_data, run_sql, find_join_path]
        self._run_sql = run_sql
        self._search_tables = search_tables
        self._get_table_schema = get_table_schema
        import core.config as cfg
        llm = ChatOllama(
            model=self.llm_model,
            temperature=cfg.SQL_TEMPERATURE,
            num_ctx=cfg.SQL_NUM_CTX,
            num_predict=2048,
        )
        db_names = ', '.join(self.db_urls.keys())
        _FALLBACK_PROMPT = f"""PostgreSQL SQL Agent. Respond in English. Databases: {db_names}.

PROCESS:
1. If the table is in KEY TABLES → go straight to run_sql (you already know the DB).
2. If you have schema from previous messages → use it, don't repeat get_table_schema.
3. Otherwise: search_tables → get_table_schema → run_sql.
4. If unsure about real values in a column, use sample_table_data before the WHERE.
5. Before joining two tables, confirm FK in the schema. No direct FK → find_join_path.
6. If 0 rows → check JOINs with find_join_path and try again. If still 0 → "No records found".

TOOL RULES:
- DATA AMNESIA: Completely ignore data (tables, numbers) you generated in previous messages. To answer any new question or filter (e.g., 'and in 2024?'), you MUST generate a new run_sql tool_call. History is only for understanding context, not as a data source.
- Emit ONLY tool_calls — never write text before you have data. The system ignores text intentions.
- After run_sql with valid result → write the final answer and STOP (no more tools).
- NEVER answer with data without run_sql. NEVER make up values.
- In follow-ups ("and in 2024?", "and by district?") you MUST execute a new run_sql. NEVER reuse numbers from previous answers or invent variations.
- If run_sql returns ERROR → fix the query (check schema) and try again. NEVER answer with data after an error.

FINAL ANSWER:
- It is STRICTLY FORBIDDEN to show SQL code, expose table names, or mention query errors in the final answer. The end user only wants to read business information.
- If the context injects an internal ID (e.g., "year": 42), that refers to `id_ano_agricola`. NEVER confuse it with the calendar year! To filter by calendar (e.g., 2025), use only `EXTRACT(YEAR) = 2025` and NEVER `EXTRACT(YEAR) = 42`.
- Simple language for farmers. No technical terms (tables, SQL, queries, DB).
- ≥3 columns → MANDATORY to use markdown table (| Name | Area | ... |). 1-2 columns → bullet points.
- Cabeçalhos legíveis: "Nome" (não "denominacao"), "Área (ha)" (não "area_total").
- Omite campos internos (gid, usercreate, userupdate, createdon, updatedon, estado).
- Números sem separadores de milhar (372265, não 372.265).
- O valor 'None' ou 'NULL' devolvido numa coluna significa apenas que esse campo está vazio na base de dados. O registo EXISTE e É VÁLIDO. NUNCA digas que "não encontraste resultados" se o SQL te devolver linhas com 'None'. Mostra a tabela na mesma.
- Mostra TODOS os resultados devolvidos pelo SQL. NUNCA trunces com "algumas delas" ou "...".
- Para listagens, seleciona 3-5 colunas relevantes (denominacao + área, altitude, etc.) e apresenta como tabela markdown:
| Nome | Área (ha) | Altitude (m) |
|------|-----------|--------------|
| ... | ... | ... |

        SQL RULES:
- Tabelas em SINGULAR: recurso, compra, pratica_cultural (nunca pluralizar).
- Especifica SEMPRE a BD — tabelas com nomes iguais existem em BDs diferentes.
- O campo para nomes é SEMPRE "denominacao" — NUNCA uses "nome" (não existe).
- JOIN com tabelas tipo_* para mostrar denominacao em vez de IDs numéricos.
- Filtros de texto: ILIKE '%valor%' (nunca = exacto).
- Estado: WHERE estado = 'A' para incluir apenas registos activos. Registos com estado NULL, 'Y' ou 'Z' são inactivos ou eliminados e devem ser excluídos.
- Agregações (COUNT/SUM/AVG): sem LIMIT.
- Usa LIMIT 50 por defeito para listagens genéricas de forma a proteger a BD. No entanto, se o utilizador usar palavras como 'todas', 'lista completa' ou 'tudo', NÃO USES LIMIT na query SQL.
- Monetário: SUM(preco * quantidade_comp).
- Sem backticks, sem prefixos de BD nos nomes de tabela.
- Ordenações (ORDER BY): Usa SEMPRE 'NULLS LAST' (ex: ORDER BY area_total DESC NULLS LAST). NUNCA deixes valores nulos aparecerem no topo de um TOP / LIMIT.
- NUNCA descrevas as tuas ações ou as ferramentas que vais usar. NUNCA digas "Vou executar a query X" ou "Vamos verificar a estrutura da tabela". Se precisares de usar uma ferramenta (como get_table_schema), INVOCA-A DIRETAMENTE e silenciosamente. O utilizador só quer ver os dados finais.

BD PREFERENCIAL POR TABELA:
- Operations: parcela, propriedade, concelho, distrito, freguesia, fase_parcelar, tipo_orgao_planta, fitofarmaco, fitofarmaco_uso_tc, fitofarmaco_empresa, tipo_rega, parcela_trega, parcela_tsolo, tipo_solo
- Plots: pratica_cultural, pratica_cultural_colheita, recurso, recurso_humano, recurso_equipamento, compra, compra_item, rota, rota_map, recurso_grupo_rec, orcamento
- Cellar: lote, cuba, rececao, vinho, vinho_tipo, vinho_categoria, variedade, barrica, lote_composicao, adega_lote, entrada_granel, adega, produto, stock_compra, consumivel, tanoeiro, fornecedor, equipamento_adg
- variedade: Cellar (castas/vindima) ou Operations (parcelas agrícolas)

TABELAS CHAVE (NUNCA adivinhes colunas, usa apenas as indicadas ou pede o schema. Se a coluna que queres não estiver aqui, OBRIGATORIAMENTE pede o schema primeiro):

[Operations]:
- parcela — gid, denominacao, area_total, id_propriedade, ano_plantacao. (ATENÇÃO: 'ano_plantacao' é do tipo NUMERIC. Usa diretamente ex: ano_plantacao = 2020. NUNCA uses funções de data como EXTRACT. NÃO existe id_entidade).
- propriedade — gid, denominacao, area_total, id_entidade.

[Plots]:
- pratica_cultural — A tabela central para operações de campo. Cruza com os recursos usados.
- recurso — O "Hub" central para equipamentos, recursos humanos e adubos/fitofármacos.
- compra_item — Itens comprados, quantidades e custos.

[Cellar]:
- lote — O conceito central de rastreabilidade do vinho/mosto.
- cuba — O recipiente físico (usa 'capacidade_total' para o tamanho da cuba, e não o volume de vinho).
- rececao — A entrada de uva. (ATENÇÃO: Usa 'peso_total' para a quantidade real de uva recebida em kg, NUNCA a capacidade da cuba).
- consumivel — O "Hub" central de inventário e produtos usados na adega (56 relações).

EXCEPÇÕES DE DOMÍNIO:
- Horas de práticas → pratica_cultural.quantidade_total_horas
- Área de parcelas → area_total ou area_util (hectares), nunca "area"
- Rega/Solo → JOIN via parcela_trega / parcela_tsolo (Operations)
- GPS de rotas → rota_map

ADEGA — VINHO:
- Cor: vinho.id_vinho_categoria = 'Tinto' (gid=nome, sem subquery)
- Tipo: vinho.id_vinho_tipo = 'Vinho DOP' (gid=nome)
- Volume (litros): SUM(lote_composicao.litragem_est)
- Peso uva (kg): rececao.peso_total
- Capacidade cuba ≠ conteúdo: cuba.capacidade_total=tamanho, cuba.capacidade_atual=conteúdo
Exemplo volume tinto: SELECT SUM(lc.litragem_est) FROM lote_composicao lc JOIN lote l ON l.gid=lc.id_lote JOIN vinho v ON v.gid=l.id_vinho WHERE v.id_vinho_categoria='Tinto';
"""
        # Guardar referências para recriar o agent quando o prompt mudar
        self._llm = llm
        self._tools_list = tools_list
        self._fallback_prompt = _FALLBACK_PROMPT
        self._current_prompt_version = None
        self._prompt_last_check = 0
        self._prompt_cache_ttl = 60  # verificar Langfuse a cada 1 min
        self._rebuild_agent_if_needed(force=True)

    def _rebuild_agent_if_needed(self, force=False):
        """Recarrega prompt do Langfuse se houver nova versão. Rebuild do agent se mudou."""
        now = time.time()
        if not force and (now - self._prompt_last_check) < self._prompt_cache_ttl:
            return
        self._prompt_last_check = now
        system_prompt = self._fallback_prompt
        new_version = "fallback"
        lf = cfg.get_langfuse_client()
        if lf:
            try:
                lf_prompt = lf.get_prompt("sql-agent-system-2", type="text")
                system_prompt = lf_prompt.prompt
                new_version = f"v{lf_prompt.version}"
            except Exception:
                pass
        if new_version != self._current_prompt_version:
            print(f"  [SQL Agent] Prompt updated: {self._current_prompt_version} -> {new_version}")
            self._current_prompt_version = new_version
            self.agent = create_agent(
                model=self._llm,
                tools=self._tools_list,
                system_prompt=system_prompt,
            )

    def _hybrid_table_search(self, question: str, k: int = 8) -> list:
        """BM25 + MMR hibrido 70/30 para o table vectorstore."""
        import numpy as np

        if self.bm25_index is None or not self.bm25_docs:
            mmr_results = self.table_vectorstore.max_marginal_relevance_search(
                question, k=k, fetch_k=k * 2
            )
            return mmr_results

        # BM25
        tokens = question.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokens)
        top_n = min(k * 3, len(self.bm25_docs))
        bm25_top_idx = bm25_scores.argsort()[::-1][:top_n]
        bm25_max = bm25_scores[bm25_top_idx[0]] if bm25_scores[bm25_top_idx[0]] > 0 else 1
        bm25_norm = {i: bm25_scores[i] / bm25_max for i in bm25_top_idx}

        # MMR
        mmr_results = self.table_vectorstore.max_marginal_relevance_search(
            question, k=k * 3, fetch_k=k * 5
        )
        mmr_keys = [(d.metadata["db_name"], d.metadata["table_name"]) for d in mmr_results]

        # Combinar
        scores: dict = {}
        for rank, idx in enumerate(bm25_top_idx):
            key = (self.bm25_docs[idx].metadata["db_name"], self.bm25_docs[idx].metadata["table_name"])
            scores[key] = scores.get(key, 0) + 0.7 * bm25_norm[idx]
        for rank, key in enumerate(mmr_keys):
            mmr_score = 1.0 - (rank / max(len(mmr_keys), 1))
            scores[key] = scores.get(key, 0) + 0.3 * mmr_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Reconstruir lista de documentos na ordem híbrida
        doc_map = {(d.metadata["db_name"], d.metadata["table_name"]): d for d in mmr_results}
        for idx in bm25_top_idx:
            d = self.bm25_docs[idx]
            key = (d.metadata["db_name"], d.metadata["table_name"])
            if key not in doc_map:
                doc_map[key] = d

        return [doc_map[key] for key, _ in ranked[:k] if key in doc_map]

    # Palavras-chave que indicam perguntas do Processo 2
    _P2_KEYWORDS = {
        "pratica", "praticas", "fitofarmaco", "fitofármacos", "fito", "adubo", "adubos",
        "empreiteiro", "empreiteiros", "jornaleiro", "jornaleira", "trabalhador",
        "equipamento", "trator", "máquina", "maquina", "recurso humano", "rh",
        "horas trabalhadas", "dias trabalhados", "custo producao", "poda",
        "aplicacao", "aplicação", "historico pratica", "histórico",
        "calendario", "calendário", "tipo de pratica", "tipo pratica",
    }

    # Palavras-chave Cellar que INIBEM o routing P2
    _CELLAR_KEYWORDS = {
        "vinificação", "vinificacoes", "vinificações", "cuba", "cubas",
        "lote", "lotes", "barrica", "barricas", "garrafa", "garrafas",
        "adega", "engarrafamento", "rotulagem", "embalamento",
        "vinho", "vinhos", "consumivel", "consumiveis", "consumível",
        "produto enológico", "enologico", "enológico",
        "fermentação", "fermentacao", "trasfega", "estagio",
        "perdas de vinho", "fases da adega", "comercializacao",
        "vinho tinto", "vinho branco", "tintos", "brancos",
        "tipo de vinho", "tipos de vinho", "variedades de vinho",
        "registo de amostra", "certificacao", "lote de vinho",
    }

    def _is_cellar_question(self, question: str) -> bool:
        """Detects questions about the Cellar module."""
        q = question.lower()
        return any(kw in q for kw in self._CELLAR_KEYWORDS)

    def _is_p2_question(self, question: str) -> bool:
        """Detects questions about agricultural practices.
        Does not activate for Cellar questions (different domain)."""
        if self._is_cellar_question(question):
            return False
        q = question.lower()
        return any(kw in q for kw in self._P2_KEYWORDS)

    def _get_few_shot_examples(self, question: str, k: int = 2) -> str:
        """Retrieves the k most similar few-shot examples to the question.
        For Cellar questions, filters only examples with 'Cellar' in pattern or SQL."""
        if not self._few_shot_vectors or not _FEW_SHOT_EXAMPLES:
            return ""
        import numpy as np
        is_adega = self._is_cellar_question(question)
        q_vec = self.embeddings.embed_query(question)
        q_arr = np.array(q_vec)
        scores = []
        for i, ex_vec in enumerate(self._few_shot_vectors):
            ex = _FEW_SHOT_EXAMPLES[i]
            is_cellar_example = "Cellar" in ex.get("pattern", "") or "v_cellar" in ex.get("sql", "")
            if is_adega and not is_cellar_example:
                continue
            if not is_adega and is_cellar_example:
                continue
            ex_arr = np.array(ex_vec)
            sim = float(np.dot(q_arr, ex_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(ex_arr) + 1e-9))
            scores.append((sim, i))
        top = sorted(scores, reverse=True)[:k]
        lines = []
        for _, idx in top:
            ex = _FEW_SHOT_EXAMPLES[idx]
            lines.append(f"Similar question: {ex['question']}\nCorrect SQL:\n{ex['sql']}")
        return "\n\n".join(lines)

    # Schema Pruning
    MAX_TABLE_SCHEMAS = 4

    def _prefetch_schemas(self, question: str) -> tuple[str, list[tuple[str, str]]]:
        """BM25+MMR híbrido para encontrar tabelas. Views injectadas no topo.
        Schema Pruning: máximo MAX_TABLE_SCHEMAS tabelas regulares para reduzir ruído.
        Para perguntas P2 força v_plot_resource_full ou mv_plot_resource_full."""
        search_result = self._search_tables.invoke({"question": question})
        lines = search_result.strip().split("\n")
        schemas = []
        tables_found = []
        seen = set()
        regular_tables_added = 0
        for line in lines:
            line = line.strip()
            if line.startswith("[") and "]" in line:
                db_name = line.split("]")[0].lstrip("[")
                table_name = line.split("]")[1].strip()
                key = (db_name, table_name)
                is_view = table_name.startswith(("v_", "view_", "mv_"))
                #limit regular tables, views always pass
                if not is_view and regular_tables_added >= self.MAX_TABLE_SCHEMAS:
                    continue
                if key not in seen:
                    seen.add(key)
                    schema = self._get_table_schema.invoke({"db_name": db_name, "table_name": table_name})
                    schemas.append(f"[{db_name}.{table_name}]\n{schema}")
                    tables_found.append(key)
                    if not is_view:
                        regular_tables_added += 1

        # force mv_plot_resource_full to the top
        if self._is_p2_question(question):
            # Prefer materialized view (faster) if it exists
            for view_name in ("mv_plot_resource_full", "v_plot_resource_full"):
                key = ("Plots", view_name)
                if key not in seen:
                    schema = self._get_table_schema.invoke({"db_name": "Plots", "table_name": view_name})
                    if "não encontrada" not in schema.lower() and "not found" not in schema.lower():
                        seen.add(key)
                        schemas.insert(0, f"[Plots.{view_name}]\n{schema}")
                        tables_found.insert(0, key)
                        break

        # View-first for Cellar
        if self._is_cellar_question(question):
            q_lower = question.lower()
            # Determine main view by context
            if any(kw in q_lower for kw in ("perda", "perdas")):
                cellar_primary = "v_cellar_losses"
            elif any(kw in q_lower for kw in ("quantidade de vinho", "litros", "fase de vinif", "tipo de armazenamento", "cubas barricas", "armazenamento")):
                cellar_primary = "v_cellar_volumes"
            elif any(kw in q_lower for kw in ("calendario", "calendário", "horas trabalhadas", "recurso humano", "rh")):
                cellar_primary = "v_cellar_hr_calendar"
            else:
                cellar_primary = "v_cellar_operations"

            key = ("Cellar", cellar_primary)
            if key not in seen:
                schema = self._get_table_schema.invoke({"db_name": "Cellar", "table_name": cellar_primary})
                if "não encontrada" not in schema.lower() and "not found" not in schema.lower():
                    seen.add(key)
                    schemas.insert(0, f"[Adega.{cellar_primary}]\n{schema}")
                    tables_found.insert(0, key)

            # For consumables, inject v_cellar_consumables
            if any(kw in q_lower for kw in ("consumiv", "garrafa", "produto enológico", "enologico", "anidrido")):
                key = ("Cellar", "v_cellar_consumables")
                if key not in seen:
                    schema = self._get_table_schema.invoke({"db_name": "Cellar", "table_name": "v_cellar_consumables"})
                    if "não encontrada" not in schema.lower():
                        seen.add(key)
                        schemas.insert(0, f"[Adega.v_cellar_consumables]\n{schema}")
                        tables_found.insert(0, key)

        # Inject views for Cellar questions, do not inject views from other DBs to avoid conflict
        is_adega = self._is_cellar_question(question)
        views_injected = 0
        for doc in self._hybrid_table_search(question, k=10):
            if views_injected >= 2:
                break
            t = doc.metadata.get("table_name", "")
            db = doc.metadata.get("db_name", "")
            if is_adega and db != "Cellar":
                continue 
            if not is_adega and db == "Cellar":
                continue
            if (t.startswith("v_") or t.startswith("view_") or t.startswith("mv_")) and (db, t) not in seen:
                seen.add((db, t))
                schema = self._get_table_schema.invoke({"db_name": db, "table_name": t})
                schemas.insert(0, f"[{db}.{t}]\n{schema}")
                tables_found.insert(0, (db, t))
                views_injected += 1

        if not schemas:
            return "", []
        text = "PRE-LOADED SCHEMAS (use ONLY these columns, do NOT guess):\n\n" + "\n\n".join(schemas)
        return text, tables_found

    # Pipeline

    _PIPELINE_SQL_PROMPT = """Generate ONE PostgreSQL SQL query to answer the user's question.
Use ONLY the columns listed in the schemas below. Do NOT invent columns.
Respond ONLY with the SQL query, no explanations, no markdown, no ```sql```.

{schemas}

{context}

Question: {question}

RULES:
- Status: WHERE estado = 'A' for active records. Views (v_* or view_*) do NOT have a status column — omit this filter.
- Dates: If the context has "ano": 42, that refers to column `id_ano_agricola = 42`. If the user asks for the calendar year (e.g., "2025"), use `EXTRACT(YEAR FROM data_inicio) = 2025`. NEVER use `EXTRACT(YEAR) = 42`, since year 42 doesn't exist in the calendar.
- Cellar Views (v_cellar_*): the `ano` column is a CALENDAR YEAR (2025, 2024, ...). To filter for 2025 use `WHERE ano = 2025`. NEVER replace `ano` with `id_ano_agricola` — they are completely different values.
- Tank in v_cellar_operations: to filter by tank use directly `WHERE cuba ILIKE '%name%'`. No additional JOIN needed.
- Batch in v_cellar_operations: to filter by batch use `WHERE lote ILIKE '%batch_name%'`. No JOIN to the batch table needed.
- Sorting: ORDER BY ... DESC NULLS LAST.
- JOINS IN VIEWS: If using a view (v_* or view_*), do NOT JOIN tipo_* tables to resolve names — text columns are already resolved in the view. You can JOIN other views or tables only when you need data the view doesn't have (e.g., JOIN v_field_base with parcela_tpenxerto for rootstocks). Always use view columns directly with WHERE column ILIKE '%value%' or WHERE column IS NOT NULL.
- RESOURCE NAMES: If the user asks about "pesticide", "fertilizer", "product" or "equipment", NEVER try to find tables with those names. Use ONLY the 'denominacao' column from v_plot_resource_full.
- Text: ILIKE '%value%'.
- Applied resources (fertilizers, pesticides): ALWAYS include `quantidade` and `custo_total` columns in the SELECT — the user needs to know how much was applied and how much it cost.
- HR calendars: use `mv_calendar_hr` (columns: recurso, mes, ano, total_horas, dias_trabalhados).
- Equipment calendars: use `mv_calendar_equipment` (columns: recurso, mes, ano, total_horas, dias_uso).
- mv_plot_resource_full: hours is `n_hora` (NOT `total_horas`), cost is `custo_total` (NOT `total_custo`). Always use `SUM(n_hora)` and `SUM(custo_total)` in this view. NEVER use `total_horas` or `total_custo` as column names in this table.
- LIMIT 50 by default. If the user asks for "all" or "everything", omit LIMIT.
- The name field is "denominacao", EXCEPT in Views where you must use STRICTLY the column names listed in the schema (e.g., "recurso", "cultura", "propriedade").
- JOIN with tipo_* tables to show denominacao instead of IDs — but only in normal tables.
- VIEWS (names starting with v_ or view_): text columns already contain resolved names. Do NOT JOIN other tables or other databases — use directly WHERE column ILIKE '%value%'. Use ONLY the columns listed in the view schema. If a view answers the question, use ONLY that view and its database — ignore other tables from schemas.
- No backticks, no database prefixes.
- Specify the database in comment format on the first line: -- DB: DatabaseName

EXAMPLES:

Costs and hours per property in mv_plot_resource_full — CORRECT (n_hora and custo_total, NOT total_horas or total_custo):
-- DB: Plots
SELECT propriedade, tipo_recurso, SUM(n_hora) AS total_horas, SUM(custo_total) AS total_custo
FROM mv_plot_resource_full
WHERE id_entidade = X AND id_ano_agricola = Y
GROUP BY propriedade, tipo_recurso
ORDER BY propriedade, tipo_recurso NULLS LAST
LIMIT 50

Practices/resources view — CORRECT query:
-- DB: Plots
SELECT tipo_pratica, SUM(n_hora) AS horas, SUM(custo_total) AS custos
FROM v_plot_resource_full
WHERE tipo_recurso = 'Pratica Cultural Fito' AND id_entidade = X AND EXTRACT(YEAR FROM data_inicio) = 2025
GROUP BY tipo_pratica
ORDER BY horas DESC NULLS LAST

Plots view — CORRECT query (rega and armacao are text columns):
-- DB: Operations
SELECT propriedade, parcela, area_total, rega
FROM v_field_base
WHERE id_entidade = X
ORDER BY area_total DESC NULLS LAST
LIMIT 50

Filter by specific property in view — CORRECT:
-- DB: Operations
SELECT parcela, armacao, embardamento, ano_plantacao
FROM v_field_base
WHERE propriedade ILIKE '%[PROPERTY_NAME]%' AND id_entidade = X
ORDER BY ano_plantacao DESC NULLS LAST
LIMIT 50

Normal table — CORRECT query:
-- DB: Operations
SELECT t.denominacao, r.denominacao AS referencia, t.area
FROM tabela t
JOIN tipo_referencia r ON r.gid = t.id_tipo_referencia
WHERE r.denominacao ILIKE '%value%' AND t.id_propriedade IN (SELECT gid FROM propriedade WHERE id_entidade = X)
ORDER BY t.area DESC NULLS LAST
LIMIT 50

SQL:"""

    _PIPELINE_FORMAT_PROMPT = """Format the data below as an answer for a farmer. Respond in English.
- It is STRICTLY FORBIDDEN to present SQL code, refer to table names, mention queries, or detail the database in your final answer. Only present the information for the user.
- Simple language, no technical terms.
- If the table is empty (0 results) or the query returns empty, respond "No records were found matching the criteria." and STOP. NEVER apologize or expose SQL.
- Readable headers: "Name" (not "denominacao"), "Area (ha)" (not "area_total").
- Omit internal fields (gid, usercreate, userupdate, createdon, updatedon).
- Show ALL results. NEVER truncate.
- If 0 results → "No records were found matching the criteria."

Question: {question}
Data:
{data}

Answer:"""

    def _validate_and_fix_sql(self, sql: str, db_name: str, schemas_text: str, llm) -> str:
        """Valida sintaxe (sqlglot) e colunas (schemas em memória). Corrige via LLM se necessário."""
        import sqlglot
        import sqlglot.expressions as exp

        try:
            parsed = sqlglot.parse_one(sql, dialect="postgres")
        except Exception as sg_err:
            print(f"[PIPELINE] Erro sintático (sqlglot): {sg_err}")
            fixed = llm.invoke(
                f"The SQL query has a syntax error. Fix it. Respond with ONLY the query.\n"
                f"Query: {sql}\nError: {sg_err}\nFixed SQL:"
            ).content.strip()
            fixed = re.sub(r'```sql\s*', '', fixed, flags=re.IGNORECASE)
            sql = re.sub(r'\s*```', '', fixed).strip()
            try:
                parsed = sqlglot.parse_one(sql, dialect="postgres")
            except Exception:
                return sql

        alias_map: dict[str, str] = {}
        for table_expr in parsed.find_all(exp.Table):
            tname = table_expr.name.lower()
            alias = (table_expr.alias or tname).lower()
            alias_map[alias] = tname

        known_cols: dict[str, set] = {}
        for (db, tbl), schema_text in self.table_schemas.items():
            if db == db_name:
                lines = schema_text.split("\n")
                cols = set()
                for line in lines:
                    line = line.strip()
                    if line.startswith("- ") or (line and "(" in line and not line.startswith("Base") and not line.startswith("Tabela") and not line.startswith("Chave")):
                        col = line.lstrip("- ").split("(")[0].strip().lower()
                        if col:
                            cols.add(col)
                known_cols[tbl.lower()] = cols

        bad_cols = []
        for col_expr in parsed.find_all(exp.Column):
            col_name = col_expr.name.lower()
            table_ref = (col_expr.table or "").lower()
            if table_ref and table_ref in alias_map:
                real_table = alias_map[table_ref]
                if real_table in known_cols and known_cols[real_table]:
                    if col_name not in known_cols[real_table] and col_name not in ("*",):
                        bad_cols.append(f"{table_ref}.{col_name} (table: {real_table})")

        if bad_cols:
            print(f"[PIPELINE] Invalid columns detected: {bad_cols}")

            valid_cols_hint = []
            for alias, real_table in alias_map.items():
                if real_table in known_cols and known_cols[real_table]:
                    cols_list = ", ".join(sorted(known_cols[real_table])[:30])
                    valid_cols_hint.append(f"  {alias} ({real_table}): {cols_list}")
            valid_hint = "\n".join(valid_cols_hint)

            fix_prompt = (
                f"The SQL query uses columns that DO NOT EXIST. Rewrite it from scratch using ONLY the columns listed below.\n"
                f"If a table is a view (v_* or view_*), use its text columns directly — do NOT JOIN tipo_* tables.\n\n"
                f"VALID COLUMNS BY TABLE:\n{valid_hint}\n\n"
                f"Invalid columns to remove: {', '.join(bad_cols)}\n\n"
                f"Original query:\n{sql}\n\n"
                f"Respond with ONLY the corrected SQL, no backticks, no explanations:"
            )
            fixed = llm.invoke(fix_prompt).content.strip()
            fixed = re.sub(r'```sql\s*', '', fixed, flags=re.IGNORECASE)
            fixed = re.sub(r'\s*```', '', fixed).strip()
            llm_fixed = fixed if fixed.upper().startswith(("SELECT", "WITH", "--")) else sql

            if llm_fixed.strip().rstrip(";") == sql.strip().rstrip(";"):
                view_tables = [(a, t) for a, t in alias_map.items()
                               if t.startswith("v_") or t.startswith("view_")]
                if view_tables:
                    view_alias, view_table = view_tables[0]
                    vcols = known_cols.get(view_table, set())
                    if vcols:
                        wanted = []
                        for m in re.finditer(r'(?:[\w.]+\s+AS\s+)(\w+)', sql, re.IGNORECASE):
                            c = m.group(1).lower()
                            if c in vcols:
                                wanted.append(c)
                        for m in re.finditer(rf'\b{re.escape(view_alias)}\.(\w+)\b', sql, re.IGNORECASE):
                            c = m.group(1).lower()
                            if c in vcols and c not in wanted:
                                wanted.append(c)
                        where_sql = ""
                        order_sql = ""
                        limit_sql = ""
                        try:
                            where_node = parsed.find(exp.Where)
                            if where_node:
                                where_sql = f"WHERE {where_node.this.sql(dialect='postgres')}"
                            order_node = parsed.find(exp.Order)
                            if order_node:
                                order_sql = order_node.sql(dialect='postgres')
                            limit_node = parsed.find(exp.Limit)
                            if limit_node:
                                limit_sql = limit_node.sql(dialect='postgres')
                        except Exception:
                            pass
                        if wanted:
                            cols_str = ", ".join(wanted)
                            prog_sql = f"SELECT {cols_str} FROM {view_table} {where_sql} {order_sql} {limit_sql}".strip()
                            print(f"[PIPELINE] SQL corrigido (programático): {prog_sql}")
                            return prog_sql

            if llm_fixed != sql:
                print(f"[PIPELINE] SQL corrigido (colunas): {llm_fixed}")
                return llm_fixed

        return sql

    def _pipeline_answer(self, question, chat_history=None, entidade_id=None, ano_agricola_id=None):
        """Pipeline SQL controlado: search → schema → gerar SQL → executar → formatar."""
        import core.config as cfg

        schemas_text, tables_found = self._prefetch_schemas(question)
        if not schemas_text:
            return None, None

        ctx_parts = []
        if entidade_id:
            ctx_parts.append(
                f"Entidade: id_entidade = {entidade_id}. "
                f"Filtra SEMPRE por esta entidade. Verifica no schema se a tabela tem id_entidade. "
                f"If it does NOT have it (e.g., parcela), use subquery: WHERE id_propriedade IN "
                f"(SELECT gid FROM propriedade WHERE id_entidade = {entidade_id})."
            )
        if ano_agricola_id:
            ctx_parts.append(
                f"ATTENTION: id_ano_agricola = {ano_agricola_id} is an internal ID, NOT a calendar year. "
                f"Use id_ano_agricola ONLY in tables/views that explicitly have that column. "
                f"To filter by year on date columns (data_inicio, data_registo, etc.), use the year the user mentioned in the question (e.g., 2025, 2024). "
                f"NEVER use {ano_agricola_id} in EXTRACT(YEAR ...) or in date conditions."
            )
        context = " ".join(ctx_parts) if ctx_parts else ""

        #inject similar examples
        few_shot_text = self._get_few_shot_examples(question, k=2)
        if few_shot_text:
            context = context + "\n\nCORRECT SQL EXAMPLES FOR SIMILAR PATTERNS:\n" + few_shot_text if context else "CORRECT SQL EXAMPLES FOR SIMILAR PATTERNS:\n" + few_shot_text

        llm = ChatOllama(model=self.llm_model, temperature=0, num_ctx=cfg.SQL_NUM_CTX, num_predict=1024)
        sql_prompt = self._PIPELINE_SQL_PROMPT.format(
            schemas=schemas_text, context=context, question=question
        )
        print(f"\n--- PIPELINE: A gerar SQL ---")
        raw_sql = llm.invoke(sql_prompt).content.strip()
        raw_sql = re.sub(r'```sql\s*', '', raw_sql, flags=re.IGNORECASE)
        raw_sql = re.sub(r'\s*```', '', raw_sql)

        if "-- BD:" in raw_sql:
            raw_sql = raw_sql[raw_sql.find("-- BD:"):]
        raw_sql = raw_sql.strip()

        db_name = None
        if raw_sql.startswith("-- BD:"):
            db_line, _, raw_sql = raw_sql.partition("\n")
            db_name = db_line.replace("-- BD:", "").strip()
            raw_sql = raw_sql.strip()
        if not db_name and tables_found:
            view_dbs = [db for db, t in tables_found if t.startswith("v_") or t.startswith("view_")]
            db_name = view_dbs[0] if view_dbs else tables_found[0][0]

        if not raw_sql.upper().startswith(("SELECT", "WITH")):
            print(f"[PIPELINE] SQL invalid: {raw_sql[:100]}")
            return None, None

        if db_name == "Plots":
            mv_key = ("Plots", "mv_plot_resource_full")
            if mv_key in self.table_schemas:
                raw_sql_new = re.sub(
                    r'\bv_plot_resource_full\b',
                    'mv_plot_resource_full',
                    raw_sql
                )
                if raw_sql_new != raw_sql:
                    print(f"[PIPELINE] Replaced v_plot_resource_full → mv_plot_resource_full")
                    raw_sql = raw_sql_new

        print(f"[PIPELINE] BD: {db_name}")
        print(f"[PIPELINE] SQL: {raw_sql}")

        raw_sql = self._validate_and_fix_sql(raw_sql, db_name, schemas_text, llm)

        result = None
        sql_used = raw_sql
        for attempt in range(2):
            result = self._run_sql.invoke({"db_name": db_name, "query": sql_used})
            if "SQL ERROR" not in result:
                break
            if attempt == 0:
                print(f"[PIPELINE] Error on attempt {attempt+1}, fixing...")
                fix_prompt = (
                    f"The following SQL query returned an error. Fix it using ONLY the columns from the schemas.\n"
                    f"Respond with ONLY the corrected query, no explanations.\n\n"
                    f"Original query:\n{sql_used}\n\nError:\n{result}\n\n{schemas_text}\n\nCorrected SQL:"
                )
                fixed = llm.invoke(fix_prompt).content.strip()
                fixed = re.sub(r'```sql\s*', '', fixed, flags=re.IGNORECASE)
                fixed = re.sub(r'\s*```', '', fixed)
                sql_used = fixed.strip()
                print(f"[PIPELINE] SQL corrected: {sql_used}")

        if not result or "ERRO SQL" in result:
            print(f"[PIPELINE] Failed after 2 attempts, falling back")
            return None, None

        print(f"[PIPELINE] Formatting response with Python...")
        
        if not result or "0 linhas" in result or "ERRO" in result:
            response = "No records were found matching the criteria."
        else:
            response = result

        print(f"[PIPELINE] Success!")
        return response, sql_used

    # Metrics counters
    _metrics = {"pipeline_ok": 0, "pipeline_fail": 0, "fallback_ok": 0, "fallback_fail": 0, "cache_hit": 0}

    @classmethod
    def print_metrics(cls):
        m = cls._metrics
        total = m["pipeline_ok"] + m["pipeline_fail"] + m["fallback_ok"] + m["fallback_fail"]
        if total == 0:
            return
        pipeline_rate = round(m["pipeline_ok"] / max(m["pipeline_ok"] + m["pipeline_fail"], 1) * 100)
        print(f"\n[SQL METRICS] Total:{total} | Pipeline:{m['pipeline_ok']}✓ {m['pipeline_fail']}✗ ({pipeline_rate}%) | Fallback:{m['fallback_ok']}✓ {m['fallback_fail']}✗ | Cache:{m['cache_hit']}")

    def answer(self, question, chat_history=None, entidade_id=None, ano_agricola_id=None):
        """Controlled pipeline as primary; ReAct agent as fallback."""
        self._rebuild_agent_if_needed()
        start = time.time()
        cache_key = unicodedata.normalize("NFKD", question.strip().lower())
        if entidade_id:
            cache_key += f"_ent{entidade_id}"
        if ano_agricola_id:
            cache_key += f"_ano{ano_agricola_id}"
        if not chat_history:
            cached = self._cache.get(cache_key)
            if cached:
                answer_c, sql_c, ts = cached
                if time.time() - ts < self.CACHE_TTL:
                    print(f"[CACHE HIT] '{question}' ({int(time.time() - ts)}s ago)")
                    SQLAgentTools._metrics["cache_hit"] += 1
                    return answer_c, sql_c, time.time() - start

        try:
            resp, sql = self._pipeline_answer(question, chat_history, entidade_id, ano_agricola_id)
            if resp is not None:
                elapsed = time.time() - start
                SQLAgentTools._metrics["pipeline_ok"] += 1
                self.print_metrics()
                if not chat_history and sql:
                    self._cache[cache_key] = (resp, sql, time.time())
                return resp, sql, elapsed
            SQLAgentTools._metrics["pipeline_fail"] += 1
        except Exception as e:
            print(f"[PIPELINE ERROR]: {e} — using ReAct fallback")
            SQLAgentTools._metrics["pipeline_fail"] += 1

        print("[FALLBACK]: Using ReAct agent")
        prefetched, _ = self._prefetch_schemas(question)
        print(f"\n--- PREFETCH (FALLBACK) ---\n{prefetched}\n----------------\n")

        messages_hist = []
        if chat_history:
            messages_hist.extend(chat_history)

        ctx_parts = []
        if prefetched:
            ctx_parts.append(prefetched)
        if entidade_id:
            ctx_parts.append(
                f"Entity: id_entidade = {entidade_id}. "
                f"ALWAYS filter by this entity. Check in the schema if the table has id_entidade. "
                f"If it does NOT have it (e.g., parcela), use subquery: WHERE id_propriedade IN (SELECT gid FROM propriedade WHERE id_entidade = {entidade_id})."
            )
        if ano_agricola_id:
            ctx_parts.append(
                f"ATTENTION: id_ano_agricola = {ano_agricola_id} is an internal ID, NOT a calendar year. "
                f"Use id_ano_agricola ONLY in tables/views that explicitly have that column (e.g., pratica_cultural, rota, lote). "
                f"Tables without this column (e.g., parcela, propriedade, cuba) do NOT filter by year. "
                f"To filter by year on date columns, use the year the user mentioned in the question. "
                f"NEVER use {ano_agricola_id} in EXTRACT(YEAR ...) or in date conditions."
            )
        if ctx_parts:
            ctx = "USER CONTEXT: " + " ".join(ctx_parts)
            messages_hist.append(HumanMessage(content=f"{ctx}\n\nQuestion: {question}"))
        else:
            messages_hist.append(HumanMessage(content=question))
        from langgraph.errors import GraphRecursionError
        _lf_callbacks = [self._langfuse_handler] if self._langfuse_handler else []
        for attempt in range(2):
            try:
                result = self.agent.invoke(
                    {"messages": messages_hist},
                    {"recursion_limit": 30, "callbacks": _lf_callbacks},
                )
            except GraphRecursionError:
                elapsed = time.time() - start
                return "The question is too complex to be processed automatically. Try rephrasing it more directly or break it into simpler parts.", "", elapsed
            messages = result["messages"]
            final_response = messages[-1].content if messages else "No response."
            last_msg_has_tool = hasattr(messages[-1], 'tool_calls') and len(messages[-1].tool_calls) > 0
            sql_used = ""
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get('name') == 'run_sql':
                            sql_used = tc.get('args', {}).get('query', '')
            if not last_msg_has_tool:
                lower_resp = final_response.lower()
                if any(w in lower_resp for w in ["i will now", "write a query", "check the relationship", "we first need", "write the query", "try again", "i will check", "i will open", "i will execute", "an error occurred", "it seems there was"]):
                    print("\n[PYTHON RETRY]: LLM stopped prematurely rambling. Forcing tool_call...")
                    messages_hist = messages + [HumanMessage(content="INTERNAL ERROR: You responded with text of your plan but forgot to emit the actual tool (tool_call JSON). Emit ONLY the tool now without apologizing.")]
                    continue
            break
        print("\n--- LLM THOUGHT ---")
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                print(f"[{msg.__class__.__name__}]: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[Executing Tool]: {tc['name']} with args {tc['args']}")
        print("-------------------------\n")
        sql_used = ""
        for msg in messages:
            if hasattr(msg, 'tool_calls'):
                for tc in msg.tool_calls:
                    if tc.get('name') == 'run_sql':
                        sql_used = tc.get('args', {}).get('query', '')
        import re
        if not sql_used and "```sql" in final_response.lower():
            sql_match = re.search(r'```sql\s+(.*?)\s+```', final_response, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_used = sql_match.group(1).strip()
                db_name = "Operations" if "monitorizacao" in final_response.lower() else "Plots"
                print(f"\n[FALLBACK SQL DETECTED]: Running '{sql_used}' on DB '{db_name}'...")
                fallback_result = self._run_sql.invoke({"db_name": db_name, "query": sql_used})
                final_response = f"**Data calculated automatically by Fallback:**\n\n{fallback_result}\n\n*Note: The model behaved unexpectedly, but the response was successfully handled.*"
        final_response = re.sub(r'```sql[\s\S]*?```', '', final_response, flags=re.IGNORECASE)
        final_response = re.sub(r'```[\s\S]*?```', '', final_response)
        final_response = re.sub(r'(?im)^\s*(SELECT|WITH)\s+.*?(FROM|JOIN)\b.*$', '', final_response)
        final_response = re.sub(r'(?im)^\s*(FROM|JOIN|WHERE|GROUP BY|ORDER BY|LIMIT)\s+.*$', '', final_response)
        final_response = re.sub(r'ERRO SQL\s*\[.*?\]:.*', '', final_response, flags=re.IGNORECASE)
        final_response = re.sub(r'DICA DO SISTEMA:.*', '', final_response, flags=re.IGNORECASE)
        final_response = re.sub(
            r'(I will now|To determine|We need to first|Sorry for the confusion|'
            r'it seems there was an error|table.*not found|I.*will check|I.*will open|'
            r'After several attempts|I will fix|Here is the query|I will execute|'
            r'I will use the tool|Let us execute|is not a numeric ID|'
            r'The query results|path of JOINs|The query returned)[^\n]*',
            '', final_response, flags=re.IGNORECASE
        ).strip()
        final_response = re.sub(r'`[a-z_]+`', '', final_response).strip()
        final_response = re.sub(r'[ \t]{2,}', ' ', final_response).strip()
        final_response = re.sub(r'\n{3,}', '\n\n', final_response).strip()
        if not final_response:
            final_response = "Sorry, your request seems to require joining too many data sources at once. Could you try rephrasing it more directly or break it into smaller parts?"
        if not chat_history and sql_used:
            self._cache[cache_key] = (final_response, sql_used, time.time())
        elapsed = time.time() - start
        if sql_used:
            SQLAgentTools._metrics["fallback_ok"] += 1
        else:
            SQLAgentTools._metrics["fallback_fail"] += 1
        self.print_metrics()
        return final_response, sql_used, elapsed
