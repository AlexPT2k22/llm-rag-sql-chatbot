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
        "desc": "Dados geográficos e administrativos: distritos, concelhos, freguesias, perfil solo geológico, ano agrícola, entidades, riscos",
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
        self.bm25_index = BM25Okapi([doc.page_content.lower().split() for doc in table_docs])
        print(f"  BM25 indexado: {len(table_docs)} documentos.")

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
            output = f"BDs envolvidas: {', '.join(dbs_used)}\nTabelas encontradas:\n" + "\n".join(formatted)
            return output
        @tool
        def get_table_schema(db_name: str, table_name: str) -> str:
            """Obtém o schema completo de uma tabela (colunas, tipos, chaves estrangeiras).
            Usa ANTES de escrever SQL para verificar que colunas existem.
            IMPORTANTE: passa sempre o db_name (ex: 'Operations') e o table_name (ex: 'parcela')."""
            schema = agent_ref.table_schemas.get((db_name, table_name))
            if schema:
                desc_entry = TABLE_DESCRIPTIONS.get(db_name, {}).get(table_name)
                if desc_entry and desc_entry.get("descricao"):
                    schema = f"DESCRIÇÃO: {desc_entry['descricao']}\n\n{schema}"
                fks = agent_ref.table_fks.get((db_name, table_name), [])
                if len(fks) >= 2:
                    joins = [f"JOIN {ref_t} ON {ref_t}.{ref_col} = {table_name}.{col}" for col, ref_t, ref_col in fks]
                    schema += f"\n\nESTA É UMA TABELA DE ASSOCIAÇÃO. Padrão de JOIN típico:\n  SELECT ... FROM {table_name}\n  " + "\n  ".join(joins)
                return schema
            matches = [(db, t) for (db, t) in agent_ref.table_schemas if t == table_name]
            if matches:
                other_dbs = [db for db, _ in matches]
                results = []
                for db, t in matches:
                    results.append(f"--- {db}.{t} ---\n{agent_ref.table_schemas[(db, t)]}")
                header = f"AVISO: Tabela '{table_name}' NÃO existe em '{db_name}'. Encontrada em: {', '.join(other_dbs)}.\nUsa db_name='{other_dbs[0]}' nas próximas queries.\n\n"
                return header + "\n\n".join(results)
            return f"Tabela '{table_name}' não encontrada. Usa search_tables para encontrar tabelas disponíveis."
        @tool
        def run_sql(db_name: str, query: str) -> str:
            """Executa uma query SQL SELECT read-only na base de dados PostgreSQL especificada.
            Apenas SELECT é permitido. Retorna as colunas e linhas do resultado.
            IMPORTANTE: especifica sempre o db_name (ex: 'Operations' ou 'Plots')."""
            sql_upper = query.strip().upper()
            if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
                return "ERRO: Apenas queries SELECT são permitidas."
            if db_name not in agent_ref.db_urls:
                available = ', '.join(agent_ref.db_urls.keys())
                return f"ERRO: BD '{db_name}' não encontrada. BDs disponíveis: {available}"
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
                    return f"[{db_name}] Colunas: {columns}\nResultado: 0 linhas (sem dados)"
                max_display = 50
                display_rows = rows[:max_display]
                header = "| " + " | ".join([str(c).replace("_", " ").title() for c in columns]) + " |"
                separator = "|" + "|".join(["---" for _ in columns]) + "|"
                table_rows = []
                for row in display_rows:
                    clean_vals = [sanitize(str(val)) if val is not None else "-" for val in row]
                    table_rows.append("| " + " | ".join(clean_vals) + " |")
                
                md_table = "\n".join([header, separator] + table_rows)
                
                text = f"**Resultados encontrados ({len(rows)} registos):**\n\n{md_table}"
                
                if len(rows) > max_display:
                    text += f"\n\n*A mostrar apenas os primeiros {max_display} registos.*"
                
                return text
            except Exception as e:
                return (
                    f"ERRO SQL [{db_name}]: {e}\n\n"
                    f"DICA DO SISTEMA: Pára de adivinhar nomes de colunas! "
                    f"És OBRIGADO a usar a ferramenta 'get_table_schema' ou 'sample_table_data' "
                    f"para verificar a estrutura exata da tabela antes de tentares o run_sql novamente."
                )
        @tool
        def sample_table_data(db_name: str, table_name: str) -> str:
            """Obtém 3 linhas de exemplo de uma tabela para perceber os valores reais das colunas
            (ex: se uma coluna usa 'M/F' ou 'Masculino/Feminino', se tem NULLs, formatos de data, etc.).
            Usa quando não tens a certeza sobre os valores possíveis de uma coluna antes de escrever o WHERE."""
            if db_name not in agent_ref.db_urls:
                available = ', '.join(agent_ref.db_urls.keys())
                return f"ERRO: BD '{db_name}' não encontrada. BDs disponíveis: {available}"
            if (db_name, table_name) not in agent_ref.table_schemas:
                return f"Tabela '{table_name}' não encontrada em '{db_name}'. Usa search_tables primeiro."
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
                    return f"Tabela '{table_name}' está vazia."
                results = [sanitize(str(dict(zip(columns, row)))) for row in rows]
                return f"[{db_name}.{table_name}] Exemplo de dados ({len(rows)} linhas):\n" + "\n".join(results)
            except Exception as e:
                return f"ERRO ao obter dados de '{table_name}': {e}"
        @tool
        def find_join_path(db_name: str, from_table: str, to_table: str) -> str:
            """Encontra o caminho de JOINs entre duas tabelas usando o grafo de Foreign Keys.
            Usa quando precisas de ligar duas tabelas que não têm FK directa (ex: colheita → tipo_cultura).
            Retorna os JOINs SQL prontos a usar."""
            G = agent_ref.fk_graphs.get(db_name)
            if G is None:
                return f"ERRO: BD '{db_name}' não encontrada."
            if from_table not in G:
                return f"ERRO: Tabela '{from_table}' não existe em '{db_name}'."
            if to_table not in G:
                return f"ERRO: Tabela '{to_table}' não existe em '{db_name}'."
            try:
                path = nx.shortest_path(G, from_table, to_table)
            except nx.NetworkXNoPath:
                return f"Não existe caminho de FK entre '{from_table}' e '{to_table}' em '{db_name}'."
            if len(path) == 1:
                return f"'{from_table}' e '{to_table}' são a mesma tabela."
            joins = []
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                edge = G.edges[src, dst]
                joins.append(f"JOIN {dst} ON {dst}.{edge['ref_col']} = {src}.{edge['col']}")
            return (
                f"Caminho: {' -> '.join(path)}\n"
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
        _FALLBACK_PROMPT = f"""Agente SQL PostgreSQL. Responde em PT-PT (formas europeias). BDs: {db_names}.

PROCESSO:
1. Se a tabela está em TABELAS CHAVE → vai direto a run_sql (já sabes a BD).
2. Se tens schema de mensagens anteriores → usa-o, não repitas get_table_schema.
3. Caso contrário: search_tables → get_table_schema → run_sql.
4. Em caso de dúvida sobre valores reais duma coluna, usa sample_table_data antes do WHERE.
5. Antes de JOIN entre duas tabelas, confirma FK no schema. Sem FK directa → find_join_path.
6. Se 0 linhas → verifica JOINs com find_join_path e tenta de novo. Se continuar 0 → "Não foram encontrados registos".

REGRAS DE TOOL:
- AMNÉSIA DE DADOS: Ignora completamente os dados (tabelas, números) gerados por ti em mensagens anteriores. Para responder a qualquer nova pergunta ou filtro (ex: 'e em 2024?'), tens OBRIGATORIAMENTE de gerar uma nova tool_call run_sql. O histórico serve apenas para entender o contexto, não como fonte de dados.
- Emite APENAS tool_calls — nunca escrevas texto antes de teres dados. O sistema ignora intenções textuais.
- Após run_sql com resultado válido → escreve a resposta final e PÁRA (sem mais tools).
- NUNCA respondas com dados sem run_sql. NUNCA inventes valores.
- Em follow-ups ("e em 2024?", "e por distrito?") DEVES executar nova run_sql. NUNCA reutilizes números de respostas anteriores nem inventes variações.
- Se run_sql devolver ERRO → corrige a query (verifica schema) e tenta de novo. NUNCA respondas com dados após um erro.

RESPOSTA FINAL:
- É ESTRITAMENTE PROIBIDO mostrar o código SQL, expor o nome das tabelas, ou mencionar erros de query na resposta final. O utilizador final só quer ler a informação de negócio.
- Se o contexto injetar um ID interno (ex: "ano": 42), isso refere-se ao `id_ano_agricola`. NUNCA o confundas com o ano de calendário! Para filtrar pelo calendário (ex: 2025), usa apenas `EXTRACT(YEAR) = 2025` e NUNCA `EXTRACT(YEAR) = 42`.
- Linguagem simples para agricultores. Sem termos técnicos (tabelas, SQL, queries, BD).
- ≥3 colunas → OBRIGATÓRIO usar tabela markdown (| Nome | Área | ... |). 1-2 colunas → bullet points.
- Cabeçalhos legíveis: "Nome" (não "denominacao"), "Área (ha)" (não "area_total").
- Omite campos internos (gid, usercreate, userupdate, createdon, updatedon, estado).
- Números sem separadores de milhar (372265, não 372.265).
- O valor 'None' ou 'NULL' devolvido numa coluna significa apenas que esse campo está vazio na base de dados. O registo EXISTE e É VÁLIDO. NUNCA digas que "não encontraste resultados" se o SQL te devolver linhas com 'None'. Mostra a tabela na mesma.
- Mostra TODOS os resultados devolvidos pelo SQL. NUNCA trunces com "algumas delas" ou "...".
- Para listagens, seleciona 3-5 colunas relevantes (denominacao + área, altitude, etc.) e apresenta como tabela markdown:
| Nome | Área (ha) | Altitude (m) |
|------|-----------|--------------|
| ... | ... | ... |

REGRAS SQL:
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
            print(f"  [SQL Agent] Prompt atualizado: {self._current_prompt_version} -> {new_version}")
            self._current_prompt_version = new_version
            self.agent = create_agent(
                model=self._llm,
                tools=self._tools_list,
                system_prompt=system_prompt,
            )

    def _hybrid_table_search(self, question: str, k: int = 8) -> list:
        """BM25 + MMR híbrido 70/30 para o table vectorstore."""
        import numpy as np

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
        """Detecta perguntas sobre o modulo Cellar."""
        q = question.lower()
        return any(kw in q for kw in self._CELLAR_KEYWORDS)

    def _is_p2_question(self, question: str) -> bool:
        """Detecta perguntas sobre práticas culturais agrícolas.
        Não activa para perguntas do Cellar (dominio diferente)."""
        if self._is_cellar_question(question):
            return False
        q = question.lower()
        return any(kw in q for kw in self._P2_KEYWORDS)

    def _get_few_shot_examples(self, question: str, k: int = 2) -> str:
        """Recupera os k exemplos few-shot mais similares à pergunta.
        Para perguntas Adega, filtra apenas exemplos com 'Cellar' no pattern ou SQL."""
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
            lines.append(f"Pergunta similar: {ex['question']}\nSQL correcto:\n{ex['sql']}")
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
                #limitar tabelas regulares, views passam sempre
                if not is_view and regular_tables_added >= self.MAX_TABLE_SCHEMAS:
                    continue
                if key not in seen:
                    seen.add(key)
                    schema = self._get_table_schema.invoke({"db_name": db_name, "table_name": table_name})
                    schemas.append(f"[{db_name}.{table_name}]\n{schema}")
                    tables_found.append(key)
                    if not is_view:
                        regular_tables_added += 1

        # forçar mv_plot_resource_full no topo
        if self._is_p2_question(question):
            # Preferir materialized view (mais rápida) se existir
            for view_name in ("mv_plot_resource_full", "v_plot_resource_full"):
                key = ("Plots", view_name)
                if key not in seen:
                    schema = self._get_table_schema.invoke({"db_name": "Plots", "table_name": view_name})
                    if "não encontrada" not in schema.lower() and "not found" not in schema.lower():
                        seen.add(key)
                        schemas.insert(0, f"[Plots.{view_name}]\n{schema}")
                        tables_found.insert(0, key)
                        break

        # View-first para Cellar
        if self._is_cellar_question(question):
            q_lower = question.lower()
            # Determinar view principal pelo contexto
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

            # Para consumíveis, injectar v_cellar_consumables
            if any(kw in q_lower for kw in ("consumiv", "garrafa", "produto enológico", "enologico", "anidrido")):
                key = ("Cellar", "v_cellar_consumables")
                if key not in seen:
                    schema = self._get_table_schema.invoke({"db_name": "Cellar", "table_name": "v_cellar_consumables"})
                    if "não encontrada" not in schema.lower():
                        seen.add(key)
                        schemas.insert(0, f"[Adega.v_cellar_consumables]\n{schema}")
                        tables_found.insert(0, key)

        # Injectar views para perguntas Cellar, nao injectar views de outras BDs para evitar conflito
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
        text = "SCHEMAS PRÉ-CARREGADOS (usa APENAS estas colunas, NÃO adivinhes):\n\n" + "\n\n".join(schemas)
        return text, tables_found

    # Pipeline

    _PIPELINE_SQL_PROMPT = """Gera UMA query SQL PostgreSQL para responder à pergunta do utilizador.
Usa APENAS as colunas listadas nos schemas abaixo. NÃO inventes colunas.
Responde APENAS com a query SQL, sem explicações, sem markdown, sem ```sql```.

{schemas}

{context}

Pergunta: {question}

REGRAS:
- Estado: WHERE estado = 'A' para registos activos. Views (v_* ou view_*) NÃO têm coluna estado — omite este filtro.
- Datas: Se o contexto tiver "ano": 42, isso refere-se à coluna `id_ano_agricola = 42`. Se o utilizador pedir pelo ano do calendário (ex: "2025"), podes usar `EXTRACT(YEAR FROM data_inicio) = 2025`. NUNCA FAÇAS `EXTRACT(YEAR) = 42`, pois o ano 42 não existe no calendário.
- Views Cellar (v_cellar_*): a coluna `ano` é um ANO DE CALENDÁRIO (2025, 2024, ...). Para filtrar por 2025 usa `WHERE ano = 2025`. NUNCA substituas `ano` por `id_ano_agricola` — são valores completamente diferentes.
- Cuba em v_cellar_operations: para filtrar por cuba usa directamente `WHERE cuba ILIKE '%nome%'`. NÃO precisas de JOIN adicional.
- Lote em v_cellar_operations: para filtrar por lote usa `WHERE lote ILIKE '%nome_lote%'`. NÃO precisas de JOIN à tabela lote.
- Ordenações: ORDER BY ... DESC NULLS LAST.
- JOINS EM VIEWS: Se usares uma view (v_* ou view_*), NÃO faças JOIN a tabelas tipo_* para resolver nomes — as colunas de texto já estão resolvidas na view. Podes fazer JOIN a outras views ou tabelas apenas quando precisas de dados que a view não tem (ex: JOIN v_field_base com parcela_tpenxerto para porta-enxertos). Usa SEMPRE as colunas da view directamente com WHERE coluna ILIKE '%valor%' ou WHERE coluna IS NOT NULL.
- NOMES DOS RECURSOS: Se o utilizador perguntar por "fitofármaco", "adubo", "produto" ou "equipamento", NUNCA tentes ir buscar tabelas com esses nomes. Usa ÚNICA E EXCLUSIVAMENTE a coluna 'denominacao' da view v_plot_resource_full.
- Texto: ILIKE '%valor%'.
- Recursos aplicados (adubos, fitofármacos): inclui SEMPRE as colunas `quantidade` e `custo_total` no SELECT — o utilizador precisa saber quanto foi aplicado e quanto custou.
- Calendários de RH: usa `mv_calendar_hr` (colunas: recurso, mes, ano, total_horas, dias_trabalhados).
- Calendários de Equipamentos: usa `mv_calendar_equipment` (colunas: recurso, mes, ano, total_horas, dias_uso).
- mv_plot_resource_full: horas é `n_hora` (NÃO `total_horas`), custo é `custo_total` (NÃO `total_custo`). Usa SEMPRE `SUM(n_hora)` e `SUM(custo_total)` nesta view. NUNCA uses `total_horas` nem `total_custo` como nomes de coluna nesta tabela.
- LIMIT 50 por defeito. Se o utilizador pedir "todas" ou "tudo", sem LIMIT.
- O campo de nomes é "denominacao", EXCETO nas Views, onde deves usar ESTRITAMENTE o nome das colunas listadas no schema (ex: "recurso", "cultura", "propriedade").
- JOIN com tabelas tipo_* para mostrar denominacao em vez de IDs — MAS só em tabelas normais.
- VIEWS (nomes começando com v_ ou view_): as colunas de texto já contêm os nomes resolvidos. NÃO faças JOIN a outras tabelas nem a outras BDs — usa directamente WHERE coluna ILIKE '%valor%'. Usa APENAS as colunas listadas no schema da view. Se uma view responde à pergunta, usa SÓ essa view e a sua BD — ignora as outras tabelas dos schemas.
- Sem backticks, sem prefixos de BD.
- Especifica a BD no formato de comentário na primeira linha: -- BD: NomeDaBD

EXEMPLOS:

Custos e horas por propriedade em mv_plot_resource_full — CORRECTO (n_hora e custo_total, NAO total_horas nem total_custo):
-- BD: Plots
SELECT propriedade, tipo_recurso, SUM(n_hora) AS total_horas, SUM(custo_total) AS total_custo
FROM mv_plot_resource_full
WHERE id_entidade = X AND id_ano_agricola = Y
GROUP BY propriedade, tipo_recurso
ORDER BY propriedade, tipo_recurso NULLS LAST
LIMIT 50

View de práticas/recursos — query CORRECTA:
-- BD: Plots
SELECT tipo_pratica, SUM(n_hora) AS horas, SUM(custo_total) AS custos
FROM v_plot_resource_full
WHERE tipo_recurso = 'Pratica Cultural Fito' AND id_entidade = X AND EXTRACT(YEAR FROM data_inicio) = 2025
GROUP BY tipo_pratica
ORDER BY horas DESC NULLS LAST

View de parcelas — query CORRECTA (rega e armacao são colunas de texto):
-- BD: Operations
SELECT propriedade, parcela, area_total, rega
FROM v_field_base
WHERE id_entidade = X
ORDER BY area_total DESC NULLS LAST
LIMIT 50

Filtrar por propriedade específica em view — CORRECTO:
-- BD: Operations
SELECT parcela, armacao, embardamento, ano_plantacao
FROM v_field_base
WHERE propriedade ILIKE '%[NOME_PROPRIEDADE]%' AND id_entidade = X
ORDER BY ano_plantacao DESC NULLS LAST
LIMIT 50

Tabela normal — query CORRECTA:
-- BD: Operations
SELECT t.denominacao, r.denominacao AS referencia, t.area
FROM tabela t
JOIN tipo_referencia r ON r.gid = t.id_tipo_referencia
WHERE r.denominacao ILIKE '%valor%' AND t.id_propriedade IN (SELECT gid FROM propriedade WHERE id_entidade = X)
ORDER BY t.area DESC NULLS LAST
LIMIT 50

SQL:"""

    _PIPELINE_FORMAT_PROMPT = """Formata os dados abaixo como resposta para um agricultor em português de Portugal.
- É ESTRITAMENTE PROIBIDO apresentar o código SQL, referir nomes de tabelas, mencionar queries, ou detalhar a BD na tua resposta final. Apresenta apenas a informação para o utilizador.
- Linguagem simples, sem termos técnicos.
- Se a tabela estiver vazia (0 resultados) ou a query retornar vazia, responde "Não foram encontrados registos que correspondam aos critérios." e PÁRA. NUNCA dês desculpas nem exponhas SQL.
- Cabeçalhos legíveis: "Nome" (não "denominacao"), "Área (ha)" (não "area_total").
- Omite campos internos (gid, usercreate, userupdate, createdon, updatedon).
- Mostra TODOS os resultados. NUNCA trunces.
- Se 0 resultados → "Não foram encontrados registos que correspondam aos critérios."

Pergunta: {question}
Dados:
{data}

Resposta:"""

    def _validate_and_fix_sql(self, sql: str, db_name: str, schemas_text: str, llm) -> str:
        """Valida sintaxe (sqlglot) e colunas (schemas em memória). Corrige via LLM se necessário."""
        import sqlglot
        import sqlglot.expressions as exp

        try:
            parsed = sqlglot.parse_one(sql, dialect="postgres")
        except Exception as sg_err:
            print(f"[PIPELINE] Erro sintático (sqlglot): {sg_err}")
            fixed = llm.invoke(
                f"A query SQL tem erro de sintaxe. Corrige-a. Responde APENAS com a query.\n"
                f"Query: {sql}\nErro: {sg_err}\nSQL corrigido:"
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
                        bad_cols.append(f"{table_ref}.{col_name} (tabela: {real_table})")

        if bad_cols:
            print(f"[PIPELINE] Colunas inválidas detectadas: {bad_cols}")

            valid_cols_hint = []
            for alias, real_table in alias_map.items():
                if real_table in known_cols and known_cols[real_table]:
                    cols_list = ", ".join(sorted(known_cols[real_table])[:30])
                    valid_cols_hint.append(f"  {alias} ({real_table}): {cols_list}")
            valid_hint = "\n".join(valid_cols_hint)

            fix_prompt = (
                f"A query SQL usa colunas que NÃO EXISTEM. Reescreve-a do zero usando APENAS as colunas listadas abaixo.\n"
                f"Se uma tabela é uma view (v_* ou view_*), usa directamente as suas colunas de texto — NÃO faças JOIN a tabelas tipo_*.\n\n"
                f"COLUNAS VÁLIDAS POR TABELA:\n{valid_hint}\n\n"
                f"Colunas inválidas a remover: {', '.join(bad_cols)}\n\n"
                f"Query original:\n{sql}\n\n"
                f"Responde APENAS com o SQL corrigido, sem backticks, sem explicações:"
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
                f"Se NÃO tiver (ex: parcela), usa subquery: WHERE id_propriedade IN "
                f"(SELECT gid FROM propriedade WHERE id_entidade = {entidade_id})."
            )
        if ano_agricola_id:
            ctx_parts.append(
                f"ATENÇÃO: id_ano_agricola = {ano_agricola_id} é um ID interno, NÃO é um ano de calendário. "
                f"Usa id_ano_agricola APENAS em tabelas/vistas que tenham explicitamente essa coluna. "
                f"Para filtrar por ano em colunas de data (data_inicio, data_registo, etc.), usa o ano que o utilizador mencionou na pergunta (ex: 2025, 2024). "
                f"NUNCA uses {ano_agricola_id} em EXTRACT(YEAR ...) ou em condições de data."
            )
        context = " ".join(ctx_parts) if ctx_parts else ""

        #injectar exemplos similares
        few_shot_text = self._get_few_shot_examples(question, k=2)
        if few_shot_text:
            context = context + "\n\nEXEMPLOS DE SQL CORRECTO PARA PADRÕES SIMILARES:\n" + few_shot_text if context else "EXEMPLOS DE SQL CORRECTO PARA PADRÕES SIMILARES:\n" + few_shot_text

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
            print(f"[PIPELINE] SQL inválido: {raw_sql[:100]}")
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
                    print(f"[PIPELINE] Substituído v_plot_resource_full → mv_plot_resource_full")
                    raw_sql = raw_sql_new

        print(f"[PIPELINE] BD: {db_name}")
        print(f"[PIPELINE] SQL: {raw_sql}")

        raw_sql = self._validate_and_fix_sql(raw_sql, db_name, schemas_text, llm)

        result = None
        sql_used = raw_sql
        for attempt in range(2):
            result = self._run_sql.invoke({"db_name": db_name, "query": sql_used})
            if "ERRO SQL" not in result:
                break
            if attempt == 0:
                print(f"[PIPELINE] Erro na tentativa {attempt+1}, a corrigir...")
                fix_prompt = (
                    f"A query SQL seguinte deu erro. Corrige-a usando APENAS as colunas dos schemas.\n"
                    f"Responde APENAS com a query corrigida, sem explicações.\n\n"
                    f"Query original:\n{sql_used}\n\nErro:\n{result}\n\n{schemas_text}\n\nSQL corrigido:"
                )
                fixed = llm.invoke(fix_prompt).content.strip()
                fixed = re.sub(r'```sql\s*', '', fixed, flags=re.IGNORECASE)
                fixed = re.sub(r'\s*```', '', fixed)
                sql_used = fixed.strip()
                print(f"[PIPELINE] SQL corrigido: {sql_used}")

        if not result or "ERRO SQL" in result:
            print(f"[PIPELINE] Falhou após 2 tentativas, a cair no fallback")
            return None, None

        print(f"[PIPELINE] A formatar resposta com Python...")
        
        if not result or "0 linhas" in result or "ERRO" in result:
            response = "Não foram encontrados registos que correspondam aos critérios."
        else:
            response = result

        print(f"[PIPELINE] Sucesso!")
        return response, sql_used

    # Contadores de métricas
    _metrics = {"pipeline_ok": 0, "pipeline_fail": 0, "fallback_ok": 0, "fallback_fail": 0, "cache_hit": 0}

    @classmethod
    def print_metrics(cls):
        m = cls._metrics
        total = m["pipeline_ok"] + m["pipeline_fail"] + m["fallback_ok"] + m["fallback_fail"]
        if total == 0:
            return
        pipeline_rate = round(m["pipeline_ok"] / max(m["pipeline_ok"] + m["pipeline_fail"], 1) * 100)
        print(f"\n[MÉTRICAS SQL] Total:{total} | Pipeline:{m['pipeline_ok']}✓ {m['pipeline_fail']}✗ ({pipeline_rate}%) | Fallback:{m['fallback_ok']}✓ {m['fallback_fail']}✗ | Cache:{m['cache_hit']}")

    def answer(self, question, chat_history=None, entidade_id=None, ano_agricola_id=None):
        """Pipeline controlado como primário; ReAct agent como fallback."""
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
                    print(f"[CACHE HIT] '{question}' ({int(time.time() - ts)}s atrás)")
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
            print(f"[PIPELINE ERRO]: {e} — a usar ReAct fallback")
            SQLAgentTools._metrics["pipeline_fail"] += 1

        print("[FALLBACK]: A usar ReAct agent")
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
                f"Entidade: id_entidade = {entidade_id}. "
                f"Filtra SEMPRE por esta entidade. Verifica no schema se a tabela tem id_entidade. "
                f"Se NÃO tiver (ex: parcela), usa subquery: WHERE id_propriedade IN (SELECT gid FROM propriedade WHERE id_entidade = {entidade_id})."
            )
        if ano_agricola_id:
            ctx_parts.append(
                f"ATENÇÃO: id_ano_agricola = {ano_agricola_id} é um ID interno, NÃO é um ano de calendário. "
                f"Usa id_ano_agricola APENAS em tabelas/vistas que tenham explicitamente essa coluna (ex: pratica_cultural, rota, lote). "
                f"Tabelas sem esta coluna (ex: parcela, propriedade, cuba) NÃO se filtram por ano. "
                f"Para filtrar por ano em colunas de data, usa o ano que o utilizador mencionou na pergunta. "
                f"NUNCA uses {ano_agricola_id} em EXTRACT(YEAR ...) ou em condições de data."
            )
        if ctx_parts:
            ctx = "CONTEXTO DO UTILIZADOR: " + " ".join(ctx_parts)
            messages_hist.append(HumanMessage(content=f"{ctx}\n\nPergunta: {question}"))
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
                return "A pergunta é demasiado complexa para ser processada automaticamente. Tenta reformulá-la de forma mais directa ou divide-a em partes mais simples.", "", elapsed
            messages = result["messages"]
            final_response = messages[-1].content if messages else "Sem resposta."
            last_msg_has_tool = hasattr(messages[-1], 'tool_calls') and len(messages[-1].tool_calls) > 0
            sql_used = ""
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get('name') == 'run_sql':
                            sql_used = tc.get('args', {}).get('query', '')
            if not last_msg_has_tool:
                lower_resp = final_response.lower()
                if any(w in lower_resp for w in ["vou agora", "escrever uma query", "verificar a relação", "precisamos primeiro", "escrever a query", "tentar novamente", "vou verificar", "vou abrir", "vou executar", "ocorreu um erro", "parece que houve"]):
                    print("\n[PYTHON RETRY]: IA parou prematuramente a divagar. A forçar a tool_call...")
                    messages_hist = messages + [HumanMessage(content="ERRO INTERNO: Respondeste com o texto do teu plano mas esqueceste-te de emitir a ferramenta real (tool_call JSON). Emite APENAS a ferramenta agora sem pedir desculpa.")]
                    continue
            break
        print("\n--- PENSAMENTO DA LLM ---")
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                print(f"[{msg.__class__.__name__}]: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[A Executar Tool]: {tc['name']} com args {tc['args']}")
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
                print(f"\n[FALLBACK SQL DETETADO]: Executando '{sql_used}' na BD '{db_name}'...")
                fallback_result = self._run_sql.invoke({"db_name": db_name, "query": sql_used})
                final_response = f"**Dados calculados automaticamente pelo Fallback:**\n\n{fallback_result}\n\n*Nota: O modelo comportou-se de forma não esperada, mas a resposta foi contornada com sucesso.*"
        final_response = re.sub(r'```sql[\s\S]*?```', '', final_response, flags=re.IGNORECASE)
        final_response = re.sub(r'```[\s\S]*?```', '', final_response)
        final_response = re.sub(r'(?im)^\s*(SELECT|WITH)\s+.*?(FROM|JOIN)\b.*$', '', final_response)
        final_response = re.sub(r'(?im)^\s*(FROM|JOIN|WHERE|GROUP BY|ORDER BY|LIMIT)\s+.*$', '', final_response)
        final_response = re.sub(r'ERRO SQL\s*\[.*?\]:.*', '', final_response, flags=re.IGNORECASE)
        final_response = re.sub(r'DICA DO SISTEMA:.*', '', final_response, flags=re.IGNORECASE)
        final_response = re.sub(
            r'(Vou agora|Para determinar|Precisamos primeiro|Desculp[ea] pela confus|'
            r'parece que houve um erro|tabela.*não encontrada|Vou.*verificar|Vou.*abrir|'
            r'Após várias tentativas|Vou corrigir|Aqui está a query|Vou executar|'
            r'Vou usar a ferramenta|Vamos executar|não é um ID numérico|'
            r'Os resultados da consulta|caminho de JOINs|A consulta retornou)[^\n]*',
            '', final_response, flags=re.IGNORECASE
        ).strip()
        final_response = re.sub(r'`[a-z_]+`', '', final_response).strip()
        final_response = re.sub(r'[ \t]{2,}', ' ', final_response).strip()
        final_response = re.sub(r'\n{3,}', '\n\n', final_response).strip()
        if not final_response:
            final_response = "Desculpa, a tua instrução parece necessitar de cruzar demasiados dados em simultâneo. Podes tentar fazer a pergunta de forma mais direta ou dividi-la em partes mais pequenas?"
        if not chat_history and sql_used:
            self._cache[cache_key] = (final_response, sql_used, time.time())
        elapsed = time.time() - start
        if sql_used:
            SQLAgentTools._metrics["fallback_ok"] += 1
        else:
            SQLAgentTools._metrics["fallback_fail"] += 1
        self.print_metrics()
        return final_response, sql_used, elapsed
