"""
Router de classificacao de perguntas — RAG / SQL / BOTH / GREETING / CHITCHAT.
Baseado em keywords + entidades da BD. Instantaneo, sem LLM.
"""
import re
import unicodedata

ABBREVIATIONS = {
    "q": "que", "pq": "porque", "tb": "tambem", "tbm": "tambem",
    "vc": "voce", "td": "tudo", "tds": "todos", "qnd": "quando",
    "qd": "quando", "qto": "quanto", "qta": "quanta", "qts": "quantos",
    "qtas": "quantas", "n": "nao", "nd": "nada", "blz": "beleza",
    "obg": "obrigado", "msg": "mensagem", "ent": "entao",

}

DB_ENTITIES = {
    "parcela", "parcelas", "propriedade", "propriedades",
    "fitofarmaco", "fitofarmacos", "fitofarmaceutico", "fitofarmaceuticos",
    "pratica", "praticas", "pratica", "pratica cultural", "pratica cultural", "praticas culturais",
    "equipamento", "equipamentos", "recurso", "recursos",
    "colheita", "colheitas", "tratamento", "tratamentos",
    "rota", "rotas", "compra", "compras",
    "rebanho", "rebanhos", "animal", "animais",
    "variedade", "variedades", "casta", "castas",
    "adubo", "adubos", "fertilizante", "fertilizantes",
    "fornecedor", "fornecedores", "produto", "produtos",
    "calda", "caldas", "rega", "regas",
    "operacao", "operacoes", "operacao", "operacoes",
    "manutencao", "manutencoes", "manutencao",
    "distrito", "distritos", "concelho", "concelhos", "freguesia", "freguesias",
    "utilizador", "utilizadores",
    "perfil de solo", "perfis de solo",
    "orcamento", "orcamentos", "orcamento", "orcamentos",
    "empreiteiro", "empreiteiros",
    "unidade de gestao", "unidades de gestao",
    "entidade", "entidades",
    "nutriente", "nutrientes",
    "cultura", "culturas", "solo", "solos",
    "exposicao", "exposicao", "armacao", "armacao",
    "conducao", "conducao", "producao", "producao",
    "gps", "ponto gps", "pontos gps",
    "stock", "baixa", "baixas",
    "colhida", "colhido",
    "vinho", "vinhos", "lote", "lotes", "cuba", "cubas",
    "adega", "adegas", "lagar", "lagares",
    "vindima", "vindimas", "rececao", "rececao",
    "fermentacao", "fermentacao", "fermentacoes",
    "engarrafamento", "engarrafar", "engarrafado",
    "barrica", "barricas", "pipa", "pipas",
    "estagio", "estagio", "trasfega", "trasfegas",
    "rotulagem", "rotular", "embalamento",
    "lote de vinho", "lotes de vinho",
    "tinto", "branco", "rose", "rose",
    "certificacao", "certificacao", "certificado", "certificados",
    "a granel", "vendas a granel",
    "tanoeiro", "tanoeiros",
    "produto enologico", "produtos enologicos",
    "bica", "bicas",

}

SQL_STRONG = {
    "quantas", "quantos", "quantia", "total", "soma",
    "media", "media", "custo", "custos", "valor",
    "registados", "registadas", "registos",
    "ultimo", "ultima", "este ano", "em 2024", "em 2025", "em 2026",
    "quem fez", "quem registou", "quando foi",
    "existe", "existem", "ha", "temos",
    "algum", "alguma", "alguns", "algumas",
    "com mais", "com menos", "mais de", "menos de",
    "esta semana", "este mes", "este mes",
    "em stock",

}

SQL_WEAK = {
    "lista", "listar", "mostrar", "mostra-me", "mete",
    "quais os", "quais as", "quais foram",

}

RAG_ONLY = {
    "como", "onde encontrar", "onde esta", "onde fica",
    "onde registo", "onde registar", "onde vejo", "onde ver",
    "onde encontro", "onde acedo", "onde aceder", "onde configuro",
    "onde adiciono", "onde edito", "onde clico",
    "passos", "procedimento", "passo a passo",
    "para que serve", "o que e", "o que faz",
    "menu", "botao", "interface",
    "navegacao", "navegar", "funcionalidade",
    "campos", "obrigatorio", "obrigatorios", "configurar", "configuracao",
    "diferenca", "fluxo", "exportar",
    "posso", "disponiveis", "no sistema",

}

FORCE_RAG = {
    "opcoes", "opcoes", "botao", "botao",
    "como vejo", "como ver", "como visualizo", "como visualizar",
    "como posso ver", "onde vejo", "onde ver",
    "o que sao", "o que sao", "o que e ", "o que e ",
    "para que serve", "para que sao", "para que sao",
    "qual a diferenca", "qual a diferenca",
}

META_PATTERNS = {
    "o que podes fazer", "o que sabes fazer", "o que fazes", "o que consegues",
    "o que sabes", "para que serves", "qual a tua funcao", "quem es tu",
    "que modulos", "quais modulos", "quais os modulos", "que modulos existem",
    "quais funcionalidades", "que funcionalidades", "lista de modulos",
    "como podes ajudar", "em que podes ajudar", "ajudas em que",
    "do que es capaz", "tuas capacidades",

}

GREETINGS = {"ola", "ola", "bom dia", "boa tarde", "boa noite", "ei", "hey", "hi", "hello", "oi"}

CHITCHAT = {
    "ok", "okay", "sim", "nao", "nao", "obrigado", "obrigada", "thanks",
    "certo", "entendi", "percebi", "fixe", "tudo bem", "claro",

}

GREETING_RESPONSE = "Ola! Sou o assistente de suporte ao AgriSystem. Como posso ajudar?"
CHITCHAT_RESPONSE = "Entendido! Se tiver mais alguma questao sobre o AgriSystem, estou aqui para ajudar."

def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII").lower()
    return re.sub(r"\b(\w+)\b", lambda m: ABBREVIATIONS.get(m.group(1), m.group(1)), t)

def classify(question: str) -> str:
    q_raw = question.strip()
    q = normalize(q_raw)
    if q in {normalize(g) for g in GREETINGS}:
        return "GREETING"
    if q in {normalize(c) for c in CHITCHAT}:
        return "CHITCHAT"
    if any(normalize(p) in q for p in META_PATTERNS):
        return "META"
    has_rag       = any(normalize(kw) in q for kw in RAG_ONLY)
    has_strong    = any(normalize(kw) in q for kw in SQL_STRONG)
    has_weak      = any(normalize(kw) in q for kw in SQL_WEAK)
    has_entity    = any(normalize(ent) in q for ent in DB_ENTITIES)
    has_force_rag = any(normalize(kw) in q for kw in FORCE_RAG)
    if has_force_rag:
        return "RAG"
    if has_rag:
        if has_strong and has_entity:
            return "BOTH"
        return "RAG"
    if (has_strong or has_weak) and has_entity:
        return "SQL"
    if has_entity:
        return "SQL"
    return "RAG"
