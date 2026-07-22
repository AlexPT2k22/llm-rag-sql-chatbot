import sys
import os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.router_classify import classify, normalize

GREETING_CASES = [
    "olá", "ola", "Olá", "OLÁ", "oLá",
    "bom dia", "Bom Dia", "BOM DIA", " bom dia", "bom dia",
    "boa tarde", "Boa Tarde", "BOA TARDE", "boa tarde", " boa tarde",
    "boa noite", "Boa Noite", "BOA NOITE", "boa noite ", " boa noite",
    "ei", "Ei", "EI", "hey", "Hey",
    "hi", "Hi", "HI", "hello", "oi",
]

@pytest.mark.parametrize("question", GREETING_CASES)
def test_greeting(question):
    assert classify(question.strip()) == "GREETING", f"Falhou: {question!r}"


CHITCHAT_CASES = [
    "ok", "OK", "Ok", "okay", "Okay",
    "sim", "Sim", "SIM", "não", "Não",
    "nao", "NAO", "Nao", "obrigado", "Obrigado",
    "OBRIGADO", "obrigada", "Obrigada", "thanks", "Thanks",
    "certo", "Certo", "entendi", "Entendi", "percebi",
    "Percebi", "fixe", "Fixe", "tudo bem", "claro",
]

@pytest.mark.parametrize("question", CHITCHAT_CASES)
def test_chitchat(question):
    assert classify(question) == "CHITCHAT", f"Falhou: {question!r}"


RAG_CASES = [
    # Cellar (12)
    "O que faz o módulo Cellar?",
    "Como adiciono uma cuba à cellar?",
    "Como configuro um lote de vinho?",
    "Como faço uma vinificação?",
    "Onde registo a entrada de uvas na cellar?",
    "Como adiciono uma análise de mosto?",
    "Como faço o engarrafamento?",
    "Como registo uma fermentação?",
    "Como faço a gestão de barricas?",
    "Como vejo os lotes de vinho disponíveis?",
    "Como crio um lagar de azeite?",
    "O que é a representação estrutural da cellar?",

    # Field Monitoring de Parcelas (12)
    "Como adiciono uma parcela?",
    "Como crio uma propriedade?",
    "Para que serve o módulo de monitorização de parcelas?",
    "Como apago uma parcela?",
    "Como edito as coordenadas de uma propriedade?",
    "Como adiciono uma rota de monitorização?",
    "Como registo um acidente climático?",
    "Para que serve o campo de exposição solar?",
    "Como configuro perfis de solo?",
    "Como agendo uma colheita?",
    "Como registo uma visita técnica?",
    "Como vejo o cadastro das explorações?",

    # Field Operations (11)
    "Como funciona o módulo de práticas agrícolas?",
    "Onde encontro os relatórios de fitofármacos?",
    "Onde acedo ao stock de fitofármacos?",
    "Como adiciono um equipamento?",
    "Como gero um relatório de produção?",
    "Como faço o registo de uma parcela nova?",
    "Como registo uma prática cultural?",
    "Como registo uma compra?",
    "Como faço a gestão de recursos humanos?",
    "Como configuro alertas de pragas?",
    "Como registo uma rota diária?",
]

@pytest.mark.parametrize("question", RAG_CASES)
def test_rag(question):
    assert classify(question) == "RAG", f"Falhou: {question!r}"


@pytest.mark.parametrize("text,expected", [
    ("OLÁ", "ola"),
    ("Não", "nao"),
    ("Açúcar", "acucar"),
    ("Mónica", "monica"),
    ("Bom Dia", "bom dia"),
])
def test_normalize(text, expected):
    assert normalize(text) == expected


SQL_CASES = [
    # Field Monitoring
    "Quantas parcelas existem?",
    "Qual o total de propriedades registadas?",
    "Lista as parcelas no distrito do Porto",
    "Mostra-me todos os fitofármacos em stock",
    "Quais os concelhos com mais parcelas?",
    "Existem perfis de solo registados?",
    "Quantos distritos temos?",
    "Lista as freguesias com propriedades",
    "Quantas parcelas foram registadas em 2025?",
    "Qual a média de área das parcelas?",
    # Field Operations
    "Quantos equipamentos temos?",
    "Soma dos custos das compras este mês",
    "Lista os fornecedores de adubos",
    "Quantas práticas culturais foram registadas em 2024?",
    "Qual o stock atual de fertilizantes?",
    "Mostra as últimas rotas registadas",
    "Quantos tratamentos foram feitos esta semana?",
    "Lista os produtos com mais baixas",
    "Total de operações de rega registadas",
    "Quem registou a última colheita?",
    # Cellar
    "Quantos lotes de vinho existem?",
    "Lista as cubas em uso",
    "Quantas fermentações estão em curso?",
    "Soma do volume de vinho tinto",
    "Quantas barricas temos?",
    "Lista os lotes engarrafados em 2025",
    "Mostra as últimas trasfegas",
    "Quantas vindimas foram registadas este ano?",
    "Total de produtos enológicos em stock",
    "Lista os tanoeiros fornecedores",
]

@pytest.mark.parametrize("question", SQL_CASES)
def test_sql(question):
    assert classify(question) == "SQL", f"Falhou: {question!r}"


BOTH_CASES = [
    "Como adiciono uma parcela e quantas tenho registadas?",
    "Como crio um lote e quantos lotes existem?",
    "Como registo uma compra e qual o total de compras este ano?",
    "Como funciona o módulo de cellar e quantas cubas temos?",
    "Como adiciono um equipamento e quantos equipamentos existem?",
    "Como registo uma fermentação e quantas estão em curso?",
    "Como faço o engarrafamento e quantos lotes engarrafados temos?",
    "Como configuro alertas e quantos fitofármacos em stock temos?",
    "Como crio uma propriedade e quantas propriedades existem?",
    "Como registo uma prática cultural e quantas foram registadas?",
    "Como adiciono uma rota e quantas rotas existem?",
    "Como gero um relatório e qual o total de produção?",
    "Como faço uma vindima e quantas vindimas registadas?",
    "Como adiciono uma análise de mosto e quantas cubas temos?",
    "Como registo uma trasfega e quantas trasfegas existem?",
]

@pytest.mark.parametrize("question", BOTH_CASES)
def test_both(question):
    assert classify(question) == "BOTH", f"Falhou: {question!r}"


SQL_EDGE_CASES = [
    "quantas parcelas existem",
    "lista as cubas em uso",
    "qual o total de propriedades registadas",
    "QUANTAS PARCELAS EXISTEM?",
    "LISTA OS LOTES DE VINHO",
    "  quantas parcelas existem?  ",
    "lista    as   cubas",
    "qts parcelas existem?",
    "qtos lotes de vinho temos?",
    "diz-me quantas parcelas há",
    "mostra-me os lotes",
    "quero ver as cubas",
    "podes contar as parcelas?",
    "dá-me o total de fitofármacos",
    "preciso da lista de equipamentos",
]

@pytest.mark.parametrize("question", SQL_EDGE_CASES)
def test_sql_edge(question):
    assert classify(question) == "SQL", f"Falhou: {question!r}"


RAG_EDGE_CASES = [
    "como adiciono uma parcela",
    "como configuro um lote de vinho",
    "como faco uma vinificacao",
    "COMO ADICIONO UMA PARCELA?",
    "onde registo a entrada de uvas",
    "onde acedo aos relatorios",
    "onde configuro alertas",
    "onde vejo os lotes",
    "explica-me como criar uma parcela",
    "mostra-me como adicionar uma cuba",
]

@pytest.mark.parametrize("question", RAG_EDGE_CASES)
def test_rag_edge(question):
    assert classify(question) == "RAG", f"Falhou: {question!r}"

HARD_CASES = [
    ("explica a diferença entre os dois fluxos de trabalho", "RAG"),
    ("quais os passos para configurar a interface", "RAG"),
    ("para que serve o botão de exportar", "RAG"),
    ("posso navegar pelo menu principal", "RAG"),
    ("como funciona a navegação no sigp", "RAG"),
    ("qual o procedimento para exportar dados", "RAG"),
    ("explica passo a passo o que faço", "RAG"),
    ("o que é necessário para configurar perfis de solo", "RAG"),
    ("qual a funcionalidade de exportar relatórios", "RAG"),
    ("explica o conceito de fluxo de trabalho", "RAG"),
    ("preciso saber o número total de parcelas", "SQL"),
    ("diz-me todas as cubas que temos", "SQL"),
    ("queria ver as últimas trasfegas", "SQL"),
    ("podes contar quantos lotes existem em cellar", "SQL"),
    ("soma o volume total de vinho tinto produzido este ano", "SQL"),
    ("média de área das parcelas no distrito do porto", "SQL"),
    ("lista os fornecedores ativos", "SQL"),
    ("alguma cuba está vazia", "SQL"),
    ("ha lotes engarrafados em 2026", "SQL"),
    ("vinho tinto disponivel em stock", "SQL"),
    ("operações de rega registadas em 2025", "SQL"),
    ("concelhos do distrito de viseu", "SQL"),
    ("fitofarmacos com baixa rotacao", "SQL"),
    ("produtos enologicos disponiveis", "SQL"),
    ("tanoeiros parceiros da cellar", "SQL"),
    ("parcelas com mais área", "SQL"),
    ("cubas com menos de 1000 litros", "SQL"),
    ("qual o ultimo relatorio de producao registado", "SQL"),
    ("mete-me as parcelas no ecra", "SQL"),
    ("quais foram as compras deste mes", "SQL"),
    ("como adiciono uma parcela e quantas existem registadas", "BOTH"),
    ("como configuro alertas e quantos fitofarmacos temos em stock", "BOTH"),
    ("para que serve o modulo de cellar e quantas cellars temos", "BOTH"),
    ("o que faz o lagar e quantos lagares existem", "BOTH"),
    ("como crio uma cuba e diz quantas ha", "BOTH"),
    ("como registo uma compra e quantas compras existem este ano", "BOTH"),
    ("explica como criar lotes e diz-me quantos lotes ha", "BOTH"),
    ("o que é uma vindima e quantas vindimas existem em 2025", "BOTH"),
    ("como funciona o stock e quantos fitofarmacos temos", "BOTH"),
    ("para que serve o lagar e quantos lagares ha em portugal", "BOTH"),
    ("o que podes fazer", "META"),
    ("quais modulos existem no sigp", "META"),
    ("do que es capaz exatamente", "META"),
    ("lista de modulos disponiveis", "META"),
    ("quais as tuas capacidades", "META"),
    ("como podes ajudar-me hoje", "META"),
    ("em que podes ajudar", "META"),
    ("ajudas em que tarefas", "META"),
    ("para que serves no sigp", "META"),
    ("o que sabes sobre o sigp", "META"),
]

@pytest.mark.parametrize("question,expected", HARD_CASES)
def test_hard(question, expected):
    assert classify(question) == expected, f"Falhou: {question!r}"
