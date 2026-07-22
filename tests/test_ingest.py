"""
Testes unitários ao pipeline de ingestão (splitter, prefixos, metadata).
100 testes.
"""
import sys
import os
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.config import CHUNK_SIZE, CHUNK_OVERLAP

HEADERS = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]

@pytest.fixture
def md_splitter():
    return MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)

@pytest.fixture
def text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

MD_DOCS = [
    ("# T1\nTexto", 1),
    ("# T1\n## T2\nTexto", 1),
    ("# T1\nA\n# T2\nB", 2),
    ("## H2\nApenas H2", 1),
    ("### H3\nApenas H3", 1),
    ("# A\n## B\n### C\nFim", 1),
    ("# A\nTexto1\n# B\nTexto2\n# C\nTexto3", 3),
    ("# Módulo\n## Secção\nConteúdo", 1),
    ("# Cellar\n## Cubas\nLista", 1),
    ("# Field Monitoring\n## Parcelas\n### Propriedades\nDados", 1),
    ("# T\n" + "Linha\n" * 50, 1),
    ("# A\n\n## B\n\n## C\nFim", 1),
    ("# 1\n## 1.1\nTexto A\n# 2\n## 2.1\nTexto B", 2),
    ("# Á\nAcento", 1),
    ("# Título com espaços\nX", 1),
    ("# A1\nX\n# A2\nY\n# A3\nZ\n# A4\nW", 4),
    ("# T\n## S1\nX\n## S2\nY", 1),
    ("### Só H3\nTexto", 1),
    ("# Top\nIntro\n## Sub\nDetalhe", 1),
    ("# X\nTexto único", 1),
]

@pytest.mark.parametrize("md,min_splits", MD_DOCS)
def test_md_split_count(md_splitter, md, min_splits):
    splits = md_splitter.split_text(md)
    assert len(splits) >= min_splits

TEXT_CASES = [
    "x" * 100,
    "x" * 400,
    "x" * 800,
    "x" * 1500,
    "x" * 3000,
    "palavra " * 50,
    "palavra " * 100,
    "palavra " * 200,
    "frase curta.",
    "Frase 1.\n\nFrase 2.\n\nFrase 3.",
    "linha 1\nlinha 2\nlinha 3\nlinha 4",
    "a\n\nb\n\nc\n\nd",
    "parágrafo " * 80,
    "Texto longo. " * 60,
    "P1\n\n" + ("dados " * 100),
    "## Cabeçalho\n" + ("conteúdo " * 100),
    "Lista:\n- A\n- B\n- C\n- D",
    "Tabela:\n| a | b |\n|---|---|\n| 1 | 2 |",
    "código:\n```py\nprint('x')\n```",
    "url: https://example.com/path",
    "Acentuação: ção, ã, õ, é, í, ú",
    "Mixed: ABC abc 123 áéí",
    "Quebras\rinusuais\rno\rtexto",
    "Tabs:\tA\tB\tC",
    "Múltiplos\n\n\n\nseparadores",
    "x",
    "Texto com pontuação! E? Sim; também: outras, vírgulas.",
    ("a " * 200).strip(),
    "Linha única muito longa " * 30,
    "Documento\n\nCom\n\nVários\n\nParágrafos\n\nDistintos",
]

@pytest.mark.parametrize("text", TEXT_CASES)
def test_text_split_no_overflow(text_splitter, text):
    chunks = text_splitter.split_text(text)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE + 100

@pytest.mark.parametrize("text", TEXT_CASES)
def test_text_split_returns_chunks(text_splitter, text):
    chunks = text_splitter.split_text(text)
    assert len(chunks) >= 1


FILE_CASES = [
    ("notes.md", True),
    ("sigp_notes.md", True),
    ("cellar.md", True),
    ("_relatorio.md", False),
    ("_draft.md", False),
    ("_temp.md", False),
    ("module.md", True),
    ("_internal_notes.md", False),
    ("monitoring_parcelas_notes.md", True),
    ("_relatorio_cap5.md", False),
]

@pytest.mark.parametrize("filename,should_index", FILE_CASES)
def test_file_filter(filename, should_index):
    is_indexed = not filename.startswith("_")
    assert is_indexed == should_index


@pytest.mark.parametrize("source,module", [
    ("cellar.md", "Cellar"),
    ("cellar_notes.md", "Cellar"),
    ("cellar_notas.md", "Cellar"),
    ("cellar_module_notes.md", "Cellar Module"),
    ("monitoring.md", "Operations"),
    ("monitoring_notes.md", "Operations"),
    ("monitoring_parcelas.md", "Operations Parcelas"),
    ("monitoring_parcelas_notes.md", "Operations Parcelas"),
    ("praticas_agricolas.md", "Praticas Agricolas"),
    ("praticas_agricolas_notes.md", "Praticas Agricolas"),
])
def test_module_name_extraction(source, module):
    name = (
        source.replace("_notes.md", "")
        .replace("_notas.md", "")
        .replace(".md", "")
        .replace("_", " ")
        .title()
    )
    assert name == module
