import os
import re
from functools import lru_cache
from core.config import DOCS_DIR
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)

@lru_cache(maxsize=1)
def get_modules_overview() -> str:
    if not os.path.isdir(DOCS_DIR):
        return "Nao tenho modulos indexados."
    EXCLUDE = {"system_notes.md"}
    seen: set[str] = set()
    modulos: list[str] = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if not fname.endswith(".md") or fname.startswith("_") or fname in EXCLUDE:
            continue
        path = os.path.join(DOCS_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                head = f.read(2048)
        except OSError:
            continue
        m = _TITLE_RE.search(head)
        if not m:
            continue
        titulo = m.group(1).strip()
        nome = re.split(r"\s[—–-]\s", titulo, maxsplit=1)[0].strip()
        nome = re.sub(r"^M[ÓO]DULO\s+", "", nome, flags=re.IGNORECASE).strip()
        key = nome.lower()
        if key in seen:
            continue
        seen.add(key)
        modulos.append(nome)
    if not modulos:
        return "Nao tenho modulos indexados."
    linhas = [f"- **{nome}**" for nome in modulos]
    return (
        "Sou o assistente de suporte ao AgriSystem. Posso ajudar-te com "
        "os seguintes modulos:\n\n"
        + "\n".join(linhas)
        + "\n\nPergunta-me como executar uma tarefa especifica (ex.: "
        '"Como adiciono uma parcela?") ou pede dados das bases de dados '
        "(Field Operations, Plot Management e Cellar Management)."
    )
