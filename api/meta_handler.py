import os
import re
from functools import lru_cache
from core.config import DOCS_DIR
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)

@lru_cache(maxsize=1)
def get_modules_overview() -> str:
    if not os.path.isdir(DOCS_DIR):
        return "I don't have any modules indexed."
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
        return "I don't have any modules indexed."
    linhas = [f"- **{nome}**" for nome in modulos]
    return (
        "I am the AgriSystem support assistant. I can help you with "
        "the following modules:\n\n"
        + "\n".join(linhas)
        + "\n\nAsk me how to perform a specific task (e.g., "
        '"How do I add a plot?") or request data from the databases '
        "(Field Operations, Plot Management and Cellar Management)."
    )
