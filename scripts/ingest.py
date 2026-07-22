import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.vectorstore import get_embeddings, init_vectorstore
import core.config as cfg

def main():
    if os.path.exists(cfg.CHROMA_DIR):
        print(f"Vectorstore já existe em {cfg.CHROMA_DIR}/")
        print("Para reindexar, apaga a pasta e corre novamente.")
        return
    print("A carregar embeddings...")
    embeddings = get_embeddings()
    print(f"A indexar documentos de {cfg.DOCS_DIR}/...")
    vectorstore = init_vectorstore(embeddings)
    col = vectorstore._collection
    print(f"Ingestão concluída — {col.count()} chunks indexados em {cfg.CHROMA_DIR}/")


if __name__ == "__main__":
    main()
