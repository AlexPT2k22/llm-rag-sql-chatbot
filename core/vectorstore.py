import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from core.config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    CHROMA_DIR, DOCS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,

)

HEADER_KEYS = ["Header 1", "Header 2", "Header 3"]


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def init_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Carrega o vectorstore existente ou cria-o a partir dos documentos Markdown."""
    if os.path.exists(CHROMA_DIR):
        vs = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        n = vs._collection.count()
        print(f"[vectorstore] Índice existente carregado ({n} chunks) — {CHROMA_DIR}")
        return vs
    print(f"[vectorstore] Índice não encontrado. A executar ingestão de {DOCS_DIR}/ ...")
    md_files = [
        f for f in glob.glob(os.path.join(DOCS_DIR, "*.md"))
        if not os.path.basename(f).startswith("_")
    ]
    print(f"[vectorstore] {len(md_files)} ficheiros Markdown encontrados")
    headers_to_split_on = [
        ("#",   "Header 1"),
        ("##",  "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    documents = []
    for md_path in md_files:
        loader = TextLoader(md_path, encoding="utf-8")
        raw_docs = loader.load()
        module_name = (
            os.path.basename(md_path)
            .replace("_notes.md", "")
            .replace("_notas.md", "")
            .replace("_", " ")
            .title()
        )
        for raw_doc in raw_docs:
            splits = markdown_splitter.split_text(raw_doc.page_content)
            for split in splits:
                split.metadata["source"] = md_path
                split.metadata["module"] = module_name
                documents.append(split)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        module = chunk.metadata.get("module", "Desconhecido")
        header_context = " > ".join(
            [chunk.metadata[k] for k in HEADER_KEYS if k in chunk.metadata]
        )
        prefix = f"[Módulo: {module}]"
        if header_context:
            prefix += f" [Secção: {header_context}]"
        chunk.page_content = f"{prefix}\n{chunk.page_content}"
    print(f"[vectorstore] {len(chunks)} chunks gerados. A indexar...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"[vectorstore] Ingestão concluída — {CHROMA_DIR}")
    return vs
