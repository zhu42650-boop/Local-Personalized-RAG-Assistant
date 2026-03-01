from config.env_check import ensure_dirs
from config.loader import Settings
from ingest.indexer import build_or_update_index
from ingest.loaders import load_documents
from ingest.splitters import split_documents
import json
import os


def run_ingest(settings: Settings, paths: dict) -> int:
    ensure_dirs(paths)
    kb_dir = paths["knowledge_base_dir"]
    vs_dir = paths["vector_store_dir"]

    docs = load_documents(kb_dir)
    category_config = {
        "paper": {
            "size": settings.get("chunk.paper_size", settings.get("chunk.size")),
            "overlap": settings.get("chunk.paper_overlap", settings.get("chunk.overlap")),
        },
        "note": {
            "size": settings.get("chunk.note_size", settings.get("chunk.size")),
            "overlap": settings.get("chunk.note_overlap", settings.get("chunk.overlap")),
        },
    }
    chunks = split_documents(
        docs,
        chunk_size=settings.get("chunk.size"),
        chunk_overlap=settings.get("chunk.overlap"),
        category_config=category_config,
    )
    chunks_file = paths.get("chunks_file")
    if chunks_file:
        if os.path.isdir(chunks_file):
            chunks_file = os.path.join(chunks_file, "chunks.jsonl")
        os.makedirs(os.path.dirname(chunks_file), exist_ok=True)
        with open(chunks_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                record = {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return build_or_update_index(
        chunks,
        chroma_dir=vs_dir,
        model_name=settings.get("embedding.model_name"),
        device=settings.get("embedding.device"),
        batch_size=settings.get("embedding.batch_size"),
    )
