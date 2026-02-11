from config.env_check import ensure_dirs
from config.loader import Settings
from ingest.indexer import build_or_update_index
from ingest.loaders import load_documents
from ingest.splitters import split_documents


def run_ingest(settings: Settings, paths: dict) -> int:
    ensure_dirs(paths)
    kb_dir = paths["knowledge_base_dir"]
    vs_dir = paths["vector_store_dir"]

    docs = load_documents(kb_dir)
    chunks = split_documents(
        docs,
        chunk_size=settings.get("chunk.size"),
        chunk_overlap=settings.get("chunk.overlap"),
    )
    return build_or_update_index(
        chunks,
        chroma_dir=vs_dir,
        model_name=settings.get("embedding.model_name"),
        device=settings.get("embedding.device"),
        batch_size=settings.get("embedding.batch_size"),
    )
