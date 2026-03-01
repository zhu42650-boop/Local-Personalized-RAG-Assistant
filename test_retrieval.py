#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.join(ROOT, "framework")
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from framework.config.loader import load_settings, resolve_paths  # noqa: E402
from framework.ingest.service import run_ingest  # noqa: E402
from framework.rag.retriever import get_retriever  # noqa: E402


def _clear_path(path: str) -> None:
    if not path:
        return
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG retrieval test.")
    parser.add_argument("--config", default="framework/config/settings.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--reindex", action="store_true", help="Rebuild index from knowledge base.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    paths = resolve_paths(settings, args.config)

    if args.reindex:
        _clear_path(paths.get("vector_store_dir"))
        _clear_path(paths.get("chunks_file"))

    count = run_ingest(settings, paths)
    print(f"Indexed chunks: {count}")

    top_k = args.top_k or settings.get("retriever.top_k")
    retriever = get_retriever(
        chroma_dir=paths["vector_store_dir"],
        model_name=settings.get("embedding.model_name"),
        device=settings.get("embedding.device"),
        batch_size=settings.get("embedding.batch_size"),
        top_k=top_k,
        chunks_file=paths.get("chunks_file", ""),
        top_k_vector=settings.get("retriever.top_k_vector"),
        top_k_bm25=settings.get("retriever.top_k_bm25"),
        top_k_final=settings.get("retriever.top_k_final"),
        rerank_model=settings.get("rerank.model_name") or "",
    )

    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(args.query)
    else:
        docs = retriever.invoke(args.query)
    print(f"Retrieved {len(docs)} docs")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = snippet[:400] + ("..." if len(snippet) > 400 else "")
        print("=" * 80)
        print(f"RANK {i} | source={meta.get('source','')} | page={meta.get('page','')}")
        print(f"category={meta.get('category','')} | section={meta.get('section','')}")
        print("-" * 80)
        print(snippet)


if __name__ == "__main__":
    main()
