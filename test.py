#!/usr/bin/env python3
import argparse
import os

from langchain_community.document_loaders import PyPDFLoader

from framework.ingest.splitters import split_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Test paper chunking for a single PDF.")
    parser.add_argument("--pdf", required=True, help="Path to a paper PDF.")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")

    docs = PyPDFLoader(args.pdf).load()
    for doc in docs:
        doc.metadata["category"] = "paper"

    chunks = split_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        category_config={
            "paper": {"size": args.chunk_size, "overlap": args.chunk_overlap}
        },
    )

    for i, chunk in enumerate(chunks, start=1):
        section = chunk.metadata.get("section", "unknown")
        page = chunk.metadata.get("page", "n/a")
        source = chunk.metadata.get("source", "")
        print("=" * 80)
        print(f"CHUNK {i} | section={section} | page={page} | source={source}")
        print("-" * 80)
        print(chunk.page_content.strip())
        print()


if __name__ == "__main__":
    main()
