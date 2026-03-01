#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


def _split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def _sorted_pdfs(input_dir: str) -> List[str]:
    return sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(".pdf")
        ]
    )

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_references(docs: List[Document]) -> List[Document]:
    if not docs:
        return docs

    ref_pattern = re.compile(
        r"^\s*(references|bibliography|acknowledgments|acknowledgements)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    cleaned_docs: List[Document] = []
    stop = False
    for doc in docs:
        if stop:
            break
        text = doc.page_content or ""
        match = ref_pattern.search(text)
        if match:
            text = text[: match.start()].strip()
            stop = True
        if text:
            new_doc = Document(page_content=text, metadata=dict(doc.metadata))
            cleaned_docs.append(new_doc)
    return cleaned_docs


def chunk_papers(
    input_dir: str,
    output_path: str,
    chunk_size: int,
    chunk_overlap: int,
    start_paper_id: int,
    start_chunk_id: int,
    skip_bad_pdf: bool,
    strip_references: bool,
    clean_text: bool,
) -> Tuple[int, int]:
    pdf_files = _sorted_pdfs(input_dir)
    if not pdf_files:
        return 0, start_chunk_id

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_chunks = 0
    chunk_id = start_chunk_id

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, pdf_path in enumerate(pdf_files, start=start_paper_id):
            try:
                docs = _load_pdf(pdf_path)
            except Exception as exc:
                msg = f"[WARN] Failed to load PDF: {os.path.basename(pdf_path)} ({exc})"
                if skip_bad_pdf:
                    print(msg)
                    continue
                raise
            if strip_references:
                docs = _strip_references(docs)
            if clean_text:
                for doc in docs:
                    doc.page_content = _clean_text(doc.page_content)

            chunks = _split_docs(docs, chunk_size, chunk_overlap)

            for i, chunk in enumerate(chunks):
                record = {
                    "paper_id": idx,
                    "chunk_id": chunk_id,
                    "page": chunk.metadata.get("page"),
                    "source": chunk.metadata.get("source"),
                    "text": chunk.page_content.strip(),
                }
                if not record["text"]:
                    continue
                out.write(json.dumps(record, ensure_ascii=True) + "\n")
                total_chunks += 1
                chunk_id += 1

            print(f"[{idx}/{len(pdf_files)}] paper_{idx}: {len(chunks)} chunks")

    return total_chunks, chunk_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk PDFs into JSONL records.")
    parser.add_argument("--input-dir", default="data/raw_papers", help="Directory with PDF files.")
    parser.add_argument("--output", default="data/chunks.jsonl", help="Output JSONL path.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters.")
    parser.add_argument("--start-paper-id", type=int, default=1, help="Start value for paper_id.")
    parser.add_argument("--start-chunk-id", type=int, default=1, help="Start value for chunk_id.")
    parser.add_argument(
        "--skip-bad-pdf",
        action="store_true",
        help="Skip unreadable PDFs instead of raising.",
    )
    parser.add_argument(
        "--no-strip-references",
        action="store_true",
        help="Keep references section (default is to drop it).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable text cleaning (newline/whitespace normalization).",
    )
    args = parser.parse_args()

    total, last_chunk_id = chunk_papers(
        input_dir=args.input_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        start_paper_id=args.start_paper_id,
        start_chunk_id=args.start_chunk_id,
        skip_bad_pdf=args.skip_bad_pdf,
        strip_references=not args.no_strip_references,
        clean_text=not args.no_clean,
    )
    print(f"Total chunks: {total} (last_chunk_id={last_chunk_id - 1})")


if __name__ == "__main__":
    main()
