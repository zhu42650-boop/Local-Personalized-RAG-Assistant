import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)


def _infer_category(source_path: str, base_dir: str) -> str:
    try:
        rel = os.path.relpath(source_path, base_dir)
    except ValueError:
        rel = source_path
    parts = rel.split(os.sep)
    if parts:
        return parts[0]
    return "default"


def load_documents(input_dir: str) -> List[Document]:
    # Simple multi-format loader; extend as needed.
    loaders = [
        DirectoryLoader(input_dir, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True),
        DirectoryLoader(input_dir, glob="**/*.md", loader_cls=TextLoader, silent_errors=True),
        DirectoryLoader(input_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True),
        DirectoryLoader(input_dir, glob="**/*.csv", loader_cls=CSVLoader, silent_errors=True),

        # --- Optional extensions (uncomment after installing deps) ---
        # doc/docx
        # DirectoryLoader(input_dir, glob="**/*.docx", loader_cls=Docx2txtLoader, silent_errors=True),
        # DirectoryLoader(input_dir, glob="**/*.doc", loader_cls=UnstructuredWordDocumentLoader, silent_errors=True),

        # ppt/pptx
        # DirectoryLoader(input_dir, glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader, silent_errors=True),
        # DirectoryLoader(input_dir, glob="**/*.ppt", loader_cls=UnstructuredPowerPointLoader, silent_errors=True),

        # xlsx
        # DirectoryLoader(input_dir, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader, silent_errors=True),

        # html
        # DirectoryLoader(input_dir, glob="**/*.html", loader_cls=UnstructuredHTMLLoader, silent_errors=True),
        # DirectoryLoader(input_dir, glob="**/*.htm", loader_cls=UnstructuredHTMLLoader, silent_errors=True),

        # code files
        # DirectoryLoader(input_dir, glob="**/*.py", loader_cls=TextLoader, silent_errors=True),
        # DirectoryLoader(input_dir, glob="**/*.js", loader_cls=TextLoader, silent_errors=True),
    ]
    docs: List[Document] = []
    for loader in loaders:
        docs.extend(loader.load())
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source:
            doc.metadata["category"] = _infer_category(source, input_dir)
    return docs
