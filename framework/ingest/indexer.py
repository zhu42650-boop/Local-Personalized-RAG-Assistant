import os
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def _build_embeddings(model_name: str, device: str, batch_size: int) -> HuggingFaceEmbeddings:
    # Avoid mmap load issues with some HF checkpoints on older torch builds.
    os.environ.setdefault("TORCH_LOAD_USE_MMAP", "0")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
            "trust_remote_code": True,
        },
        encode_kwargs={"batch_size": batch_size},
    )


def build_or_update_index(
    docs: List[Document],
    chroma_dir: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> int:
    if not docs:
        return 0

    os.makedirs(chroma_dir, exist_ok=True)
    embeddings = _build_embeddings(model_name, device, batch_size)

    if os.listdir(chroma_dir):
        db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        db.add_documents(docs)
    else:
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=chroma_dir,
        )

    db.persist()
    return db._collection.count()
