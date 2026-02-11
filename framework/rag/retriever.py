from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def _build_embeddings(model_name: str, device: str, batch_size: int) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"batch_size": batch_size},
    )


def get_retriever(
    chroma_dir: str, model_name: str, device: str, batch_size: int, top_k: int
):
    embeddings = _build_embeddings(model_name, device, batch_size)
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": top_k})
