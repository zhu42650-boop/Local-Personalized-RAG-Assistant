import json
import os
import re
from typing import List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def _build_embeddings(model_name: str, device: str, batch_size: int) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"batch_size": batch_size},
    )


def get_retriever(
    chroma_dir: str,
    model_name: str,
    device: str,
    batch_size: int,
    top_k: int,
    chunks_file: str = "",
    top_k_vector: Optional[int] = None,
    top_k_bm25: Optional[int] = None,
    top_k_final: Optional[int] = None,
    rerank_model: str = "",
):
    embeddings = _build_embeddings(model_name, device, batch_size)
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    if chunks_file and os.path.isfile(chunks_file):
        return HybridRetriever(
            db=db,
            chunks_file=chunks_file,
            top_k_vector=top_k_vector or top_k,
            top_k_bm25=top_k_bm25 or top_k,
            top_k_final=top_k_final or top_k,
            rerank_model=rerank_model,
        )
    return db.as_retriever(search_kwargs={"k": top_k})


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _load_chunks(path: str) -> List[Document]:
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            meta = row.get("metadata", {}) or {}
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:120])
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


class HybridRetriever:
    def __init__(
        self,
        db: Chroma,
        chunks_file: str,
        top_k_vector: int,
        top_k_bm25: int,
        top_k_final: int,
        rerank_model: str,
    ) -> None:
        self.db = db
        self.top_k_vector = top_k_vector
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        self.rerank_model = rerank_model

        self.bm25_docs = _load_chunks(chunks_file)
        tokenized = [_tokenize(d.page_content) for d in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized)
        self.reranker = CrossEncoder(rerank_model) if rerank_model else None

    def _bm25_search(self, query: str) -> List[Document]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = ranked[: self.top_k_bm25]
        return [self.bm25_docs[i] for i in top]

    def _vector_search(self, query: str) -> List[Document]:
        return self.db.similarity_search(query, k=self.top_k_vector)

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not self.reranker or not docs:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked]

    def get_relevant_documents(self, query: str) -> List[Document]:
        vec_docs = self._vector_search(query)
        bm25_docs = self._bm25_search(query)
        merged = _dedupe_docs(vec_docs + bm25_docs)
        reranked = self._rerank(query, merged)
        return reranked[: self.top_k_final]

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
