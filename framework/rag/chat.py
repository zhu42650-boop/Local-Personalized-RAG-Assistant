from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from rag.prompt import build_prompt


def _join_context(docs: List[Document]) -> str:
    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        text = doc.page_content.strip()
        if source:
            chunks.append(f"[{source}] {text}")
        else:
            chunks.append(text)
    return "\n\n".join(chunks)


def _retrieve_docs(retriever, question: str) -> List[Document]:
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(question)
    return retriever.invoke(question)


def _join_history(history: Optional[List[dict]], max_turns: int = 6) -> str:
    if not history:
        return ""
    trimmed = history[-max_turns:]
    lines = []
    for msg in trimmed:
        role = msg.get("role", "")
        text = msg.get("text", "")
        if role and text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _summarize_docs(
    docs: List[Document],
    llm: ChatOpenAI,
    max_chars_per_chunk: int,
) -> List[Document]:
    summarized: List[Document] = []
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk]
        prompt = (
            "Summarize the passage into 2-3 bullet points focused on key facts. "
            "Keep citations if present. Passage:\n\n"
            f"{text}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        summary = resp.content.strip()
        summarized.append(Document(page_content=summary, metadata=doc.metadata))
    return summarized


def answer_question(
    question: str,
    retriever,
    llm: ChatOpenAI,
    chat_history: Optional[List[dict]] = None,
    summary_llm: Optional[ChatOpenAI] = None,
    summary_cfg: Optional[Dict] = None,
) -> str:
    docs = _retrieve_docs(retriever, question)
    context = _join_context(docs)

    if summary_llm and summary_cfg:
        max_context_chars = int(summary_cfg.get("max_context_chars", 0) or 0)
        if max_context_chars and len(context) > max_context_chars:
            max_chars_per_chunk = int(summary_cfg.get("max_chars_per_chunk", 900))
            docs = _summarize_docs(docs, summary_llm, max_chars_per_chunk)
            context = _join_context(docs)

    history = _join_history(chat_history)
    prompt = build_prompt(context=context, question=question, history=history)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
