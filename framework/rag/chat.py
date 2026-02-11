from typing import List, Optional

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


def answer_question(
    question: str, retriever, llm: ChatOpenAI, chat_history: Optional[List[dict]] = None
) -> str:
    docs = _retrieve_docs(retriever, question)
    context = _join_context(docs)
    history = _join_history(chat_history)
    prompt = build_prompt(context=context, question=question, history=history)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
