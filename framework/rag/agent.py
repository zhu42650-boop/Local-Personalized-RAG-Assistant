from typing import List

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from rag.chat import answer_question


def build_tools(retriever, llm: ChatOpenAI) -> List:
    @tool("rag_qa")
    def rag_qa(question: str) -> str:
        """Use local knowledge base to answer the question."""
        return answer_question(question, retriever, llm)

    @tool("refresh_index")
    def refresh_index(_: str) -> str:
        """Refresh local knowledge base index."""
        return "Index refresh is not wired yet. Please run the ingest pipeline."

    return [rag_qa, refresh_index]


def build_agent(llm: ChatOpenAI, tools: List):
    system = SystemMessage(content="You are a research assistant. Use tools when needed.")
    return create_agent(model=llm, tools=tools, system_prompt=system)
