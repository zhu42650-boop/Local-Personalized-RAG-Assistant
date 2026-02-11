RAG_PROMPT_TEMPLATE = """你是一个科研协作助手。优先使用给定的本地知识库内容回答问题。
要求：
1) 若上下文包含相关信息，务必基于上下文作答，并标注来自上下文的要点
2) 若上下文不足或无关，可以基于你的通用知识进行回答
3) 回答要简洁、条理清晰

对话历史（可选）：
{history}

上下文：
{context}

问题：
{question}
"""


def build_prompt(context: str, question: str, history: str = "") -> str:
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question, history=history)
