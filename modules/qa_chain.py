from langchain_community.llms import Ollama
from config import Config
from modules.utils import log

PROMPT_TEMPLATE = """
You are a rigorous academic assistant.
Please answer the questions based on the context provided.
If the answer does not appear in the content, respond with: Not mentioned in the data.

────────────────────
Context:
{context}
────────────────────
question：
{query}
"""

def generate_answer(user_query, retrieved_docs):
    """
    user_query: 使用者問題
    retrieved_docs: 已檢索到的文件列表 (list of str)
    """
    context = "\n".join(retrieved_docs)

    llm = Ollama(model=Config.LLM_MODEL)
    prompt = PROMPT_TEMPLATE.format(context=context, query=user_query)
    answer = llm.invoke(prompt)
    log(f"回答完成")
    return answer.strip()
