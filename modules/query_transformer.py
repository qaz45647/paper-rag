from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from config import Config
from modules.utils import log

TRANSFORM_PROMPT = """
你是一位熟悉學術論文檢索的助理，
請將使用者的問題改寫成更正式、明確、具檢索意圖的學術問句。
只需輸出改寫後的問題。

使用者問題：
{query}
"""

def transform_query(user_query: str) -> str:
    try:
        llm = Ollama(model=Config.LLM_MODEL)
        prompt = PromptTemplate.from_template(TRANSFORM_PROMPT)
        reformulated = llm.invoke(prompt.format(query=user_query))
        cleaned = reformulated.strip()
        log(f"Query 改寫：{user_query} → {cleaned}")
        return cleaned
    except Exception as e:
        log(f"Query Transform 發生錯誤：{e}")
        return user_query
