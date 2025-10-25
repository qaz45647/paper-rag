"""

因為針對每個chunk生成問題成本太高，較不適合會頻繁上傳PDF的系統，因此廢棄此方法。
電腦太爛，跑一個chunk要20秒，花一個小時以上建立一篇論文的向量庫不現實:(

"""
from langchain_ollama import OllamaLLM
from modules.utils import log, save_json
from config import Config

PROMPT_TEMPLATE = """
請設計三個問題，問題能從以下段落找到答案。
範例：
- 段落：「愛因斯坦於 1921 年獲得諾貝爾物理學獎，以表彰他對理論物理學的貢獻，尤其是他對光電效應定律的發現，這也是他在科學界受到最大榮譽的年份。」
    - 愛因斯坦什麼時候獲得諾貝爾獎？
    - 他是因為什麼成就而獲獎？
    - 哪一年愛因斯坦在科學界受到最大榮譽？
------------------
段落：
{chunk}
"""

def generate_questions(chunks, save_path=None):
    """使用 LLM 為每個 chunk 生成問題"""
    llm = OllamaLLM(model=Config.LLM_MODEL)
    results = []

    for c in chunks:
        prompt = PROMPT_TEMPLATE.format(chunk=c["content"])
        try:
            res = llm.invoke(prompt)
            questions = [q.strip(" -•") for q in res.split("\n") if len(q.strip()) > 3]
            data = {
                "chunk_id": c["id"],
                "page": c["page"],
                "questions": questions
            }
            results.append(data)
        except Exception as e:
            log(f"生成問題錯誤 chunk={c['id']}：{e}")

    if save_path:
        save_json(results, save_path)
        log(f"問題已儲存：{save_path}")
    return results
