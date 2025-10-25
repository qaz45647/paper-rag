import os
#from modules.loader import load_pdf
from modules.splitter import split_documents
#from modules.question_generator import generate_questions
from modules.embedder import store_vectors
from modules.query_transformer import transform_query
from modules.retriever import hybrid_search
from modules.qa_chain import generate_answer
from config import Config



# --- 1. 定義 PDF 檔案路徑與檔名 ---
PDF_FILE_NAME = "sample.pdf"
PDF_FULL_PATH = os.path.join(Config.PDF_DIR, PDF_FILE_NAME)


# -----------------------------
# 2. 切塊
# -----------------------------
#chunks = split_documents(pages)
chunks = split_documents(PDF_FULL_PATH) # 使用完整路徑
print(f"[測試] 切塊完成，共 {len(chunks)} 個 chunk")


# -----------------------------
# 3. 向量化
# -----------------------------
store_vectors(chunks, collection_name="chunks")
print("[測試] 向量化完成 (chunks + questions)")

"""
# -----------------------------
# Query Transformation 測試
# -----------------------------
#好像沒必要 停用
user_query = "這篇論文主要在做什麼研究？"
transformed_query = transform_query(user_query)
print(f"[測試] Query Transformation：\n原始: {user_query}\n改寫: {transformed_query}")
"""

# -----------------------------
# 4. 混合檢索
# -----------------------------
transformed_query = "How does the RAG-Anything framework process and represent the results it extracts from different content types (e.g., text, diagrams, tables, and mathematical expressions) to ensure fidelity of content and structural context?"

retrieved_docs = hybrid_search(query=transformed_query, filename=PDF_FILE_NAME)


# -----------------------------
# 5. 問答生成
# -----------------------------
answer = generate_answer(transformed_query, retrieved_docs)
print(f"[測試] 最終回答：\n{answer}")
