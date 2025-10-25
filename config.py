import pathlib

class Config:

    # -----------------------------
    # 資料夾路徑
    # -----------------------------
    BASE_DIR = pathlib.Path(__file__).parent
    PDF_DIR = BASE_DIR / "data/pdfs"
    VECTOR_DIR = BASE_DIR / "data/vectors"
    VECTOR_CHUNK_DB = VECTOR_DIR / "chunks"             # chunk 向量資料庫
    VECTOR_QUESTION_DB = VECTOR_DIR / "questions"      # question 向量資料庫
    QUESTION_DIR = BASE_DIR / "data/generated_questions" # 原始問題 JSON 存放

    # -----------------------------
    # 模型設定
    # -----------------------------
    OLLAMA_HOST = "http://localhost:11434"

    EMBEDDING_MODEL = "bge-m3"                      # Ollama embedding 模型
    LLM_MODEL = "llama3.2:1b"                       # Ollama LLM 模型
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"      # huggingface reranker模型

    # -----------------------------
    # 檢索參數
    # -----------------------------
    VECTOR_TOP_K = 50                               # 向量檢索回傳的初始chunk數量
    MID_TOP_M = 20                                  # 混合排序後回傳的chunk數量 (我將其設定為較小的 5 份作為範例)
    FINAL_TOP_M = 5                                 # reranker後最終輸出的chunk數量

    ALPHA = 0.6                                     # 向量相似度權重
    BETA = 0.4                                      # BM25 關鍵詞分數權重 

    # -----------------------------
    # Chunk 分割參數
    # -----------------------------
    # UnstructuredPDFLoader 切完超出 CHUNK_SIZE 才會切
    CHUNK_SIZE = 4000                               # 字元數參考，實際依 tokenizer
    CHUNK_OVERLAP = 200                             # chunk 重疊字元