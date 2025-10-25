from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from modules.utils import log
from config import Config


def get_embedder():
    """回傳 Ollama embedding 模型，從 Config 讀取"""
    return OllamaEmbeddings(model=Config.EMBEDDING_MODEL)


def store_vectors(chunks, collection_name="chunks"):
    """將 chunks  向量化並存入 Chroma"""
    log(f"建立向量庫：{collection_name}")

    if not chunks:
        log("傳入的 chunks 為空，已跳過。")
        return

    # 取得當前要處理的 PDF 檔案名稱
    current_filename = chunks[0]["metadata"].get("filename")
    log(f"檢查檔案：{current_filename}")

    db_path = (
        Config.VECTOR_CHUNK_DB
        if collection_name == "chunks"
        else Config.VECTOR_QUESTION_DB
    )
    embedder = get_embedder()

    # 1. 讀取既有向量資料庫
    vectorstore = Chroma(
        embedding_function=embedder,
        persist_directory=str(db_path),
    )

    # 2. 讀取現有資料庫中的所有 metadata
    existing_metadatas = vectorstore.get(include=["metadatas"]).get("metadatas", [])
    existing_filenames = set(
        m.get("filename") for m in existing_metadatas if m and m.get("filename")
    )

    # 3. 檢查是否已存在該 PDF
    if current_filename in existing_filenames:
        log(f"檔案 {current_filename} 已存在於向量庫中，跳過向量化。")
        return vectorstore

    # ----------------------------------------------------
    # 4. 若未重複，則進行向量化
    texts = [c["content"] for c in chunks]
    metadatas = [
        {
            "id": c["id"],
            "page": c["page"],
            "title": c["title"],
            "filename": c["metadata"].get("filename"),
            "category": c["metadata"].get("category"),
        }
        for c in chunks
    ]

    # 5. 加入新資料
    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    # 6. 寫入硬碟
    vectorstore.persist()
    log(f"向量化完成，共 {len(chunks)} 筆資料（檔案：{current_filename}）")
    return vectorstore
