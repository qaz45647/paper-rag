import os
import json
import nltk
from datetime import datetime
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.utils import log
from config import Config

# -----------------------------
# NLTK 模型檢查
# -----------------------------
for model_name in ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(model_name)
    except LookupError:
        log(f"[INFO] 缺少 {model_name}，正在下載...")
        nltk.download(model_name)


def split_documents(pdf_path: str, min_words=3):
    """使用 UnstructuredPDFLoader 自動解析 PDF 元素並切塊，並過濾垃圾 chunk。"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到 PDF 檔案：{pdf_path}")

    log(f"使用 UnstructuredPDFLoader 讀取 PDF：{pdf_path}")
    loader = UnstructuredPDFLoader(pdf_path, mode="elements")
    docs = loader.load()
    log(f"讀取完成，共 {len(docs)} 個元素")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    chunks = []
    for idx, doc in enumerate(docs):
        text = doc.page_content.strip()
        if not text:
            continue

        # 頁碼（僅作記錄，不進 title）
        page_num = doc.metadata.get("page_number", None) or "未知頁"

        # 從內容提取第一句話作為標題
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            title = sentences[0].strip() if sentences else text[:50]
        except Exception as e:
            log(f"[WARN] 解析第一句失敗：{e}")
            title = text[:50]  # fallback：取前 50 字

        # 分塊
        splits = splitter.split_text(text)
        for i, c in enumerate(splits):
            content_stripped = c.strip() # 先處理一次內容
            chunk = {
                "id": f"{idx}_{i}",
                "page": page_num,
                "title": title[:100],  # 限制長度
                "content": content_stripped,
                "metadata": doc.metadata
            }

            # -----------------------------
            # 過濾垃圾 chunk
            # -----------------------------
            if chunk["metadata"].get("category") == "UncategorizedText":
                continue
            if len(chunk["content"].split()) < min_words:
                continue

            chunks.append(chunk)

    log(f"完成切塊並過濾垃圾 chunk，共 {len(chunks)} 個有效 chunk")

    #去重
    unique_chunks = []
    seen_contents = set()
    
    for chunk in chunks:
        # 使用內容作為去重鍵
        content_key = chunk["content"]
        
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            unique_chunks.append(chunk)

    log(f"完成內容去重，最終保留 {len(unique_chunks)} 個 chunk")
    # -------------------------------------------------------------

    # 儲存 JSON 
    os.makedirs("data/chunks", exist_ok=True)
    output_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("data/chunks", f"{output_name}_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, ensure_ascii=False, indent=2)

    log(f"已保存 JSON：{output_path}")
    return unique_chunks # 回傳去重後的列表