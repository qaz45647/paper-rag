"""

改由在splitter.py統一執行

"""

import re
from langchain_community.document_loaders import PyPDFLoader
from modules.utils import log



def load_pdf(pdf_path: str):
    """讀取 PDF 並清理雜訊"""
    log(f"載入 PDF：{pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    cleaned_docs = []

    for d in documents:
        text = d.page_content

        # 保留換行符號，讓章節正則能抓到行首
        text = re.sub(r"[ \t]+", " ", text)  # 去多餘空白
        text = re.sub(r"\n{2,}", "\n", text)  # 多重換行壓縮成一行

        cleaned_docs.append({
            "page": d.metadata.get("page", 0),
            "text": text.strip()
        })

    log(f"文件載入完成，共 {len(cleaned_docs)} 頁")
    return cleaned_docs
