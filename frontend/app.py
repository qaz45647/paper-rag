import gradio as gr
import os
import shutil
from typing import List, Tuple

# ----------------------------------------------------
# 模組匯入
# ----------------------------------------------------
try:
    from modules.splitter import split_documents
    from modules.embedder import store_vectors
    from modules.retriever import hybrid_search
    from modules.qa_chain import generate_answer
    # from modules.query_transformer import transform_query
    from config import Config
except ImportError as e:
    print(f"模組匯入錯誤: {e}")
    class DummyConfig:
        PDF_DIR = "data/pdfs"
    Config = DummyConfig


# ----------------------------------------------------
# 工具函式
# ----------------------------------------------------
def get_processed_files():
    """取得目前已上傳並處理的 PDF 清單"""
    os.makedirs(Config.PDF_DIR, exist_ok=True)
    files = [f for f in os.listdir(Config.PDF_DIR) if f.endswith(".pdf")]
    if not files:
        return ["請先處理文件"]
    return files


# ----------------------------------------------------
# A. 文件處理流程
# ----------------------------------------------------
def process_pdf_file(pdf_file) -> str:
    """
    上傳後的 PDF 進行切塊與向量化
    """
    if pdf_file is None:
        return "錯誤：請先上傳 PDF 檔案。"

    pdf_file_name = os.path.basename(pdf_file.name)

    try:
        # 儲存上傳檔案到 Config 指定路徑
        os.makedirs(Config.PDF_DIR, exist_ok=True)
        target_path = os.path.join(Config.PDF_DIR, pdf_file_name)
        shutil.copy(pdf_file.name, target_path)

        # 切塊
        chunks = split_documents(target_path)
        if len(chunks) == 0:
            return f"警告：'{pdf_file_name}' 未能切出任何內容。"

        # 向量化
        store_vectors(chunks, collection_name="chunks")

        return f"檔案 '{pdf_file_name}' 處理完成，共切出 {len(chunks)} 個片段，已儲存至向量資料庫。"

    except Exception as e:
        return f"錯誤：處理檔案時發生問題：{e}"


def update_dropdown_after_upload(pdf_file):
    """
    重新整理 Dropdown 並自動選中新上傳的檔案
    """
    if pdf_file is None:
        return gr.Dropdown.update(choices=get_processed_files(), value=None)

    pdf_file_name = os.path.basename(pdf_file.name)
    files = get_processed_files()

    return gr.Dropdown.update(choices=files, value=pdf_file_name)


# ----------------------------------------------------
# B. 問答流程
# ----------------------------------------------------
def rag_query(history: List[Tuple[str, str]], query: str, pdf_file_name: str) -> str:
    if not query:
        return "請輸入您的問題。"
    if not pdf_file_name or pdf_file_name == "請先處理文件":
        return "請先上傳並處理 PDF 檔案。"

    try:
        # transformed_query = transform_query(query)
        transformed_query = query  # 暫時使用原始 query
        retrieved_docs = hybrid_search(query=query, filename=pdf_file_name)

        if not retrieved_docs:
            return "檢索失敗：未找到與問題相關的內容。"

        answer = generate_answer(transformed_query, retrieved_docs)
        return answer
    except Exception as e:
        return f"問答過程中發生錯誤：{e}"


# ----------------------------------------------------
# C. Gradio 介面
# ----------------------------------------------------
with gr.Blocks(title="Paper RAG 論文問答系統") as demo:
    gr.Markdown("#Paper RAG 論文問答系統")

    with gr.Row():
        with gr.Column(scale=1):
            # --- 文件上傳與處理區 ---
            gr.Markdown("## 步驟一：上傳與處理 PDF")
            pdf_upload = gr.File(
                label="上傳 PDF 論文",
                file_types=[".pdf"],
                type="filepath" if gr.__version__.startswith('4') else "file"
            )
            process_button = gr.Button("開始處理 (切塊 + 向量化)")
            process_output = gr.Textbox(
                label="處理結果",
                lines=3,
                interactive=False,
                value="等待上傳文件..."
            )

            # --- 文件選擇區 ---
            gr.Markdown("## 步驟二：選擇問答文件")
            file_dropdown = gr.Dropdown(
                label="選擇要問答的 PDF 檔案",
                choices=get_processed_files(),
                value=get_processed_files()[0] if get_processed_files() else None,
                interactive=True
            )

            # 綁定按鈕動作：處理後更新 dropdown
            process_button.click(
                fn=process_pdf_file,
                inputs=[pdf_upload],
                outputs=[process_output]
            ).then(
                fn=update_dropdown_after_upload,
                inputs=[pdf_upload],
                outputs=[file_dropdown]
            )

        with gr.Column(scale=2):
            # --- 問答區 ---
            gr.Markdown("## 步驟三：開始問答")

            chatbot = gr.Chatbot(
                label="RAG 問答結果",
                height=500,
                show_copy_button=True
            )
            msg = gr.Textbox(label="輸入你的問題...")
            clear_btn = gr.ClearButton([msg, chatbot], value="清空對話")

            # 問答邏輯
            def respond(history: List[Tuple[str, str]], query: str, pdf_file_name: str):
                history = history + [(query, None)]
                response = rag_query(history, query, pdf_file_name)
                history[-1] = (query, response)
                return history, ""

            msg.submit(
                fn=respond,
                inputs=[chatbot, msg, file_dropdown],
                outputs=[chatbot, msg]
            )
