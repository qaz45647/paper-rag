import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

try:
    from app import demo 
except ImportError as e:
    print(f"無法匯入 frontend/app.py 中的 'demo'：{e}")
    print("請確認 frontend/app.py 已正確定義 Gradio Blocks 或 Interface 並命名為 'demo'")
    sys.exit(1)


if __name__ == "__main__":
    print("啟動 Paper RAG Gradio 介面...")
    demo.launch(
        server_name="0.0.0.0",  
        server_port=7860,       
        debug=True              
    )
    print("Gradio 介面已停止。")