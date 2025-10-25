## paper-rag

本專案結合 LangChain 與 Ollama，打造專門針對學術論文與技術文件的 RAG 系統。採用結構化切割與混合檢索檢索的方式精準的從 PDF 內容提取相關知識，並針對文件內容的問答功能。

![流程.png](/img/flow.png)

---

## 安裝

### 透過 GitHub 安裝

**Clone the repo:**

```bash
git clone https://github.com/qaz45647/paper-rag.git
```
```bash
cd paper-rag
```
**Create a conda environment:**
```bash
conda create -n paper-rag python=3.11
conda activate paper-rag
```
**Use pip to install required packages:**
```bash
pip install -r requirements.txt
```
**Install Ollama:**
從官方網站下載並安裝Ollama : https://ollama.com/download

**Download Models via Ollama Command Line:**
```bash
ollama pull llama3:8b 
ollama pull bge-m3
```
**Usage:**
```bash
python main.py
```
## 更換模型
若要更換模型（例如將 embedding 模型切換為支援中文的 bge-large-zh-v1.5），請前往 config.py 修改相關變數。
- 由於 LLM 與 Embedding 模型 皆透過 Ollama 運行，更換後需額外執行指令以下載模型。
- Reranker 模型無需執行額外指令。
```bash
    EMBEDDING_MODEL = "bge-m3"                      # Ollama embedding 模型
    LLM_MODEL = "llama3.2:1b"                       # Ollama LLM 模型
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"      # huggingface reranker模型
```
```bash
ollama pull XXXX
```
