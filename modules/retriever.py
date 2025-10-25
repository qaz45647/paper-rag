import numpy as np
import re
import torch
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from config import Config
from modules.utils import log
from modules.embedder import get_embedder


def extract_score(model_output: str) -> float:
    """嘗試從模型輸出字串中提取最後出現的一個浮點數分數。"""
    all_matches = re.findall(r"[-+]?\d*\.\d+|\d+", model_output.strip())
    if all_matches:
        last_match = all_matches[-1]
        try:
            return float(last_match)
        except ValueError:
            log(f"[WARN] 無法解析最後一個配對項 '{last_match}' 為分數: {model_output}")
            return -9999.0
    return -9999.0


# --- 主檢索函數 ---
def hybrid_search(query: str, filename: str = None) -> List[str]:
    """執行混合檢索 (向量 + BM25) 並使用 Reranker 重新排序"""
    log(f"執行混合檢索：{query} (文件過濾: {filename or '無'})")

    embedder = get_embedder()

    # 1. 載入向量資料庫
    vector_db_chunks = Chroma(
        persist_directory=str(Config.VECTOR_CHUNK_DB),
        embedding_function=embedder
    )

    # 2. 檔案過濾設定
    chroma_filter = {"filename": filename} if filename else {}

    # 3. 向量檢索（返回距離分數）
    v_results = vector_db_chunks.similarity_search_with_score(
        query,
        k=Config.VECTOR_TOP_K,
        filter=chroma_filter
    )

    if not v_results:
        log("向量檢索結果為空，結束。")
        return []

    combined: List[Tuple[Document, float]] = v_results
    log(f"向量檢索共獲取 {len(combined)} 筆結果。")

    # 4. BM25 建立
    docs_text = [d[0].page_content for d in combined]
    tokenized_corpus = [t.split() for t in docs_text]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    # 5. 分數正規化處理 ----------------------------------------

    # --- BM25 Min-Max normalization ---
    if len(bm25_scores) > 0 and (max(bm25_scores) - min(bm25_scores)) > 1e-6:
        b_min, b_max = min(bm25_scores), max(bm25_scores)
        bm25_scores_norm = (bm25_scores - b_min) / (b_max - b_min)
    else:
        bm25_scores_norm = np.zeros_like(bm25_scores)

    # --- 向量距離轉相似度並正規化 ---
    v_scores = np.array([vscore for _, vscore in combined])
    if len(v_scores) > 0 and (max(v_scores) - min(v_scores)) > 1e-6:
        # Step 1: 距離越小越相似，先反轉
        v_scores_sim = (max(v_scores) - v_scores)
        # Step 2: 再 Min-Max normalize
        v_scores_norm = (v_scores_sim - v_scores_sim.min()) / (v_scores_sim.max() - v_scores_sim.min())
    else:
        v_scores_norm = np.ones_like(v_scores)

    # -----------------------------------------------------------

    # 6. 混合分數計算
    scored: List[Tuple[Document, float]] = []
    for (doc, _), vscore_norm, bscore_norm in zip(combined, v_scores_norm, bm25_scores_norm):
        score = Config.ALPHA * vscore_norm + Config.BETA * bscore_norm
        scored.append((doc, score))

    # 排序與初步截斷
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:Config.MID_TOP_M]
    log(f"混合檢索完成，取前 {len(scored)} 筆結果 準備進行 Reranker")

    # 7. Reranker 模型排序 --------------------------------------
    reranked: List[Tuple[Document, float]] = []
    log("使用 Hugging Face 模型進行 Rerank")

    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.RERANKER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(Config.RERANKER_MODEL)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        pairs = [(query, doc.page_content) for doc, _ in scored]

        with torch.no_grad():
            inputs = tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            scores = outputs.logits.squeeze().detach().cpu().numpy()

        for (doc, _), score in zip(scored, scores):
            reranked.append((doc, float(score)))

        reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
        final_results = reranked[:Config.FINAL_TOP_M]

        log(f"Reranker 完成，最終取 {len(final_results)} 筆結果")
        for i, (doc, score) in enumerate(final_results[:5], 1):
            preview = doc.page_content.strip().replace("\n", " ")[:100]
            log(f"{i}. {preview}... (score={score:.4f})")

    except Exception as e:
        log(f"[WARN] Reranker 失敗，使用混合檢索結果。錯誤：{e}")
        final_results = scored[:Config.FINAL_TOP_M]

    # 8. 輸出最終結果
    return [r[0].page_content for r in final_results]
