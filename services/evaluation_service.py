# services/evaluation_service.py

import pandas as pd
import numpy as np
import os
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal, List, Dict

# استيراد المسارات والإعدادات الجديدة من ملف config
from utils.config import DATASETS_BASE_DIR, EVALUATION_DATA_MAPPING

# --- تعريف دوال تحميل بيانات التقييم (تبقى كما هي) ---
def load_qrels(qrels_path: str) -> Dict[str, set]:
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"Qrels file not found at: {qrels_path}")
    qrels = {}
    with open(qrels_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, doc_id, relevance = parts[0], parts[2], int(parts[3])
                if relevance > 0:
                    qrels.setdefault(qid, set()).add(doc_id)
    return qrels

def load_queries_from_txt(queries_path: str) -> Dict[str, str]:
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries file not found at: {queries_path}")
    queries = {}
    with open(queries_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                qid, query_text = parts
                queries[qid] = query_text
    return queries

# --- تعريف دوال مقاييس الأداء (تبقى كما هي) ---
def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / k if k > 0 else 0.0

def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / len(relevant) if relevant else 0.0

def average_precision(retrieved: List[str], relevant: set) -> float:
    if not relevant: return 0.0
    score, hits = 0.0, 0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant)

def mean_reciprocal_rank(all_results: Dict[str, List[str]], qrels: Dict[str, set]) -> float:
    rr_list = []
    for qid, retrieved in all_results.items():
        relevant = qrels.get(qid, set())
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                rr_list.append(1 / (i + 1))
                break
        else:
            rr_list.append(0)
    return np.mean(rr_list) if rr_list else 0.0

# --- كود خدمة FastAPI ---
app = FastAPI(title="Evaluation Service", version="1.4.0")

SEARCH_GATEWAY_URL = "http://127.0.0.1:8000"

class EvaluationRequest(BaseModel):
    dataset_name: str = Field(..., example="antique_qa")
    model_type: Literal['tfidf', 'bert']

class EvaluationResponse(BaseModel):
    dataset_name: str
    model_type: str
    num_queries: int
    MAP: float
    MRR: float
    avg_precision_at_10: float
    avg_recall_at_10: float

@app.post("/evaluate", response_model=EvaluationResponse, tags=["Performance Evaluation"])
async def evaluate_model(request: EvaluationRequest):
    # هذا هو اسم مجموعة البيانات الذي تستخدمه نماذجك (مثل antique_qa)
    model_dataset_name = request.dataset_name
    model_type = request.model_type
    
    try:
        # ===== يبدأ التعديل هنا =====
        # 1. البحث عن اسم المجلد الصحيح من القاموس الذي أنشأناه
        eval_folder_name = EVALUATION_DATA_MAPPING.get(model_dataset_name)
        if not eval_folder_name:
            raise FileNotFoundError(f"Evaluation mapping not found for dataset '{model_dataset_name}' in config.py")

        # 2. بناء المسارات باستخدام اسم المجلد الصحيح
        print(f"Loading evaluation data for '{model_dataset_name}' from folder '{eval_folder_name}'...")
        eval_test_path = os.path.join(DATASETS_BASE_DIR, eval_folder_name, "test")
        qrels_path = os.path.join(eval_test_path, "qrels")
        queries_path = os.path.join(eval_test_path, "queries.txt")
        # ===== ينتهي التعديل هنا =====
        
        qrels = load_qrels(qrels_path)
        queries = load_queries_from_txt(queries_path)
        print(f"Loaded {len(queries)} queries for evaluation.")

        all_results = {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            for qid, query_text in queries.items():
                print(f"Processing Q{qid}...")
                
                # إرسال الطلب إلى بوابة البحث باستخدام اسم النموذج (antique_qa)
                search_payload = {"query": query_text, "top_k": 50}
                
                response = await client.post(
                    f"{SEARCH_GATEWAY_URL}/search/{model_dataset_name}/{model_type}",
                    json=search_payload
                )

                response.raise_for_status()
                retrieved_docs = response.json()
                all_results[qid] = [doc['doc_id'] for doc in retrieved_docs]

        print("Calculating final metrics...")
        p_list, r_list, ap_list = [], [], []
        for qid in queries:
            retrieved = all_results.get(qid, [])
            relevant = qrels.get(qid, set())
            
            p_at_10 = precision_at_k(retrieved, relevant, 10)
            r_at_10 = recall_at_k(retrieved, relevant, 10)
            ap = average_precision(retrieved, relevant)
            
            p_list.append(p_at_10)
            r_list.append(r_at_10)
            ap_list.append(ap)

        map_score, mrr_score, avg_p10, avg_r10 = np.mean(ap_list), mean_reciprocal_rank(all_results, qrels), np.mean(p_list), np.mean(r_list)

        return EvaluationResponse(
            dataset_name=model_dataset_name, model_type=model_type, num_queries=len(queries),
            MAP=map_score, MRR=mrr_score, avg_precision_at_10=avg_p10, avg_recall_at_10=avg_r10,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, httpx.RequestError, httpx.HTTPStatusError) as e:
        detail_msg = f"An error occurred during evaluation: {e}"
        if isinstance(e, httpx.HTTPStatusError):
            detail_msg += f" | Response: {e.response.text}"
        raise HTTPException(status_code=500, detail=detail_msg)

