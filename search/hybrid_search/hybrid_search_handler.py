# import os
# import joblib
# import numpy as np
# from collections import defaultdict
# from typing import List, Dict
# from sklearn.metrics.pairwise import cosine_similarity

# from utils.config import OUTPUT_DIR
# from database.database_handler import DatabaseHandler
# from text_processing.text_processing_handler import TextProcessingHandler

# class HybridSearchHandler:
   
#     def __init__(self, db_handler: DatabaseHandler, tfidf_assets: dict, bert_assets: dict):
#         self.db_handler = db_handler
   
#         self.text_processor = TextProcessingHandler()
#         self.RRF_K = 60
        
#         self.tfidf_assets = tfidf_assets
#         self.bert_assets = bert_assets

#     def _get_ranks(self, model_type: str, query: str) -> Dict[int, int]:
       
#         assets = self.tfidf_assets if model_type == 'tfidf' else self.bert_assets
#         vectorizer = assets["vectorizer"]
#         matrix = assets["matrix"]
#         doc_ids_map = assets["doc_ids_map"]

       
#         processed_query = self.text_processor._process_single_text(query)

#         if model_type == 'tfidf':
#             query_vector = vectorizer.transform([processed_query])
#         else: # bert
#             query_vector = vectorizer.encode([processed_query])

#         scores = cosine_similarity(query_vector, matrix).flatten()
        
#         top_indices = np.argsort(scores)[-100:][::-1]

#         rank_map = {doc_ids_map[idx]: rank + 1 for rank, idx in enumerate(top_indices) if scores[idx] > 0}
#         return rank_map

#     def search(self, query: str, top_k: int) -> List[Dict]:
        
#         tfidf_ranks = self._get_ranks('tfidf', query)
#         bert_ranks = self._get_ranks('bert', query)

       
#         all_doc_ids = set(tfidf_ranks.keys()).union(set(bert_ranks.keys()))
#         rrf_scores = defaultdict(float)

#         for doc_id in all_doc_ids:
#             score = 0.0
#             if doc_id in tfidf_ranks:
#                 score += 1 / (self.RRF_K + tfidf_ranks[doc_id])
#             if doc_id in bert_ranks:
#                 score += 1 / (self.RRF_K + bert_ranks[doc_id])
#             rrf_scores[doc_id] = score
        
#         sorted_doc_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        
#         if not sorted_doc_ids:
#             return []

       
#         documents_content = self.db_handler.find_documents_by_ids(sorted_doc_ids)

        
#         results = []
#         for doc_id in sorted_doc_ids:
#             content = documents_content.get(str(doc_id))
#             if content:
#                 results.append({
#                     "doc_id": content.get('doc_id'),
#                     "score": rrf_scores[doc_id],
#                     "document_text": content.get('raw_text', 'Content not found.')
#                 })
        
#         return results
# todo هذا الكود شغال كمان بس هو أبطأ 
import os
import joblib
import numpy as np
import requests
from collections import defaultdict
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

from utils.config import OUTPUT_DIR
from database.database_handler import DatabaseHandler
from utils.logger_config import logger

class HybridSearchHandler:
    
    def __init__(self, db_handler: DatabaseHandler, tfidf_assets: dict, bert_assets: dict, preprocess_url: str):
        self.db_handler = db_handler
        self.preprocess_url = preprocess_url  # مثال: 
        self.RRF_K = 60
        self.tfidf_assets = tfidf_assets
        self.bert_assets = bert_assets

    def _preprocess_query(self, raw_query: str) -> str:
        try:
            logger.info(f"🔁 Sending query to preprocessing service: {raw_query}")
            response = requests.post(self.preprocess_url, json={"query": raw_query})
            response.raise_for_status()
            processed_query = response.json().get("processed_query", "")
            if not processed_query:
                logger.warning("⚠️ Preprocessing service returned an empty processed query.")
            return processed_query
        except Exception as e:
            logger.error(f"❌ Failed to preprocess query via service: {e}")
            raise

    def _get_ranks(self, model_type: str, query: str) -> Dict[int, int]:
        assets = self.tfidf_assets if model_type == 'tfidf' else self.bert_assets
        vectorizer = assets["vectorizer"]
        matrix = assets["matrix"]
        doc_ids_map = assets["doc_ids_map"]

        processed_query = self._preprocess_query(query)

        if model_type == 'tfidf':
            query_vector = vectorizer.transform([processed_query])
        else:  # bert
            query_vector = vectorizer.encode([processed_query])

        scores = cosine_similarity(query_vector, matrix).flatten()
        top_indices = np.argsort(scores)[-100:][::-1]
        rank_map = {doc_ids_map[idx]: rank + 1 for rank, idx in enumerate(top_indices) if scores[idx] > 0}
        return rank_map

    def search(self, query: str, top_k: int) -> List[Dict]:
       
        tfidf_ranks = self._get_ranks('tfidf', query)
        bert_ranks = self._get_ranks('bert', query)

        all_doc_ids = set(tfidf_ranks.keys()).union(set(bert_ranks.keys()))
        rrf_scores = defaultdict(float)

        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in tfidf_ranks:
                score += 1 / (self.RRF_K + tfidf_ranks[doc_id])
            if doc_id in bert_ranks:
                score += 1 / (self.RRF_K + bert_ranks[doc_id])
            rrf_scores[doc_id] = score

        sorted_doc_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        if not sorted_doc_ids:
            return []

        documents_content = self.db_handler.find_documents_by_ids(sorted_doc_ids)

        results = []
        for doc_id in sorted_doc_ids:
            content = documents_content.get(str(doc_id))
            if content:
                results.append({
                    "doc_id": content.get('doc_id'),
                    "score": rrf_scores[doc_id],
                    "document_text": content.get('raw_text', 'Content not found.')
                })

        return results
