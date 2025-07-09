# handlers/hybrid_handler.py
import os
import joblib
import numpy as np
from collections import defaultdict
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

# Import necessary components from your project
from utils.config import OUTPUT_DIR
from database.database_handler import DatabaseHandler
from text_processing.text_processing_handler import TextProcessor, process_text_pipeline

class HybridSearchHandler:
    """
    Handles hybrid search by fetching results from both TF-IDF and BERT models,
    fusing them using RRF, and retrieving full document content.
    """
    # def __init__(self, dataset_name: str, db_handler: DatabaseHandler):
        # self.dataset_name = dataset_name
        # self.db_handler = db_handler
        # self.text_processor = TextProcessor()
        # self.RRF_K = 60  # Constant for the RRF formula

        # # Load assets for both models
        # self.tfidf_assets = self._load_model_assets('tfidf')
        # self.bert_assets = self._load_model_assets('bert')
    def __init__(self, db_handler: DatabaseHandler, tfidf_assets: dict, bert_assets: dict):
        self.db_handler = db_handler
        self.text_processor = TextProcessor()
        self.RRF_K = 60
        
        # It directly uses the data passed to it instead of loading from disk
        self.tfidf_assets = tfidf_assets
        self.bert_assets = bert_assets


    def _load_model_assets(self, model_type: str) -> Dict:
        """Loads the vectorizer, matrix, and doc_ids_map for a given model type."""
        model_dir = os.path.join(OUTPUT_DIR, self.dataset_name, model_type)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found for {model_type} at {model_dir}")
        
        return {
            "vectorizer": joblib.load(os.path.join(model_dir, "vectorizer.joblib")),
            "matrix": joblib.load(os.path.join(model_dir, "matrix.joblib")),
            "doc_ids_map": joblib.load(os.path.join(model_dir, "doc_ids_map.joblib"))
        }

    def _get_ranks(self, model_type: str, query: str) -> Dict[int, int]:
        """Performs a search for one model and returns a {doc_id: rank} dictionary."""
        assets = self.tfidf_assets if model_type == 'tfidf' else self.bert_assets
        vectorizer = assets["vectorizer"]
        matrix = assets["matrix"]
        doc_ids_map = assets["doc_ids_map"]

        # Vectorize the query
        if model_type == 'tfidf':
            processed_query = process_text_pipeline(query, self.text_processor)
            query_vector = vectorizer.transform([processed_query])
        else: # bert
            query_vector = vectorizer.encode([query])

        # Get scores
        scores = cosine_similarity(query_vector, matrix).flatten()
        
        # Get top 100 indices for ranking
        top_indices = np.argsort(scores)[-100:][::-1]

        # Create a map of {primary_key: rank}
        rank_map = {doc_ids_map[idx]: rank + 1 for rank, idx in enumerate(top_indices) if scores[idx] > 0}
        return rank_map

    def search(self, query: str, top_k: int) -> List[Dict]:
        """Performs the full hybrid search and retrieval process."""
        # 1. Get ranked lists from both models
        tfidf_ranks = self._get_ranks('tfidf', query)
        bert_ranks = self._get_ranks('bert', query)

        # 2. Fuse ranks using RRF
        all_doc_ids = set(tfidf_ranks.keys()).union(set(bert_ranks.keys()))
        rrf_scores = defaultdict(float)

        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in tfidf_ranks:
                score += 1 / (self.RRF_K + tfidf_ranks[doc_id])
            if doc_id in bert_ranks:
                score += 1 / (self.RRF_K + bert_ranks[doc_id])
            rrf_scores[doc_id] = score
        
        # 3. Get the top-k document IDs from the fused scores
        sorted_doc_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        
        if not sorted_doc_ids:
            return []

        # 4. Fetch the document content from the database [CRUCIAL STEP]
        documents_content = self.db_handler.find_documents_by_ids(sorted_doc_ids)

        # 5. Build the final, rich response
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