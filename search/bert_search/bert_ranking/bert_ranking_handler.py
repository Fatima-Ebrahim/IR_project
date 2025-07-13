# handlers/bert_ranking_handler.py
import os
import joblib
import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity

import utils.config as config
from database.database_handler import DatabaseHandler
from utils.logger_config import logger

class BertRankingHandler:
   
    def __init__(self, dataset_name: str, db_handler: DatabaseHandler):
        self.dataset_name = dataset_name
        self.db_handler = db_handler
        self.assets = {}
        self._load_assets()

    def _get_asset_paths(self) -> Dict[str, str]:
        model_dir = os.path.join(config.OUTPUT_DIR, self.dataset_name, "bert")
        return {
            "matrix": os.path.join(model_dir, "matrix.joblib"),
            "ids": os.path.join(model_dir, "doc_ids_map.joblib")
        }

    def _load_assets(self):
        paths = self._get_asset_paths()
        logger.info(f"📂 Loading ranking assets for BERT...")
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Asset file not found at: {path}")
            self.assets[name] = joblib.load(path)
        logger.info("✅ BERT ranking assets loaded successfully.")

    def rank(self, query_vector: List[float], top_k: int) -> List[Dict[str, any]]:
       
        doc_matrix = self.assets['matrix']
        doc_ids = self.assets['ids']
        
        query_vector_reshaped = np.array(query_vector).reshape(1, -1)

    
        similarities = cosine_similarity(query_vector_reshaped, doc_matrix)
        scores = similarities.flatten()

    
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        top_doc_ids = [doc_ids[i] for i in top_k_indices]
        top_scores = [scores[i] for i in top_k_indices]

        logger.info(f"   - Fetching content for top {len(top_doc_ids)} documents...")
        documents_content = self.db_handler.find_documents_by_ids(top_doc_ids)

        results = []
        for i, doc_id in enumerate(top_doc_ids):
            document_data = documents_content.get(str(doc_id), {})
            results.append({
                "doc_id": document_data.get('doc_id', str(doc_id)),
                "score": float(top_scores[i]),
                "document_text": document_data.get('raw_text', 'Content not found.')
            })
            
       
        return results
