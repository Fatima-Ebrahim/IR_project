
import os
import joblib
import numpy as np
from typing import Dict, List
from collections import defaultdict

import utils.config as config
from database.database_handler import DatabaseHandler
from text_processing.text_processing_handler import TextProcessingHandler
from utils.logger_config import logger

class TfidfRankingHandler:
   
    def __init__(self, dataset_name: str, db_handler: DatabaseHandler):
        self.dataset_name = dataset_name
        self.db_handler = db_handler
        
        
        self.text_processor = TextProcessingHandler()
        
        self.assets = {}
        self._load_assets()

    def _get_asset_paths(self) -> Dict[str, str]:
        
        index_dir = os.path.join(config.OUTPUT_DIR.replace("saved_models", "saved_indexes"), self.dataset_name, "inverted_index")
        model_dir = os.path.join(config.OUTPUT_DIR, self.dataset_name, "tfidf")
        return {
            "inverted_index": os.path.join(index_dir, "inverted_index.joblib"),
            "doc_lengths": os.path.join(index_dir, "doc_lengths.joblib"),
            "vectorizer": os.path.join(model_dir, "vectorizer.joblib")
        }

    def _load_assets(self):
        
        paths = self._get_asset_paths()
        logger.info(f"📂 Loading ranking assets for TF-IDF...")
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Asset file not found at: {path}")
            self.assets[name] = joblib.load(path)
        logger.info("✅ TF-IDF ranking assets loaded successfully.")

    def rank(self, query: str, top_k: int) -> List[Dict[str, any]]:
       
        logger.info(f"🔍 Ranking documents for query: '{query}' using inverted index.")
        
        processed_query = self.text_processor._process_single_text(query)
        query_terms = processed_query.split()

        if not query_terms:
            return []

        scores = defaultdict(float)
        vectorizer = self.assets['vectorizer']
        inverted_index = self.assets['inverted_index']
        doc_lengths = self.assets['doc_lengths']
        avg_doc_length = np.mean(list(doc_lengths.values())) if doc_lengths else 0
        idf = vectorizer.idf_
        vocab = vectorizer.vocabulary_
        k1 = 1.5
        b = 0.75

        for term in query_terms:
            if term in inverted_index:
                term_idf = idf[vocab.get(term, -1)]
                if term_idf == -1: continue

                for doc_id, tf in inverted_index[term].items():
                   
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_lengths.get(doc_id, avg_doc_length) / avg_doc_length))
                    scores[doc_id] += term_idf * (numerator / denominator)

        sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
        
        top_doc_ids = sorted_doc_ids[:top_k]

        logger.info(f"   - Fetching content for top {len(top_doc_ids)} documents...")
        documents_content = self.db_handler.find_documents_by_ids(top_doc_ids)

        results = []
        for doc_id in top_doc_ids:
            document_data = documents_content.get(str(doc_id), {})
            results.append({
                "doc_id": document_data.get('doc_id', str(doc_id)),
                "score": float(scores[doc_id]),
                "document_text": document_data.get('raw_text', 'Content not found.')
            })
            
        return results
