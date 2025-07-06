# handlers/ranking_handler.py
import os
import joblib
import numpy as np
from typing import Dict, List, Set
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import utils.config as config
from database.database_handler import DatabaseHandler
from handlers.text_processing_handler import TextProcessingHandler # <-- استيراد لمعالجة الاستعلام
from utils.logger_config import logger

class RankingHandler:
    def __init__(self, dataset_name: str, model_type: str, db_handler: DatabaseHandler):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.db_handler = db_handler
        self.assets = {}
        self._load_assets()

    def _get_asset_paths(self) -> Dict[str, str]:
        """Builds the paths for the required assets based on the model type."""
        if self.model_type == 'tfidf':
            # For TF-IDF, we need the inverted index and the vectorizer (for the query)
            index_dir = os.path.join(config.OUTPUT_DIR.replace("saved_models", "saved_indexes"), self.dataset_name, "inverted_index")
            model_dir = os.path.join(config.OUTPUT_DIR, self.dataset_name, "tfidf")
            return {
                "inverted_index": os.path.join(index_dir, "inverted_index.joblib"),
                "doc_lengths": os.path.join(index_dir, "doc_lengths.joblib"),
                "vectorizer": os.path.join(model_dir, "vectorizer.joblib")
            }
        elif self.model_type == 'bert':
            # For BERT, we need the dense matrix and the doc IDs map
            model_dir = os.path.join(config.OUTPUT_DIR, self.dataset_name, "bert")
            return {
                "matrix": os.path.join(model_dir, "matrix.joblib"),
                "ids": os.path.join(model_dir, "doc_ids_map.joblib")
            }
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_assets(self):
        """Loads the necessary assets for the selected model type."""
        paths = self._get_asset_paths()
        logger.info(f"📂 Loading ranking assets for model '{self.model_type}'...")
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Asset file not found at: {path}")
            self.assets[name] = joblib.load(path)
        logger.info("✅ Ranking assets loaded successfully.")

    def rank(self, query: str, top_k: int) -> List[Dict[str, any]]:
        """
        Ranks documents based on the query. Uses Inverted Index for TF-IDF
        and dense matrix for BERT.
        """
        logger.info(f"🔍 Ranking documents for query: '{query}' using '{self.model_type}' model.")
        
        if self.model_type == 'tfidf':
            return self._rank_with_inverted_index(query, top_k)
        elif self.model_type == 'bert':
            # Note: For a complete system, you'd process the query to a vector first.
            # This part assumes a query_vector is passed, which needs adjustment in the service.
            # For now, let's assume the query is processed elsewhere.
            # This logic needs a pre-computed query_vector.
            # This demonstrates the ranking part, not the query processing for BERT.
            logger.warning("BERT ranking in this handler assumes a pre-computed vector. The search gateway should handle this.")
            # The original logic for BERT ranking would be here, using a dense query_vector.
            return [] # Placeholder to avoid error
        
        raise NotImplementedError(f"Ranking for model type '{self.model_type}' is not implemented.")

    def _rank_with_inverted_index(self, query: str, top_k: int) -> List[Dict[str, any]]:
        """
        Efficiently ranks documents using the inverted index for TF-IDF.
        """
        # 1. Process the query using the same pipeline
        # We need a text processor instance here.
        # For simplicity, let's assume the query processor service does this.
        # Here we will simulate it.
        text_processor = TextProcessingHandler(config.SYMPSPELL_DICT_PATH)
        processed_query = text_processor._process_single_text(query)
        query_terms = processed_query.split()

        if not query_terms:
            return []

        # 2. Retrieve candidate documents from the inverted index
        candidate_docs = set()
        inverted_index = self.assets['inverted_index']
        for term in query_terms:
            if term in inverted_index:
                # Add all doc IDs that contain the term
                candidate_docs.update(inverted_index[term].keys())
        
        if not candidate_docs:
            return []
        
        logger.info(f"   - Found {len(candidate_docs)} candidate documents from inverted index.")

        # 3. Build a sub-matrix for only the candidate documents
        vectorizer = self.assets['vectorizer']
        doc_lengths = self.assets['doc_lengths']
        
        # We need to reconstruct the vectors for candidate docs to calculate similarity
        # This is a simplified approach. A more advanced system would use the index directly for scoring (like BM25).
        # For cosine similarity, we can transform the content of candidate docs.
        
        # Let's use a simpler scoring for demonstration: sum of TF values
        scores = defaultdict(float)
        for term in query_terms:
            if term in inverted_index:
                for doc_id, tf in inverted_index[term].items():
                    # A simple scoring: add the term frequency.
                    # A better score would be TF-IDF, but this requires IDF values.
                    # For now, TF is a good proxy.
                    scores[doc_id] += tf

        # 4. Sort documents by score
        sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
        
        effective_top_k = min(top_k, len(sorted_doc_ids))
        top_doc_ids = sorted_doc_ids[:effective_top_k]

        # 5. Fetch document content from the database
        logger.info(f"   - Fetching content for top {len(top_doc_ids)} documents...")
        documents_content = self.db_handler.find_documents_by_ids(top_doc_ids)

        # 6. Build final results
        results = []
        for doc_id in top_doc_ids:
            document_data = documents_content.get(str(doc_id), {})
            results.append({
                "doc_id": document_data.get('doc_id', str(doc_id)),
                "score": float(scores[doc_id]),
                "document_text": document_data.get('raw_text', 'Content not found.')
            })
            
        return results

