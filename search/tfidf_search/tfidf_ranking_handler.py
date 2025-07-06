# handlers/tfidf_ranking_handler.py
import os
import joblib
import numpy as np
from typing import Dict, List
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import utils.config as config
from database.database_handler import DatabaseHandler
# (مهم) استيراد معالج النصوص لاستخدامه في معالجة الاستعلام
from text_processing.text_processing_handler import TextProcessor, process_text_pipeline
from utils.logger_config import logger

class TfidfRankingHandler:
    """
    Ranks documents for TF-IDF queries efficiently using an inverted index,
    avoiding loading the large TF-IDF matrix into memory.
    """
    def __init__(self, dataset_name: str, db_handler: DatabaseHandler):
        self.dataset_name = dataset_name
        self.db_handler = db_handler
        
        # Initialize the text processor needed for the query
        self.text_processor = TextProcessor()
        
        # Load the necessary assets: vectorizer and inverted index
        self.assets = {}
        self._load_assets()

    def _get_asset_paths(self) -> Dict[str, str]:
        """Builds the paths for the required assets."""
        # For TF-IDF, we need the inverted index and the vectorizer (for the query)
        index_dir = os.path.join(config.OUTPUT_DIR.replace("saved_models", "saved_indexes"), self.dataset_name, "inverted_index")
        model_dir = os.path.join(config.OUTPUT_DIR, self.dataset_name, "tfidf")
        return {
            "inverted_index": os.path.join(index_dir, "inverted_index.joblib"),
            "doc_lengths": os.path.join(index_dir, "doc_lengths.joblib"),
            "vectorizer": os.path.join(model_dir, "vectorizer.joblib")
        }

    def _load_assets(self):
        """Loads the vectorizer and inverted index from disk."""
        paths = self._get_asset_paths()
        logger.info(f"📂 Loading ranking assets for TF-IDF...")
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Asset file not found at: {path}")
            self.assets[name] = joblib.load(path)
        logger.info("✅ TF-IDF ranking assets loaded successfully.")

    def rank(self, query: str, top_k: int) -> List[Dict[str, any]]:
        """
        Ranks documents using the inverted index.
        """
        logger.info(f"🔍 Ranking documents for query: '{query}' using inverted index.")
        
        # 1. Process the query using the same pipeline as the documents
        processed_query = process_text_pipeline(query, self.text_processor)
        query_terms = processed_query.split()

        if not query_terms:
            return []

        # 2. Transform the processed query into a TF-IDF vector
        vectorizer = self.assets['vectorizer']
        query_vector = vectorizer.transform([processed_query])

        # 3. Retrieve candidate documents from the inverted index
        candidate_docs = set()
        inverted_index = self.assets['inverted_index']
        for term in query_terms:
            if term in inverted_index:
                candidate_docs.update(inverted_index[term].keys())
        
        if not candidate_docs:
            return []
        
        candidate_docs = list(candidate_docs)
        logger.info(f"   - Found {len(candidate_docs)} candidate documents from inverted index.")

        # 4. Reconstruct a small TF-IDF matrix for only the candidate documents
        # This is more memory-efficient than loading the full matrix.
        # We can get the raw texts of candidate docs and transform them.
        # NOTE: For even larger scale, scoring would happen without reconstructing vectors.
        # This is a good middle-ground.
        
        # For simplicity and speed, let's use a direct scoring method with the index.
        scores = defaultdict(float)
        query_feature_names = vectorizer.get_feature_names_out()
        query_weights = {term: query_vector[0, vectorizer.vocabulary_[term]] for term in query_terms if term in vectorizer.vocabulary_}

        for term, q_weight in query_weights.items():
            if term in inverted_index:
                for doc_id, tf in inverted_index[term].items():
                    # Simplified TF-IDF-like score
                    # A proper implementation would also use IDF, which is part of the vectorizer.
                    idf = vectorizer.idf_[vectorizer.vocabulary_[term]]
                    scores[doc_id] += (q_weight * tf * idf)

        # 5. Sort documents by score
        sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
        
        top_doc_ids = sorted_doc_ids[:top_k]

        # 6. Fetch document content from the database
        logger.info(f"   - Fetching content for top {len(top_doc_ids)} documents...")
        documents_content = self.db_handler.find_documents_by_ids(top_doc_ids)

        # 7. Build final results
        results = []
        for doc_id in top_doc_ids:
            document_data = documents_content.get(str(doc_id), {})
            results.append({
                "doc_id": document_data.get('doc_id', str(doc_id)),
                "score": float(scores[doc_id]),
                "document_text": document_data.get('raw_text', 'Content not found.')
            })
            
        return results
