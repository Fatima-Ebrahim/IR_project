import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

from utils import config
from utils.logger_config import logger
from database.database_handler import DatabaseHandler


class SemanticExpansionHandler:
    def __init__(self, dataset_name: str, db_config: dict, model_name: str = 'all-mpnet-base-v2'):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.db = DatabaseHandler(db_config)
        self.db.connect()

        logger.info(f"[SemanticExpansion] Initializing for dataset: {dataset_name}")
        self.embedding_model = SentenceTransformer(model_name)

        self._load_assets()

    def _get_asset_paths(self) -> Dict[str, str]:
        base_dir = os.path.join(config.OUTPUT_DIR, self.dataset_name, "bert")
        return {
            "doc_embeddings": os.path.join(base_dir, "matrix.joblib"),
            "doc_ids": os.path.join(base_dir, "doc_ids_map.joblib"),
            "tfidf_vectorizer": os.path.join(config.OUTPUT_DIR, self.dataset_name, "tfidf", "vectorizer.joblib")
        }

    def _load_assets(self):
        paths = self._get_asset_paths()
        logger.info("Loading BERT and TF-IDF assets...")

        if not all(os.path.exists(p) for p in paths.values()):
            raise FileNotFoundError("Missing required embedding/vectorizer files.")

        self.doc_embeddings = joblib.load(paths["doc_embeddings"])
        self.doc_ids = joblib.load(paths["doc_ids"])
        self.vectorizer: TfidfVectorizer = joblib.load(paths["tfidf_vectorizer"])
        self.vocab = self.vectorizer.get_feature_names_out()

        logger.info(f"Loaded {len(self.doc_ids)} doc embeddings and TF-IDF vectorizer.")

    def _split_into_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in text.split('.') if s.strip()]

    def expand(self, query: str, top_k: int = 5, threshold: float = 0.4) -> Dict[str, Any]:
        if not query.strip():
            return {"expanded_query": "", "expansion_terms": []}

        logger.info(f"[SemanticExpansion] Expanding query: '{query}'")
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)

        similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
        candidate_indices = np.where(similarities >= threshold)[0]

        if len(candidate_indices) == 0:
            return {"expanded_query": query, "expansion_terms": []}

        top_indices = sorted(candidate_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
        matched_ids = [int(self.doc_ids[i]) for i in top_indices]

        logger.info(f"[SemanticExpansion] Retrieving texts for {len(matched_ids)} documents...")
        docs_by_id = self.db.find_documents_by_ids(matched_ids)
        matched_texts = [docs_by_id[str(doc_id)]["raw_text"] for doc_id in matched_ids if str(doc_id) in docs_by_id]

        if not matched_texts:
            logger.warning("No text content retrieved from DB.")
            return {"expanded_query": query, "expansion_terms": []}

        
        top_sentences = []
        for text in matched_texts:
            sentences = self._split_into_sentences(text)
            sentence_embeddings = self.embedding_model.encode(sentences, normalize_embeddings=True)
            sent_sims = cosine_similarity(query_embedding, sentence_embeddings).flatten()

            top_sentences.extend([
                s for sim, s in sorted(zip(sent_sims, sentences), reverse=True) if sim >= 0.5
            ])

        if not top_sentences:
            logger.warning("No similar sentences found.")
            return {"expanded_query": query, "expansion_terms": []}

        doc_term_matrix = self.vectorizer.transform(top_sentences)
        term_scores = np.asarray(doc_term_matrix.sum(axis=0)).flatten()

        top_term_indices = term_scores.argsort()[::-1]
        original_tokens = set(query.lower().split())

        expansion_terms = []
        for idx in top_term_indices:
            term = self.vocab[idx]
            if term not in original_tokens:
                expansion_terms.append(term)
            if len(expansion_terms) >= top_k:
                break

        expanded_terms = original_tokens.union(expansion_terms)
        expanded_query = " ".join(sorted(expanded_terms))

        logger.info(f"[SemanticExpansion] Expanded query: '{expanded_query}'")
        return {
            "expanded_query": expanded_query,
            "expansion_terms": expansion_terms
        }
