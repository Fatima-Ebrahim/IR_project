# handlers/query_indexer_handler.py

import os
import re
import joblib
from collections import defaultdict
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import config

class QueryIndexerHandler:
    def __init__(
        self,
        dataset_name: str,
        base_data_dir: str = None,
        save_model_dir: str = None,
        save_index_dir: str = None,
        lowercase_only: bool = True
    ):
        self.dataset_name = dataset_name
        self.base_data_dir = base_data_dir or config.DATASETS_BASE_DIR
        self.save_model_dir = save_model_dir or config.OUTPUT_DIR
        self.save_index_dir = save_index_dir or config.OUTPUT_DIR.replace("saved_models", "saved_indexes")
        self.lowercase_only = lowercase_only

        self.queries_path = os.path.join(self.base_data_dir, dataset_name, "test", "queries.txt")
        self.model_dir = os.path.join(self.save_model_dir, dataset_name, "queries")
        self.index_dir = os.path.join(self.save_index_dir, dataset_name, "queries")

        self.raw_queries: List[str] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.inverted_index: Dict[str, List[int]] = {}

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

    def load_queries(self):
        if not os.path.exists(self.queries_path):
            raise FileNotFoundError(f"queries.txt not found at: {self.queries_path}")
        with open(self.queries_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if self.lowercase_only:
            self.raw_queries = [line.lower() for line in lines]
        else:
            self.raw_queries = lines

    def build_tfidf(self):
        if not self.raw_queries:
            self.load_queries()

        self.vectorizer = TfidfVectorizer(lowercase=False)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.raw_queries)

    def build_inverted_index(self):
        if not self.raw_queries:
            self.load_queries()

        inverted = defaultdict(set)
        for doc_id, query in enumerate(self.raw_queries):
            tokens = query.split()
            for token in tokens:
                clean_token = re.sub(r'\W+', '', token)
                if clean_token:
                    inverted[clean_token].add(doc_id)
        self.inverted_index = {term: list(ids) for term, ids in inverted.items()}

    def save_all(self, save_tfidf: bool = True, save_inverted_index: bool = True):
        if save_tfidf:
            joblib.dump(self.vectorizer, os.path.join(self.model_dir, "tfidf_vectorizer.joblib"))
            joblib.dump(self.tfidf_matrix, os.path.join(self.model_dir, "tfidf_matrix.joblib"))
            joblib.dump(self.raw_queries, os.path.join(self.model_dir, "queries_raw.joblib"))

        if save_inverted_index:
            joblib.dump(self.inverted_index, os.path.join(self.index_dir, "inverted_index.joblib"))
