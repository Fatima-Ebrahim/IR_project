import os
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import List, Dict
from utils import config
from utils.logger_config import logger


class QuerySuggestionHandler:
    def __init__(self, dataset_name: str, db_handler=None):
        self.dataset_name = dataset_name
        self.db_handler = db_handler  # احتياطي

        # المسارات
        self.query_dir = os.path.join(config.OUTPUT_DIR, dataset_name, "queries")
        self.index_dir = os.path.join(
            config.OUTPUT_DIR.replace("saved_models", "saved_indexes"),
            dataset_name,
            "queries"
        )

        # تحميل الملفات
        self.raw_queries: List[str] = joblib.load(os.path.join(self.query_dir, "queries_raw.joblib"))
        self.vectorizer = joblib.load(os.path.join(self.query_dir, "tfidf_vectorizer.joblib"))
        self.tfidf_matrix = joblib.load(os.path.join(self.query_dir, "tfidf_matrix.joblib"))

        inverted_path = os.path.join(self.index_dir, "inverted_index.joblib")
        self.inverted_index = joblib.load(inverted_path) if os.path.exists(inverted_path) else None

    def get_suggestions(self, partial_query: str, top_k: int = 10) -> Dict[str, List[str]]:
        if not partial_query:
            return {"next_words": [], "full_queries": []}

        partial_query = partial_query.strip().lower()
        result_queries = []

        if self.inverted_index:
            # استخدام الفهرس المعكوس
            tokens = partial_query.split()
            matched_ids = set()
            for token in tokens:
                matched_ids.update(self.inverted_index.get(token, []))
            result_queries = [self.raw_queries[i] for i in matched_ids]
        else:
            # استخدام TF-IDF
            query_vec = self.vectorizer.transform([partial_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            result_queries = [self.raw_queries[i] for i in top_indices if similarities[i] > 0.1]

        # إعطاء أولوية للاستعلامات التي تبدأ بالجزء المكتوب
        priority_matches = []
        partial_query_lc = partial_query.lower()

        for q in result_queries:
            clean_q = q.split('\t', 1)[-1].lower() if '\t' in q else q.lower()
            if clean_q.startswith(partial_query_lc):
                priority_matches.append(q)

        other_matches = [q for q in result_queries if q not in priority_matches]
        result_queries = priority_matches + other_matches

        # استخراج الكلمات التالية
        next_words = []
        for q in result_queries:
            clean_q = q.split('\t', 1)[-1].lower() if '\t' in q else q.lower()
            tokens = clean_q.split()
            partial_tokens = partial_query_lc.split()

            for i in range(len(tokens) - len(partial_tokens)):
                if tokens[i:i + len(partial_tokens)] == partial_tokens:
                    next_word = tokens[i + len(partial_tokens)]
                    if re.match(r'^[a-zA-Z0-9\-]{2,}$', next_word):
                        next_words.append(next_word)
                    break

        sorted_next_words = [word for word, _ in Counter(next_words).most_common(10)]

        # إزالة معرفات البداية إن وُجدت
        cleaned_full_queries = []
        for q in result_queries:
            if '\t' in q:
                _, clean = q.split('\t', 1)
            else:
                clean = q
            cleaned_full_queries.append(clean)

        unique_full_queries = list(dict.fromkeys(cleaned_full_queries))[:10]

        return {
            "next_words": sorted_next_words,
            "full_queries": unique_full_queries
        }
