# build_query_index.py

import os
import joblib
import re
from typing import List
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# === المسارات اليدوية (حدّدها بنفسك)
QUERIES_PATH = r"E:\information_test\datasets\antique_qa\test\queries.txt"
TFIDF_MODEL_DIR = r"E:\information_test\saved_models\antique_qa\queries"
INVERTED_INDEX_DIR = r"E:\information_test\saved_index\antique_qa\queries"

# === تأكد من وجود المجلدات
os.makedirs(TFIDF_MODEL_DIR, exist_ok=True)
os.makedirs(INVERTED_INDEX_DIR, exist_ok=True)


def load_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f if line.strip()]
    return lines


def build_tfidf(queries: List[str]):
    vectorizer = TfidfVectorizer(lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(queries)
    return vectorizer, tfidf_matrix


def build_inverted_index(queries: List[str]):
    inverted = defaultdict(set)
    for doc_id, query in enumerate(queries):
        tokens = query.split()
        for token in tokens:
            clean_token = re.sub(r'\W+', '', token)
            if clean_token:
                inverted[clean_token].add(doc_id)
    return {term: list(doc_ids) for term, doc_ids in inverted.items()}


def save_all(queries, vectorizer, tfidf_matrix, inverted_index):
    joblib.dump(queries, os.path.join(TFIDF_MODEL_DIR, "queries_raw.joblib"))
    joblib.dump(vectorizer, os.path.join(TFIDF_MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(TFIDF_MODEL_DIR, "tfidf_matrix.joblib"))
    joblib.dump(inverted_index, os.path.join(INVERTED_INDEX_DIR, "inverted_index.joblib"))


def main():
    print("📥 Loading queries...")
    queries = load_queries(QUERIES_PATH)

    print("🔧 Building TF-IDF model...")
    vectorizer, tfidf_matrix = build_tfidf(queries)

    print("🔍 Building inverted index...")
    inverted_index = build_inverted_index(queries)

    print("💾 Saving models and index...")
    save_all(queries, vectorizer, tfidf_matrix, inverted_index)

    print("✅ Done! All data saved successfully.")


if __name__ == "__main__":
    main()
