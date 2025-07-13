# handlers/tfidf_handler.py
# todo كان ميشتغل قبل تعديل البتيكت بروسيسر  
# import os
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from typing import List, Dict, Any

# import utils.config as config
# # (تعديل) استيراد الكلاس بالاسم الصحيح
# from text_processing.text_processing_handler import TextProcessor, process_text_pipeline
# from utils.logger_config import logger

# def _simple_tokenizer(text: str) -> List[str]:
#     """
#     A named function to replace the lambda for tokenization, making the model serializable.
#     """
#     return text.split()

# class TfIdfHandler:
#     """
#     Handles TF-IDF vectorization by applying a full, advanced processing pipeline
#     on RAW text during vectorization.
#     """
#     def __init__(self):
#         # إنشاء نسخة من معالج النصوص المتقدم لاستخدامه في الدوال المخصصة
#         logger.info("Initializing TfIdfHandler with a new TextProcessor instance.")
#         # (تعديل) استخدام اسم الكلاس الصحيح
#         self.text_processor = TextProcessor()
#         self.vectorizer = None
#         self.tfidf_matrix = None
#         self.doc_ids = None

#     def _custom_preprocessor(self, text: str) -> str:
#         """
#         Custom preprocessor that will be passed to TfidfVectorizer.
#         It applies the full advanced text processing pipeline.
#         """
#         # استخدام دالة الـ pipeline التي تستدعي بدورها كل خطوات المعالجة
#         return process_text_pipeline(text, self.text_processor)

#     def _get_model_paths(self, dataset_name: str) -> Dict[str, str]:
#         """
#         Builds the paths for the model files.
#         """
#         model_specific_dir = os.path.join(config.OUTPUT_DIR, dataset_name, "tfidf")
#         os.makedirs(model_specific_dir, exist_ok=True)
#         return {
#             "vectorizer": os.path.join(model_specific_dir, "vectorizer.joblib"),
#             "matrix": os.path.join(model_specific_dir, "matrix.joblib"),
#             "ids": os.path.join(model_specific_dir, "doc_ids_map.joblib")
#         }

#     def build_representation(self, raw_documents: List[Dict[str, Any]]):
#         """
#         Builds the TF-IDF representation from RAW documents using the custom preprocessor.
#         """
#         if not raw_documents:
#             raise ValueError("Cannot build representation from an empty list of documents.")

#         raw_texts = [doc['raw_text'] for doc in raw_documents]
#         self.doc_ids = [doc['id'] for doc in raw_documents]

#         logger.info("Configuring TfidfVectorizer with custom preprocessor...")

#         self.vectorizer = TfidfVectorizer(
#             preprocessor=self._custom_preprocessor,
#             tokenizer=_simple_tokenizer, # (تعديل) استخدام الدالة المعرفة بدلاً من lambda
#             token_pattern=None,
#             max_df=0.90,
#             min_df=5
#         )

#         self.tfidf_matrix = self.vectorizer.fit_transform(raw_texts)
#         logger.info("TF-IDF matrix built successfully using custom processing functions.")
#         logger.info(f"Shape of the matrix: {self.tfidf_matrix.shape}")

#     def save_representation(self, dataset_name: str):
#         """Saves the TF-IDF model files."""
#         if self.vectorizer is None or self.tfidf_matrix is None or self.doc_ids is None:
#             raise ValueError("Representation not built yet. Call build_representation() first.")
        
#         paths = self._get_model_paths(dataset_name)
#         joblib.dump(self.vectorizer, paths["vectorizer"])
#         logger.info(f"Vectorizer saved to: {paths['vectorizer']}")
#         joblib.dump(self.tfidf_matrix, paths["matrix"])
#         logger.info(f"TF-IDF matrix saved to: {paths['matrix']}")
#         joblib.dump(self.doc_ids, paths["ids"])
#         logger.info(f"Document IDs mapping saved to: {paths['ids']}")
# tfidf_representation/tfidf_handler.py
# tfidf_representation/tfidf_handler.py
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any

from utils import config
from text_processing.text_processing_handler import TextProcessingHandler
from utils.logger_config import logger

def _simple_tokenizer(text: str) -> List[str]:
    
    return text.split()

class TfIdfHandler:
    
    def __init__(self):
        # --- (التعديل) إنشاء نسخة من الكلاس الجديد بالاسم الصحيح ---
        logger.info("Initializing TfIdfHandler with a new TextProcessingHandler instance.")
        self.text_processor = TextProcessingHandler()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = None

    def _custom_preprocessor(self, text: str) -> str:
        
        return self.text_processor._process_single_text(text)

    def _get_model_paths(self, dataset_name: str) -> Dict[str, str]:
       
        model_specific_dir = os.path.join(config.OUTPUT_DIR, dataset_name, "tfidf")
        os.makedirs(model_specific_dir, exist_ok=True)
        return {
            "vectorizer": os.path.join(model_specific_dir, "vectorizer.joblib"),
            "matrix": os.path.join(model_specific_dir, "matrix.joblib"),
            "ids": os.path.join(model_specific_dir, "doc_ids_map.joblib")
        }

    def build_representation(self, raw_documents: List[Dict[str, Any]]):
        
        if not raw_documents:
            raise ValueError("Cannot build representation from an empty list of documents.")

        raw_texts = [doc['raw_text'] for doc in raw_documents]
        self.doc_ids = [doc['id'] for doc in raw_documents]

        logger.info("Configuring TfidfVectorizer with custom preprocessor...")

        self.vectorizer = TfidfVectorizer(
            preprocessor=self._custom_preprocessor,
            tokenizer=_simple_tokenizer,
            token_pattern=None,
            max_df=0.90,
            min_df=5
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(raw_texts)
        logger.info("TF-IDF matrix built successfully using custom processing functions.")
        logger.info(f"Shape of the matrix: {self.tfidf_matrix.shape}")

    def save_representation(self, dataset_name: str):
        if self.vectorizer is None or self.tfidf_matrix is None or self.doc_ids is None:
            raise ValueError("Representation not built yet. Call build_representation() first.")
        
        paths = self._get_model_paths(dataset_name)
        joblib.dump(self.vectorizer, paths["vectorizer"])
        logger.info(f"Vectorizer saved to: {paths['vectorizer']}")
        joblib.dump(self.tfidf_matrix, paths["matrix"])
        logger.info(f"TF-IDF matrix saved to: {paths['matrix']}")
        joblib.dump(self.doc_ids, paths["ids"])
        logger.info(f"Document IDs mapping saved to: {paths['ids']}")
