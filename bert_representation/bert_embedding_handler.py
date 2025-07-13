
import os
import joblib
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from utils.logger_config import logger

class BertEmbeddingHandler:

    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
       
        logger.info(f"Initializing BertEmbeddingHandler with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise

    def generate_and_save_embeddings(
        self,
        documents: List[Dict[str, Any]],
        vectorizer_path: str,
        matrix_path: str,
        doc_map_path: str
    ) -> Tuple[int, int]:
        
        if not documents:
            raise ValueError("Document list is empty. Cannot generate embeddings.")

       
        texts = [doc.get('processed_text', '') for doc in documents]
        doc_ids = [doc['id'] for doc in documents]
        
        logger.info(f"Starting to generate embeddings for {len(texts)} documents...")
        
        
        doc_embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        embedding_dim = doc_embeddings.shape[1]
        logger.info("Embeddings generated successfully.")

        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        
        joblib.dump(self.model, vectorizer_path)
        logger.info(f"Vectorizer (model) saved to: {vectorizer_path}")
        
        joblib.dump(doc_embeddings, matrix_path)
        logger.info(f"Embeddings matrix saved to: {matrix_path}")
        
        joblib.dump(doc_ids, doc_map_path)
        logger.info(f"Document ID map saved to: {doc_map_path}")

        return len(doc_ids), embedding_dim
