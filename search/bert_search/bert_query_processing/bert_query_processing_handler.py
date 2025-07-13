
 # todo    لهيك هو ابطا شوي الكودين شغالين الفرق أن الأول أسرع أما الثاني فهو يعتمد عل التواصل مع خدمة 
    
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from utils.logger_config import logger

class BertQueryProcessorHandler:
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', preprocess_url: str = None):
        logger.info(f"Initializing BertQueryProcessorHandler with model: {model_name}")
        self.preprocess_url = preprocess_url 

        try:
            self.model = SentenceTransformer(model_name)
            logger.info("✅ SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load SentenceTransformer model: {e}")
            raise

    def process_query_to_vector(self, query: str) -> List[float]:
        if not query or not isinstance(query, str):
            logger.warning("⚠️ Received an empty or invalid query.")
            return []

        try:
            response = requests.post(self.preprocess_url, json={"query": query})
            response.raise_for_status()
            processed_query = response.json().get("processed_query", "")
        except Exception as e:
            logger.error(f"❌ Error calling query_preprocessor_service: {e}")
            raise RuntimeError("Failed to preprocess query.") from e

        logger.info(f"🧠 Processed query: '{processed_query}'")
        query_embedding = self.model.encode(processed_query, normalize_embeddings=True)
        return query_embedding.tolist()