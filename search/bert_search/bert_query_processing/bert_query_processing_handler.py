# handlers/bert_query_processing_handler.py
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from typing import List

# # استيراد معالج النصوص لضمان معالجة الاستعلام بنفس طريقة المستندات
# from text_processing.text_processing_handler import TextProcessor, process_text_pipeline
# from utils.logger_config import logger

# class BertQueryProcessorHandler:
#     """
#     Handles loading the BERT model and processing a raw query into a vector embedding.
#     """
#     def __init__(self, model_name: str = 'all-mpnet-base-v2'):
#         """
#         Initializes the handler by loading the text processor and the
#         pre-trained SentenceTransformer model.
#         """
#         logger.info(f"Initializing BertQueryProcessorHandler with model: {model_name}")
#         self.text_processor = TextProcessor()
        
#         # تحميل نموذج BERT. سيتم تنزيله تلقائياً في المرة الأولى.
#         try:
#             self.model = SentenceTransformer(model_name)
#             logger.info("SentenceTransformer model loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
#             raise

#     def process_query_to_vector(self, query: str) -> List[float]:
#         """
#         Processes a raw query string and encodes it into a dense vector embedding.
        
#         Args:
#             query (str): The raw query text from the user.

#         Returns:
#             List[float]: The query embedding as a list of floats.
#         """
#         if not query or not isinstance(query, str):
#             logger.warning("Received an empty or invalid query.")
#             return []

#         # 1. تطبيق نفس خطوات المعالجة الأولية على الاستعلام
#         logger.info(f"Processing query: '{query}'")
#         processed_query = process_text_pipeline(query, self.text_processor)
#         logger.info(f"Query after processing: '{processed_query}'")

#         # 2. تحويل النص المعالج إلى متجه باستخدام نموذج BERT
#         # normalize_embeddings=True يجعل حساب تشابه الجيب (cosine similarity) أكثر كفاءة لاحقاً
#         query_embedding = self.model.encode(
#             processed_query,
#             normalize_embeddings=True
#         )
        
#         # تحويل الناتج إلى قائمة قياسية من الأرقام العشرية (floats)
#         return query_embedding.tolist()

# search/bert_search/bert_query_processing/bert_query_processing_handler.py
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# --- (التعديل) استيراد الكلاس الجديد بالاسم الصحيح ---
from text_processing.text_processing_handler import TextProcessingHandler
from utils.logger_config import logger

class BertQueryProcessorHandler:
    """
    Handles loading the BERT model and processing a raw query into a vector embedding.
    """
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initializes the handler by loading the text processor and the
        pre-trained SentenceTransformer model.
        """
        logger.info(f"Initializing BertQueryProcessorHandler with model: {model_name}")
        # --- (التعديل) إنشاء نسخة من الكلاس الجديد ---
        self.text_processor = TextProcessingHandler()
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise

    def process_query_to_vector(self, query: str) -> List[float]:
        """
        Processes a raw query string and encodes it into a dense vector embedding.
        """
        if not query or not isinstance(query, str):
            logger.warning("Received an empty or invalid query.")
            return []

        # 1. --- (التعديل) استدعاء الدالة الصحيحة من الكلاس ---
        logger.info(f"Processing query: '{query}'")
        processed_query = self.text_processor._process_single_text(query)
        logger.info(f"Query after processing: '{processed_query}'")

        # 2. تحويل النص المعالج إلى متجه باستخدام نموذج BERT
        query_embedding = self.model.encode(
            processed_query,
            normalize_embeddings=True
        )
        
        return query_embedding.tolist()
