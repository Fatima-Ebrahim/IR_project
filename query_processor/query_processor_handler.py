# # handlers/query_processor_handler.py
# todo it was working using old text handler
# import os
# import joblib
# import numpy as np
# from typing import Dict

# from text_processing.text_processing_handler import TextProcessor, process_text_pipeline
# from utils.config import OUTPUT_DIR
# from tfidf_representation.tfidf_handler import TfIdfHandler
# class QueryProcessorHandler:
#     """
#     يعالج عمليات تحميل النماذج ومعالجة الاستعلامات.
#     """
#     def __init__(self, dataset_name: str, model_type: str):
#         self.dataset_name = dataset_name
#         self.model_type = model_type
#         self.text_processor = TextProcessor()
#         self._load_vectorizer()

#     def _get_vectorizer_path(self) -> str:
#         """
#         يبني المسار لملف الـ vectorizer بناءً على الهيكلية الجديدة.
#         """
#         # هذا المنطق صحيح ويتوافق مع الهيكلية الجديدة
#         model_specific_dir = os.path.join(OUTPUT_DIR, self.dataset_name, self.model_type)
        
#         # اسم الملف موحد الآن وهو 'vectorizer.joblib'
#         return os.path.join(model_specific_dir, "vectorizer.joblib")

#     def _load_vectorizer(self):
#         """تحميل الـ vectorizer أو نموذج التضمين فقط."""
#         vectorizer_path = self._get_vectorizer_path()
#         print(f"📂 Loading vectorizer from: {vectorizer_path}")
#         self.vectorizer = joblib.load(vectorizer_path)
#         print("✅ Vectorizer loaded successfully.")

#     def process(self, query: str) -> np.ndarray:
#         """الدالة الرئيسية: تعالج الاستعلام وتحوله إلى متجه."""
#         print(f"📝 Processing query for model '{self.model_type}': '{query}'")
        
#         # ===== يبدأ التعديل هنا =====

#         if self.model_type == 'tfidf':
#             # الإضافة الجديدة: معالجة الاستعلام بشكل منفصل لغرض الطباعة فقط
#             # الـ vectorizer سيقوم بنفس هذه العملية داخلياً
#             processed_query_display = process_text_pipeline(query, self.text_processor)
#             print(f"✅ Query after processing: '{processed_query_display}'")
            
#             query_vector = self.vectorizer.transform([query])
#             return query_vector.toarray().tolist() # تحويل لمصفوفة كثيفة ثم قائمة لإرسالها كـ JSON

#         elif self.model_type == 'bert':
#             processed_query = process_text_pipeline(query, self.text_processor)
            
#             # الإضافة الجديدة: طباعة الاستعلام بعد معالجته
#             print(f"✅ Query after processing: '{processed_query}'")

#             query_vector = self.vectorizer.encode([processed_query], normalize_embeddings=True)
#             return query_vector.tolist() # تحويل لـ list لإرسالها كـ JSON
        
#         # ===== ينتهي التعديل هنا =====
        
#         raise NotImplementedError("Model type not implemented")
# handlers/query_processor_handler.py
# handlers/query_processor_handler.py
import os
import joblib
import numpy as np
from typing import Dict, List

# --- استيراد الكلاس فقط، وليس الدالة ---
from text_processing.text_processing_handler import TextProcessingHandler
# handlers/query_processor_handler.py
import os
import joblib
import numpy as np
from typing import Dict, List

from text_processing.text_processing_handler import TextProcessingHandler
from utils.config import OUTPUT_DIR
from utils.logger_config import logger

class QueryProcessorHandler:
    """
    Processes a raw query into a vector and returns the spell-corrected
    version of the query for display purposes.
    """
    # --- الآن يستقبل كائن المعالج الجاهز ---
    def __init__(self, dataset_name: str, model_type: str, text_processor: TextProcessingHandler):
        self.dataset_name = dataset_name
        self.model_type = model_type
        # --- استخدام المعالج الذي تم تمريره ---
        self.text_processor = text_processor
        self.vectorizer = self._load_vectorizer()

    def _get_vectorizer_path(self) -> str:
        model_specific_dir = os.path.join(OUTPUT_DIR, self.dataset_name, self.model_type)
        return os.path.join(model_specific_dir, "vectorizer.joblib")

    def _load_vectorizer(self):
        vectorizer_path = self._get_vectorizer_path()
        logger.info(f"📂 Loading vectorizer from: {vectorizer_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}. Please build the representation first.")
        return joblib.load(vectorizer_path)

    def process(self, query: str) -> Dict[str, any]:
        """
        Processes a raw query and returns a dictionary containing the
        spell-corrected query for display, and the final query vector.
        """
        logger.info(f"📝 Processing query for model '{self.model_type}': '{query}'")
        
        # 1. الحصول على النص المصحح إملائياً فقط (للعرض)
        corrected_query_for_display = self.text_processor.get_spell_corrected_text(query)
        logger.info(f"✅ Spell-corrected query for display: '{corrected_query_for_display}'")

        # 2. الحصول على النص المعالج بالكامل (لإنشاء المتجه)
        fully_processed_query = self.text_processor._process_single_text(query)
        logger.info(f"✅ Fully processed query for vectorization: '{fully_processed_query}'")

        query_vector = []
        if not fully_processed_query:
            # Handle empty query after processing
            if hasattr(self.vectorizer, 'get_sentence_embedding_dimension'): # BERT
                dim = self.vectorizer.get_sentence_embedding_dimension()
                query_vector = np.zeros((1, dim)).tolist()
            else: # TF-IDF
                dim = len(self.vectorizer.vocabulary_)
                query_vector = np.zeros((1, dim)).tolist()
        else:
            # 3. تحويل النص المعالج بالكامل إلى متجه
            if self.model_type == 'tfidf':
                vector = self.vectorizer.transform([fully_processed_query])
                query_vector = vector.toarray().tolist()
            elif self.model_type == 'bert':
                vector = self.vectorizer.encode([fully_processed_query], normalize_embeddings=True)
                query_vector = vector.tolist()
            else:
                 raise NotImplementedError(f"Model type '{self.model_type}' not implemented.")

        # 4. إرجاع القاموس بالنتائج
        return {
            "corrected_query_display": corrected_query_for_display,
            "query_vector": query_vector
        }
