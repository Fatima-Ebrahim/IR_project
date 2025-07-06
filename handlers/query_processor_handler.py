# handlers/query_processor_handler.py

import os
import joblib
import numpy as np
from typing import Dict

from TextProcessing import TextProcessor, process_text_pipeline
from utils.config import OUTPUT_DIR
from handlers.tfidf_handler import TfIdfHandler
class QueryProcessorHandler:
    """
    يعالج عمليات تحميل النماذج ومعالجة الاستعلامات.
    """
    def __init__(self, dataset_name: str, model_type: str):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.text_processor = TextProcessor()
        self._load_vectorizer()

    def _get_vectorizer_path(self) -> str:
        """
        يبني المسار لملف الـ vectorizer بناءً على الهيكلية الجديدة.
        """
        # هذا المنطق صحيح ويتوافق مع الهيكلية الجديدة
        model_specific_dir = os.path.join(OUTPUT_DIR, self.dataset_name, self.model_type)
        
        # اسم الملف موحد الآن وهو 'vectorizer.joblib'
        return os.path.join(model_specific_dir, "vectorizer.joblib")

    def _load_vectorizer(self):
        """تحميل الـ vectorizer أو نموذج التضمين فقط."""
        vectorizer_path = self._get_vectorizer_path()
        print(f"📂 Loading vectorizer from: {vectorizer_path}")
        self.vectorizer = joblib.load(vectorizer_path)
        print("✅ Vectorizer loaded successfully.")

    def process(self, query: str) -> np.ndarray:
        """الدالة الرئيسية: تعالج الاستعلام وتحوله إلى متجه."""
        print(f"📝 Processing query for model '{self.model_type}': '{query}'")
        
        # ===== يبدأ التعديل هنا =====

        if self.model_type == 'tfidf':
            # الإضافة الجديدة: معالجة الاستعلام بشكل منفصل لغرض الطباعة فقط
            # الـ vectorizer سيقوم بنفس هذه العملية داخلياً
            processed_query_display = process_text_pipeline(query, self.text_processor)
            print(f"✅ Query after processing: '{processed_query_display}'")
            
            query_vector = self.vectorizer.transform([query])
            return query_vector.toarray().tolist() # تحويل لمصفوفة كثيفة ثم قائمة لإرسالها كـ JSON

        elif self.model_type == 'bert':
            processed_query = process_text_pipeline(query, self.text_processor)
            
            # الإضافة الجديدة: طباعة الاستعلام بعد معالجته
            print(f"✅ Query after processing: '{processed_query}'")

            query_vector = self.vectorizer.encode([processed_query], normalize_embeddings=True)
            return query_vector.tolist() # تحويل لـ list لإرسالها كـ JSON
        
        # ===== ينتهي التعديل هنا =====
        
        raise NotImplementedError("Model type not implemented")