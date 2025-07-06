# handlers/inverted_index_handler.py
import os
import joblib
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm

# --- تعديل: استيراد ملف الإعدادات مباشرة ---
import  utils.config as config

class InvertedIndexHandler:
    """
    Builds, saves, and manages an inverted index.
    Reads the base output directory from the project's config file.
    """
    def __init__(self):
        # --- تعديل: تحديد مسار حفظ الفهارس ديناميكيًا من ملف config ---
        # يفترض أن المسار الأساسي للفهارس هو نفس مسار النماذج ولكن مع تغيير اسم المجلد
        if 'saved_models' in config.OUTPUT_DIR:
            self.indexes_base_dir = config.OUTPUT_DIR.replace("saved_models", "saved_indexes")
        else:
            self.indexes_base_dir = os.path.join(config.OUTPUT_DIR, "saved_indexes")
        
        self.index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_lengths: Dict[int, int] = {}
        # التأكد من وجود المجلد الأساسي
        os.makedirs(self.indexes_base_dir, exist_ok=True)
        print(f"Index files will be saved under: {self.indexes_base_dir}")

    def _get_index_paths(self, dataset_name: str, index_type: str) -> str:
        """
        Creates the full path for saving the index based on the desired structure.
        Structure: .../saved_indexes/<dataset_name>/<index_type>/
        """
        # --- تعديل: استخدام اسم مجموعة البيانات (dataset_name) كما هو مطلوب ---
        index_dir = os.path.join(self.indexes_base_dir, dataset_name, index_type)
        os.makedirs(index_dir, exist_ok=True)
        return index_dir

    def build_index(self, documents: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not documents:
            raise ValueError("Document list cannot be empty.")
        
        for doc in tqdm(documents, desc="Building Index", leave=False, ncols=100):
            doc_id, text = doc.get('id'), doc.get('processed_text')
            if not doc_id or not text: continue
            tokens = text.split()
            self.doc_lengths[doc_id] = len(tokens)
            term_frequencies = defaultdict(int)
            for token in tokens:
                term_frequencies[token] += 1
            for term, freq in term_frequencies.items():
                self.index[term][doc_id] = freq
        
        return len(self.doc_lengths), len(self.index)

    def save_index(self, dataset_name: str, index_type: str = "inverted_index"):
        if not self.index:
            raise ValueError("Index is not built. Call build_index() first.")
            
        index_dir = self._get_index_paths(dataset_name, index_type)
        
        # حفظ الملفات داخل المجلد المحدد
        joblib.dump(dict(self.index), os.path.join(index_dir, "inverted_index.joblib"))
        joblib.dump(self.doc_lengths, os.path.join(index_dir, "doc_lengths.joblib"))
        print(f"✅ Index for '{dataset_name}' saved successfully in '{index_dir}'")