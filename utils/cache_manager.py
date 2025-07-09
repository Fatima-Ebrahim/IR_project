# utils/cache_manager.py
import os
import joblib
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from utils.config import OUTPUT_DIR, DATASET_CONFIGS

class CacheManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'cache'):
            self.cache = {}
            print("CacheManager initialized.")

    # ==================== يبدأ التعديل هنا ====================

    def _load_single_model(self, task: tuple) -> tuple:
        """
        دالة مساعدة لتحميل نموذج واحد. مصممة ليتم استدعاؤها بشكل متوازٍ.
        ترجع مفتاح النموذج والأصول المحملة.
        """
        dataset, model_type = task
        try:
            model_dir = os.path.join(OUTPUT_DIR, dataset, model_type)
            if not os.path.exists(model_dir):
                print(f"  ⚠️ Directory not found for {model_type}/{dataset}, skipping.")
                return dataset, model_type, None

            print(f"  - Starting load for {model_type}/{dataset}...")
            assets = {
                "vectorizer": joblib.load(os.path.join(model_dir, "vectorizer.joblib")),
                "matrix": joblib.load(os.path.join(model_dir, "matrix.joblib")),
                "doc_ids_map": joblib.load(os.path.join(model_dir, "doc_ids_map.joblib"))
            }
            print(f"  ✅ Finished load for {model_type}/{dataset}.")
            return dataset, model_type, assets
        except Exception as e:
            print(f"  ❌ Error loading assets for {model_type}/{dataset}: {e}")
            return dataset, model_type, None

    async def load_all_models(self):
        """
        النسخة المحسّنة: تقوم بتحميل كل النماذج على التوازي.
        """
        if self.cache:
            print("Models are already loaded. Skipping.")
            return

        print("🚀 Central CacheManager: Starting PARALLEL pre-load of all models...")
        
        # 1. إعداد قائمة بكل مهام التحميل التي يجب تنفيذها
        tasks = []
        for dataset in DATASET_CONFIGS.keys():
            self.cache[dataset] = {} # تهيئة القاموس
            for model_type in ["tfidf", "bert"]:
                tasks.append((dataset, model_type))

        # 2. إنشاء مجمع خيوط (Thread Pool) وتنفيذ المهام على التوازي
        # سيقوم بتشغيل عدد من المهام في نفس الوقت (حتى 10 هنا)
        with ThreadPoolExecutor(max_workers=10) as executor:
            # executor.map يطبق الدالة على كل مهمة ويجمع النتائج
            results = executor.map(self._load_single_model, tasks)

        # 3. تجميع النتائج ووضعها في الذاكرة المخبأة
        for dataset, model_type, loaded_assets in results:
            if loaded_assets:
                self.cache[dataset][model_type] = loaded_assets
        
        print("✅ Central CacheManager: All models loaded in parallel.")

    # ==================== ينتهي التعديل هنا ====================
    
    def get_assets(self, dataset: str, model_type: str):
        """
        Retrieves the assets for a specific model and dataset from the cache.
        """
        return self.cache.get(dataset, {}).get(model_type)