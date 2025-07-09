# services/hybrid_search_service.py
# كود شغال لكن الذاكرة الموقتة موجدة ضمنه ليس مركزية 
# import os
# import joblib
# from fastapi import FastAPI, Depends, HTTPException, Request
# from pydantic import BaseModel, Field
# from typing import List, Dict
# import time # <--- إضافة جديدة لحساب الوقت

# # ==================== يبدأ التعديل (الجزء الأول) ====================
# # 1. استيراد واستنساخ مدير الذاكرة المخبأة المركزي
# from utils.cache_manager import CacheManager
# cache_manager = CacheManager()
# # ==================== ينتهي التعديل (الجزء الأول) ====================

# app = FastAPI(title="Hybrid Search Service (Cached & Timed)")
# # Import the necessary components
# from utils.config import MYSQL_CONFIG, OUTPUT_DIR, DATASET_CONFIGS
# from database.database_handler import DatabaseHandler
# from .hybrid_search_handler import HybridSearchHandler
# MODEL_CACHE = {}

# # --- إعدادات FastAPI والاتصال بقاعدة البيانات ---
# app = FastAPI(
#     title="Hybrid Search Service (Cached)",
#     description="خدمة للبحث الهجين مع تحميل مسبق للنماذج في الذاكرة لتحقيق أقصى سرعة."
# )

# # --- الخطوة 2: دالة التحميل المسبق عند بدء التشغيل ---
# @app.on_event("startup")
# async def startup_event():
#     """
#     تعمل هذه الدالة مرة واحدة فقط عند بدء تشغيل الخادم.
#     تقوم بتحميل جميع أصول النماذج (TF-IDF و BERT) في الذاكرة.
#     """
#     print("🚀 Server is starting up. Pre-loading models into memory cache...")
    
#     # تحديد مجموعات البيانات وأنواع النماذج للتحميل
#     # نأخذ أسماء مجموعات البيانات من ملف الإعدادات المركزي
#     datasets_to_load = DATASET_CONFIGS.keys()
#     model_types_to_load = ["tfidf", "bert"]

#     for dataset in datasets_to_load:
#         MODEL_CACHE[dataset] = {}
#         for model_type in model_types_to_load:
#             try:
#                 model_dir = os.path.join(OUTPUT_DIR, dataset, model_type)
#                 if not os.path.exists(model_dir):
#                     print(f"⚠️ Directory not found for {model_type}/{dataset}, skipping.")
#                     continue

#                 print(f"  - Loading {model_type} assets for '{dataset}'...")
                
#                 vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
#                 matrix = joblib.load(os.path.join(model_dir, "matrix.joblib"))
#                 doc_ids_map = joblib.load(os.path.join(model_dir, "doc_ids_map.joblib"))
                
#                 # تخزين الأصول في الذاكرة المخبأة
#                 MODEL_CACHE[dataset][model_type] = {
#                     "vectorizer": vectorizer,
#                     "matrix": matrix,
#                     "doc_ids_map": doc_ids_map
#                 }
#                 print(f"  ✅ Successfully loaded {model_type} for '{dataset}'.")
#             except FileNotFoundError:
#                 print(f"  ❌ Failed to load assets for {model_type}/{dataset}. Files not found.")
#             except Exception as e:
#                 print(f"  ❌ An error occurred loading {model_type}/{dataset}: {e}")

# # --- Dependency for Database Connection ---
# def get_db_handler():
#     db_handler = DatabaseHandler(MYSQL_CONFIG)
#     try:
#         db_handler.connect()
#         yield db_handler
#     finally:
#         db_handler.disconnect()

# # --- API Models ---
# class HybridSearchRequest(BaseModel):
#     query: str
#     dataset_name: str
#     top_k: int = Field(10, gt=0, le=50)

# class SearchResult(BaseModel):
#     doc_id: str
#     score: float
#     document_text: str

# # --- FastAPI App ---
# @app.post("/search-hybrid", response_model=List[SearchResult])
# async def search_hybrid(request: HybridSearchRequest, db: DatabaseHandler = Depends(get_db_handler)):
#     """
#     يستخدم النماذج المحملة مسبقاً من الذاكرة لتنفيذ البحث بسرعة فائقة.
#     """
#     dataset = request.dataset_name
    
#     # التحقق من أن أصول النماذج المطلوبة محملة في الذاكرة
#     if dataset not in MODEL_CACHE or "tfidf" not in MODEL_CACHE[dataset] or "bert" not in MODEL_CACHE[dataset]:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Models for dataset '{dataset}' are not pre-loaded in the cache. Please ensure they are generated and the service is restarted."
#         )

#     try:
#         # تمرير الأصول المحملة مسبقاً إلى المعالج
#         handler = HybridSearchHandler(
#             db_handler=db,
#             tfidf_assets=MODEL_CACHE[dataset]["tfidf"],
#             bert_assets=MODEL_CACHE[dataset]["bert"]
#         )
#         results = handler.search(query=request.query, top_k=request.top_k)
#         return results
#     except Exception as e:
#         print(f"An unexpected error occurred in hybrid search: {e}")
#         raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# services/hybrid_search_service.py
# 
# services/hybrid_search_service.py
import time
from fastapi import FastAPI, Depends, HTTPException, Request

# 1. استيراد المكونات الضرورية والمدير المركزي
from utils.config import MYSQL_CONFIG, OUTPUT_DIR, DATASET_CONFIGS
from database.database_handler import DatabaseHandler
from .hybrid_search_handler import HybridSearchHandler
from utils.cache_manager import CacheManager
from pydantic import BaseModel, Field
from typing import List

# 2. إنشاء نسخة واحدة من المدير المركزي
cache_manager = CacheManager()

# 3. تعريف التطبيق
app = FastAPI(
    title="Hybrid Search Service (Central Cache)",
    description="Uses a central, pre-loaded cache for maximum performance."
)

# 4. دالة بدء التشغيل البسيطة
@app.on_event("startup")
async def startup_event():
    # تستدعي المدير المركزي ليقوم بالتحميل
    await cache_manager.load_all_models()

# دالة Middleware لقياس الوقت (اختياري)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

# # --- Dependency for Database Connection ---
def get_db_handler():
    db_handler = DatabaseHandler(MYSQL_CONFIG)
    try:
        db_handler.connect()
        yield db_handler
    finally:
        db_handler.disconnect()

# --- API Models ---
class HybridSearchRequest(BaseModel):
    query: str
    dataset_name: str
    top_k: int = Field(10, gt=0, le=50)

class SearchResult(BaseModel):
    doc_id: str
    score: float
    document_text: str

# 5. نقطة النهاية النظيفة
@app.post("/search-hybrid")
async def search_hybrid(request: HybridSearchRequest, db: DatabaseHandler = Depends(get_db_handler)):
    dataset = request.dataset_name

    # الحصول على الأصول من المدير المركزي
    tfidf_assets = cache_manager.get_assets(dataset, "tfidf")
    bert_assets = cache_manager.get_assets(dataset, "bert")

    if not tfidf_assets or not bert_assets:
        raise HTTPException(status_code=404, detail=f"Models for '{dataset}' not available in cache.")

    # تمرير الأصول الجاهزة إلى المعالج
    handler = HybridSearchHandler(
        db_handler=db,
        tfidf_assets=tfidf_assets,
        bert_assets=bert_assets
    )
    results = handler.search(query=request.query, top_k=request.top_k)
    return results