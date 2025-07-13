
import time
from fastapi import FastAPI, Depends, HTTPException, Request


from utils.config import MYSQL_CONFIG, OUTPUT_DIR, DATASET_CONFIGS
from database.database_handler import DatabaseHandler
from .hybrid_search_handler import HybridSearchHandler
from utils.cache_manager import CacheManager
from pydantic import BaseModel, Field
from typing import List
from utils.config import QUERY_PREPROCESSOR_URL 


cache_manager = CacheManager()


app = FastAPI(
    title="Hybrid Search Service (Central Cache)",
    description="Uses a central, pre-loaded cache for maximum performance."
)


@app.on_event("startup")
async def startup_event():
   
    await cache_manager.load_all_models()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

def get_db_handler():
    db_handler = DatabaseHandler(MYSQL_CONFIG)
    try:
        db_handler.connect()
        yield db_handler
    finally:
        db_handler.disconnect()


class HybridSearchRequest(BaseModel):
    query: str
    dataset_name: str
    top_k: int = Field(10, gt=0, le=50)

class SearchResult(BaseModel):
    doc_id: str
    score: float
    document_text: str

@app.post("/search-hybrid")
async def search_hybrid(request: HybridSearchRequest, db: DatabaseHandler = Depends(get_db_handler)):
    dataset = request.dataset_name

    tfidf_assets = cache_manager.get_assets(dataset, "tfidf")
    bert_assets = cache_manager.get_assets(dataset, "bert")

    if not tfidf_assets or not bert_assets:
        raise HTTPException(status_code=404, detail=f"Models for '{dataset}' not available in cache.")

    handler = HybridSearchHandler(
        db_handler=db,
        tfidf_assets=tfidf_assets,
        bert_assets=bert_assets,
        preprocess_url=QUERY_PREPROCESSOR_URL
    )
    results = handler.search(query=request.query, top_k=request.top_k)
    return results