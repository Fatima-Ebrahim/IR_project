# services/tfidf_search_service.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Literal

import utils.config as config
from database.database_handler import DatabaseHandler
# (مهم) استيراد الـ handler الجديد الخاص بالترتيب
from search.tfidf_search.tfidf_ranking_handler import TfidfRankingHandler 

# Dependency to get a database session
def get_db_handler():
    db_handler = DatabaseHandler(config.MYSQL_CONFIG)
    try:
        db_handler.connect()
        yield db_handler
    finally:
        db_handler.disconnect()

app = FastAPI(
    title="TF-IDF Search Service",
    description="A specialized service for efficient searching using TF-IDF and an inverted index.",
    version="1.0.0"
)

class TfidfSearchRequest(BaseModel):
    query: str
    dataset_name: str
    model_type: Literal['tfidf'] # To ensure this service only handles TF-IDF
    top_k: int = Field(10, gt=0, le=50)

class SearchResult(BaseModel):
    doc_id: str
    score: float
    document_text: str

@app.post("/search-tfidf", response_model=List[SearchResult], tags=["TF-IDF Search"])
async def search_tfidf(request: TfidfSearchRequest, db: DatabaseHandler = Depends(get_db_handler)):
    """
    Handles a search request using the efficient inverted index ranking method.
    """
    try:
        # The handler now takes the db_handler as a dependency
        ranking_handler = TfidfRankingHandler(request.dataset_name, db_handler=db)
        
        # The rank method performs the entire process: query processing, retrieval, and scoring
        ranked_docs = ranking_handler.rank(request.query, request.top_k)
        
        return ranked_docs
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Required model or index file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
