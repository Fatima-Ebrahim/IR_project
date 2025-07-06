# services/ranking_service.py
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Literal, List, Dict

from handlers.ranking_handler import RankingHandler
from database.database_handler import DatabaseHandler
import utils.config as config

def get_db_handler():
    db_handler = DatabaseHandler(config.MYSQL_CONFIG)
    try:
        db_handler.connect()
        db_handler.setup_tables()
        yield db_handler
    finally:
        db_handler.disconnect()

app = FastAPI(title="Ranking & Retrieval Service")

# **تعديل نموذج الطلب**: الآن نستقبل نص الاستعلام مباشرة
class RankRequest(BaseModel):
    query: str
    top_k: int = Field(10, gt=0, le=50)

class SearchResult(BaseModel):
    doc_id: str
    score: float
    document_text: str 

class RankResponse(BaseModel):
    ranked_documents: List[SearchResult]

@app.post(
    "/rank-documents/{dataset_name}/{model_type}",
    response_model=RankResponse,
    tags=["Ranking"]
)
async def rank_documents(
    dataset_name: str,
    model_type: Literal['tfidf', 'bert'],
    request: RankRequest,
    db: DatabaseHandler = Depends(get_db_handler)
):
    try:
        handler = RankingHandler(dataset_name, model_type, db_handler=db)
        # **تعديل الاستدعاء**: نمرر نص الاستعلام الخام
        ranked_docs = handler.rank(request.query, request.top_k)
        return RankResponse(ranked_documents=ranked_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

