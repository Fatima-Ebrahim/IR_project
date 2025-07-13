from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict

import utils.config as config
from database.database_handler import DatabaseHandler

from search.bert_search.bert_ranking.bert_ranking_handler import BertRankingHandler

def get_db_handler():
    db_handler = DatabaseHandler(config.MYSQL_CONFIG)
    try:
        db_handler.connect()
        yield db_handler
    finally:
        db_handler.disconnect()

app = FastAPI(
    title="BERT Ranking Service",
    description="A specialized service for ranking documents against a BERT query vector.",
    version="1.0.0"
)

class BertRankRequest(BaseModel):
    query_vector: List[List[float]] 
    top_k: int = Field(10, gt=0, le=50)

class SearchResult(BaseModel):
    doc_id: str
    score: float
    document_text: str

@app.post("/rank-bert/{dataset_name}", response_model=List[SearchResult], tags=["BERT Ranking"])
async def rank_bert(dataset_name: str, request: BertRankRequest, db: DatabaseHandler = Depends(get_db_handler)):
   
    try:
        handler = BertRankingHandler(dataset_name, db_handler=db)
        ranked_docs = handler.rank(request.query_vector[0], request.top_k)
        return ranked_docs
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Required model file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
