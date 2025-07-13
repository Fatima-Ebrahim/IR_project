
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict

app = FastAPI(
    title="BERT Search Service",
    description="Orchestrates the BERT search pipeline by calling the Query Processor and Ranking services.",
    version="2.0.0"
)

QUERY_PROCESSOR_URL = "http://127.0.0.1:8032" 
BERT_RANKING_URL = "http://127.0.0.1:8033"    

class SearchRequest(BaseModel):
    query: str
    dataset_name: str
    top_k: int = Field(10, gt=0, le=50)

class SearchResult(BaseModel):
    doc_id: str
    score: float
    document_text: str

@app.post("/search/bert", response_model=List[SearchResult], tags=["BERT Search"])
async def search_bert(request: SearchRequest):
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            
            processor_payload = {"query": request.query}
            processor_url = f"{QUERY_PROCESSOR_URL}/process-query/bert"
            
            processor_response = await client.post(processor_url, json=processor_payload)
            processor_response.raise_for_status()
            query_vector = processor_response.json()['query_vector']

           
            ranking_payload = {
                "query_vector": [query_vector],
                "top_k": request.top_k
            }
            ranking_url = f"{BERT_RANKING_URL}/rank-bert/{request.dataset_name}"

            ranking_response = await client.post(ranking_url, json=ranking_payload)
            ranking_response.raise_for_status()
            
            return ranking_response.json()

        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Error communicating with a downstream service: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.json())
