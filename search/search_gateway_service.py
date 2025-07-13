# services/search_gateway_service.py
import httpx
from fastapi import FastAPI, HTTPException, status, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal


TFIDF_SEARCH_URL = "http://127.0.0.1:8031" 
BERT_SEARCH_URL = "http://127.0.0.1:8034"  
HYBRID_SEARCH_URL = "http://127.0.0.1:8035"
app = FastAPI(
    title="Unified Search Gateway",
    description="A gateway that routes search requests to the appropriate specialized service (TF-IDF or BERT).",
    version="3.0.0"
)

class UnifiedSearchRequest(BaseModel):
    query: str
    dataset_name: str
    model_type: Literal['tfidf', 'bert', 'hybrid'] 
    top_k: int = Field(10, gt=0, le=50)

class SearchResult(BaseModel):
    doc_id: str
    score: float
    document_text: str

@app.post("/search", response_model=List[SearchResult], tags=["Unified Search"])
async def unified_search(request: UnifiedSearchRequest):
   
    payload = {
        "query": request.query,
        "dataset_name": request.dataset_name,
        "top_k": request.top_k
    }

   
    if request.model_type == 'tfidf':
       
        payload['model_type'] = 'tfidf'
        target_url = f"{TFIDF_SEARCH_URL}/search-tfidf"
    elif request.model_type == 'bert':
        target_url = f"{BERT_SEARCH_URL}/search/bert"
    elif request.model_type == 'hybrid': 
        target_url = f"{HYBRID_SEARCH_URL}/search-hybrid" 
    
    else:
       raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid 'model_type'. Must be 'tfidf', 'bert', or 'hybrid'."
        )

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(target_url, json=payload)
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Error communicating with a search service: {exc}")
        except httpx.HTTPStatusError as exc:
            detail_content = exc.response.text
            try:
                detail_content = exc.response.json()
            except Exception:
                pass 
            
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=detail_content
            )
        except Exception as e:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

