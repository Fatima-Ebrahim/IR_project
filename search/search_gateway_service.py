# services/search_gateway_service.py
import httpx
from fastapi import FastAPI, HTTPException, status, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal

# Define the base URLs for your specialized search services
# These should be configurable in a real-world application
TFIDF_SEARCH_URL = "http://127.0.0.1:8031" # The self-contained TF-IDF service
BERT_SEARCH_URL = "http://127.0.0.1:8034"  # The BERT orchestrator service
HYBRID_SEARCH_URL = "http://127.0.0.1:8035"
app = FastAPI(
    title="Unified Search Gateway",
    description="A gateway that routes search requests to the appropriate specialized service (TF-IDF or BERT).",
    version="3.0.0"
)

# This model will capture the full request from the user
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
    """
    Receives a search request and forwards it to the correct
    downstream service based on the 'model_type'.
    """
    # Create the payload to be forwarded.
    # The downstream services expect a slightly different structure.
    payload = {
        "query": request.query,
        "dataset_name": request.dataset_name,
        "top_k": request.top_k
    }

    # Determine the target URL and endpoint based on the model type
    if request.model_type == 'tfidf':
        # The TF-IDF service has a model_type in its request body
        payload['model_type'] = 'tfidf'
        target_url = f"{TFIDF_SEARCH_URL}/search-tfidf"
    elif request.model_type == 'bert':
        target_url = f"{BERT_SEARCH_URL}/search/bert"
    elif request.model_type == 'hybrid': 
        target_url = f"{HYBRID_SEARCH_URL}/search-hybrid" 
    
    else:
        # This case is technically handled by Pydantic validation, but it's good practice
       raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid 'model_type'. Must be 'tfidf', 'bert', or 'hybrid'."
        )

    # Forward the request to the specialized service
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(target_url, json=payload)
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Error communicating with a search service: {exc}")
        except httpx.HTTPStatusError as exc:
    # This new block is safer
            detail_content = exc.response.text
            try:
                # We try to parse it as JSON
                detail_content = exc.response.json()
            except Exception:
                # If it fails, we just use the raw text (which might be empty)
                pass 
            
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=detail_content
            )
        except Exception as e:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

