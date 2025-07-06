# services/search_gateway_service.py
import httpx
from fastapi import FastAPI, HTTPException, status, Request
from typing import List, Dict, Any

# Define the base URLs for your specialized search services
# These should be configurable in a real-world application
TFIDF_SEARCH_URL = "http://127.0.0.1:8020" # The self-contained TF-IDF service
BERT_SEARCH_URL = "http://127.0.0.1:8021"  # (New) The BERT orchestrator service

app = FastAPI(
    title="Smart Search Gateway",
    description="A gateway that routes search requests to the appropriate specialized service (TF-IDF or BERT).",
    version="2.0.0"
)

@app.post("/search", tags=["Unified Search"])
async def unified_search(request: Request):
    """
    Receives a search request and forwards it to the correct
    downstream service based on the 'model_type' in the request body.
    """
    try:
        payload = await request.json()
        model_type = payload.get("model_type")

        # Route the request to the correct service
        if model_type == 'tfidf':
            target_url = f"{TFIDF_SEARCH_URL}/search-tfidf"
        elif model_type == 'bert':
            target_url = f"{BERT_SEARCH_URL}/search-bert" # <-- Route to BERT service
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 'model_type'. Must be 'tfidf' or 'bert'."
            )

        # Forward the request payload to the specialized service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(target_url, json=payload)
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Error communicating with a search service: {exc}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.json())
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
