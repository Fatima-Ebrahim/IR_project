# services/semantic_expansion_service.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict
from contextlib import asynccontextmanager

from .semantic_expansion_handler import SemanticExpansionHandler
from utils.logger_config import logger
from utils.config import MYSQL_CONFIG

shared_handlers: Dict[str, SemanticExpansionHandler] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info("Semantic Expansion Service is starting up...")
    yield
    logger.info("Semantic Expansion Service is shutting down.")
    shared_handlers.clear()

app = FastAPI(
    title="Semantic Query Expansion Service",
    description="Expands queries using BERT to find similar terms from the corpus.",
    lifespan=lifespan
)


def get_handler(dataset_name: str) -> SemanticExpansionHandler:
    if dataset_name not in shared_handlers:
        logger.info(f"Creating a new SemanticExpansionHandler for dataset '{dataset_name}'.")
        try:
            shared_handlers[dataset_name] = SemanticExpansionHandler(dataset_name, db_config=MYSQL_CONFIG)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return shared_handlers[dataset_name]

class ExpansionRequest(BaseModel):
    query: str
    top_k: int = Field(5, gt=0, le=20)

class ExpansionResponse(BaseModel):
    expanded_query: str
    expansion_terms: List[str]

@app.post("/expand/{dataset_name}", response_model=ExpansionResponse, tags=["Semantic Expansion"])
async def expand_query_endpoint(
    dataset_name: str,
    request: ExpansionRequest,
    handler: SemanticExpansionHandler = Depends(get_handler)
):
    try:
        result = handler.expand(request.query, request.top_k)
        return ExpansionResponse(**result)
    except Exception as e:
        logger.error(f"An error occurred during expansion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during query expansion.")

