# services/semantic_expansion_service.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict
from contextlib import asynccontextmanager

from .semantic_expansion_handler import SemanticExpansionHandler
from utils.logger_config import logger
from utils import config
# --- Singleton Pattern for Handlers ---
# A dictionary to hold one handler instance per dataset
shared_handlers: Dict[str, SemanticExpansionHandler] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A lifespan manager that pre-warms the cache for specified datasets if needed.
    """
    logger.info("Semantic Expansion Service is starting up...")
    # You can pre-warm a popular dataset here if you want
    # logger.info("Pre-warming handler for 'antique_qa' dataset...")
    # shared_handlers['antique_qa'] = SemanticExpansionHandler('antique_qa')
    yield
    logger.info("Semantic Expansion Service is shutting down.")
    shared_handlers.clear()

app = FastAPI(
    title="Semantic Query Expansion Service",
    description="Expands queries using BERT to find similar terms from the corpus.",
    lifespan=lifespan
)

# --- Dependency Injection ---
def get_handler(dataset_name: str) -> SemanticExpansionHandler:
    """
    This function acts as a dependency. It creates a handler for a dataset
    the first time it's requested and then reuses it for subsequent requests.
    """
    def get_handler(dataset_name: str) -> SemanticExpansionHandler:
        if dataset_name not in shared_handlers:
            logger.info(f"Creating a new SemanticExpansionHandler for dataset '{dataset_name}'.")
            try:
                shared_handlers[dataset_name] = SemanticExpansionHandler(dataset_name, db_config=DB_CONFIG)
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
    """
    Expands the user query with semantically related terms.
    """
    try:
        result = handler.expand(request.query, request.top_k)
        return ExpansionResponse(**result)
    except Exception as e:
        logger.error(f"An error occurred during expansion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during query expansion.")

# --- How to run this service ---
# uvicorn services.semantic_expansion_service:app --reload --port 8010
