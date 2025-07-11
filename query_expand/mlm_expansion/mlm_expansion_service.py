# services/mlm_expansion_service.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from typing import List, Annotated
from contextlib import asynccontextmanager

from .mlm_expansion_handler import MlmExpansionHandler
from utils.logger_config import logger

# --- Singleton Pattern for the Handler ---
shared_handler: MlmExpansionHandler | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the heavy MLM model once when the service starts.
    """
    global shared_handler
    logger.info("MLM Expansion Service is starting up...")
    shared_handler = MlmExpansionHandler()
    logger.info("Shared MLMExpansionHandler initialized successfully.")
    yield
    logger.info("MLM Expansion Service is shutting down.")
    shared_handler = None

app = FastAPI(
    title="MLM Query Expansion Service",
    description="Expands queries using a Masked Language Model like BERT.",
    lifespan=lifespan
)

# --- Dependency Injection ---
def get_handler() -> MlmExpansionHandler:
    return shared_handler

HandlerDependency = Annotated[MlmExpansionHandler, Depends(get_handler)]

class ExpansionRequest(BaseModel):
    query: str
    top_k: int = Field(5, gt=0, le=15)

class ExpansionResponse(BaseModel):
    expanded_query: str
    expansion_terms: List[str]

@app.post("/expand-mlm", response_model=ExpansionResponse, tags=["MLM Expansion"])
async def expand_query_endpoint(request: ExpansionRequest, handler: HandlerDependency):
    """
    Expands the user query with contextually relevant terms predicted by an MLM.
    """
    result = handler.expand(request.query, request.top_k)
    return ExpansionResponse(**result)

# --- How to run this service ---
# uvicorn services.mlm_expansion_service:app --reload --port 8011
