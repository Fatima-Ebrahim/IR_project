# services/bert_query_preprocessor_service.py

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from .bert_processing_handler import BertPreprocessingHandler
from utils.logger_config import logger

bert_processor_instance = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    processed_query: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bert_processor_instance
    logger.info("🚀 Starting BERT Query Preprocessor Service...")
    bert_processor_instance = BertPreprocessingHandler()
    logger.info("✅ BertPreprocessingHandler initialized.")
    yield
    logger.info("🛑 BERT Query Preprocessor Service is shutting down.")

app = FastAPI(
    title="BERT Query Preprocessor Microservice",
    description="Service to preprocess user queries using BERT preprocessing pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/preprocess-bert-query", response_model=QueryResponse)
def preprocess_bert_query(request: QueryRequest):
    logger.info(f"📥 Received BERT query: '{request.query}'")
    processed = bert_processor_instance.preprocess_text(request.query)
    logger.info(f"✅ BERT Processed query: '{processed}'")
    return {"processed_query": processed}

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "🟢 BERT Query Preprocessor is running!"}