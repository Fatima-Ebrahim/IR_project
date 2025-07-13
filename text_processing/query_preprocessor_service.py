# services/query_preprocessor_service.py

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from text_processing.text_processing_handler import TextProcessingHandler
from utils.logger_config import logger

text_processor_instance = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    processed_query: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_processor_instance
    logger.info("🚀 Starting Query Preprocessor Service...")
    text_processor_instance = TextProcessingHandler()
    logger.info("✅ TextProcessingHandler initialized.")
    yield
    logger.info("🛑 Query Preprocessor Service is shutting down.")

app = FastAPI(
    title="Query Preprocessor Microservice",
    description="Service to preprocess user queries using full NLP pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/preprocess-query", response_model=QueryResponse)
def preprocess_query(request: QueryRequest):
    
    logger.info(f"📥 Received query: '{request.query}'")
    processed = text_processor_instance._process_single_text(request.query)
    logger.info(f"✅ Processed query: '{processed}'")
    return {"processed_query": processed}

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "🟢 Query Preprocessor is running!"}
