# services/bert_query_processor_service.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List

# استيراد الـ handler الجديد
from search.bert_search.bert_query_processing.bert_query_processing_handler import BertQueryProcessorHandler
from utils.logger_config import logger

app = FastAPI(
    title="BERT Query Processor Service",
    description="A dedicated service to process raw text queries and convert them into BERT vector embeddings.",
    version="1.0.0"
)

# إنشاء نسخة واحدة من الـ handler عند بدء تشغيل الخدمة لتحسين الأداء
# هذا يضمن أن نموذج BERT يتم تحميله في الذاكرة مرة واحدة فقط
try:
    query_handler = BertQueryProcessorHandler()
except Exception as e:
    logger.error(f"Fatal error during service startup: Could not initialize BertQueryProcessorHandler. {e}")
    query_handler = None

class QueryRequest(BaseModel):
    query: str = Field(..., example="what is the 6-71 diesel engine")

class VectorResponse(BaseModel):
    query_vector: List[float]

@app.post(
    "/process-query/bert",
    response_model=VectorResponse,
    tags=["BERT Query Processing"]
)
async def process_bert_query(request: QueryRequest):
    """
    Takes a raw query and returns its corresponding BERT vector embedding.
    """
    if query_handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BERT model is not available due to a startup error."
        )
    
    try:
        query_vector = query_handler.process_query_to_vector(request.query)
        return VectorResponse(query_vector=query_vector)
    except Exception as e:
        logger.error(f"An error occurred while processing query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "BERT Query Processor Service is running!"}

