# services/query_processor_service.py

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal, List

from handlers.query_processor_handler import QueryProcessorHandler

app = FastAPI(title="Query Processor Service")

class QueryRequest(BaseModel):
    query: str = Field(..., example="what is cancer")

class VectorResponse(BaseModel):
    query_vector: List[List[float]] # متجه ثنائي الأبعاد دائماً

@app.post(
    "/process-query/{dataset_name}/{model_type}",
    response_model=VectorResponse,
    tags=["Processing"]
)
async def process_query(
    dataset_name: str,
    model_type: Literal['tfidf', 'bert'],
    request: QueryRequest
):
    """يأخذ نص الاستعلام الخام ويعيد المتجه الرقمي الممثل له."""
    try:
        handler = QueryProcessorHandler(dataset_name, model_type)
        query_vector = handler.process(request.query)
        return VectorResponse(query_vector=query_vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))