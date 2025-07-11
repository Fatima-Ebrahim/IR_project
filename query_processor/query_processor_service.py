# # services/query_processor_service.py
# todo it was working usign old text editor
# from fastapi import FastAPI, HTTPException, status
# from pydantic import BaseModel, Field
# from typing import Literal, List

# from query_processor.query_processor_handler import QueryProcessorHandler

# app = FastAPI(title="Query Processor Service")

# class QueryRequest(BaseModel):
#     query: str = Field(..., example="what is cancer")

# class VectorResponse(BaseModel):
#     query_vector: List[List[float]] # متجه ثنائي الأبعاد دائماً

# @app.post(
#     "/process-query/{dataset_name}/{model_type}",
#     response_model=VectorResponse,
#     tags=["Processing"]
# )
# async def process_query(
#     dataset_name: str,
#     model_type: Literal['tfidf', 'bert'],
#     request: QueryRequest
# ):
#     """يأخذ نص الاستعلام الخام ويعيد المتجه الرقمي الممثل له."""
#     try:
#         handler = QueryProcessorHandler(dataset_name, model_type)
#         query_vector = handler.process(request.query)
#         return VectorResponse(query_vector=query_vector)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# services/query_processor_service.py
# services/query_processor_service.py
# services/query_processor_service.py
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Annotated
from contextlib import asynccontextmanager

from .query_processor_handler import QueryProcessorHandler
from text_processing.text_processing_handler import TextProcessingHandler
from utils.logger_config import logger

# متغير عام سيحتوي على النسخة الوحيدة من المعالج (لحل مشكلة السرعة)
shared_text_processor: TextProcessingHandler | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    يقوم بإنشاء نسخة واحدة مشتركة من TextProcessingHandler عند بدء تشغيل الخدمة.
    """
    global shared_text_processor
    logger.info("Query Processor Service is starting up...")
    shared_text_processor = TextProcessingHandler()
    logger.info("Shared TextProcessingHandler initialized successfully.")
    yield
    logger.info("Query Processor Service is shutting down.")
    shared_text_processor = None

app = FastAPI(
    title="Optimized Query Processor Service",
    description="Processes queries efficiently by loading models only once.",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    query: str = Field(..., example="what is cancre")

class ProcessedQueryResponse(BaseModel):
    corrected_query_display: str
    query_vector: List[List[float]]

def get_text_processor() -> TextProcessingHandler:
    """دالة حقن التبعية التي تعيد النسخة المشتركة."""
    return shared_text_processor

ProcessorDependency = Annotated[TextProcessingHandler, Depends(get_text_processor)]

@app.post(
    "/process-query/{dataset_name}/{model_type}",
    response_model=ProcessedQueryResponse,
    tags=["Query Processing"]
)
async def process_query(
    dataset_name: str,
    model_type: Literal['tfidf', 'bert'],
    request: QueryRequest,
    processor: ProcessorDependency
):
    """
    يأخذ الاستعلام الخام ويعالجه ويعيد النص المصحح والمتجه الخاص به.
    """
    try:
        # --- **هذا هو السطر الذي تم تصحيحه** ---
        # الآن نقوم بتمرير النسخة المشتركة (processor) من معالج النصوص
        # إلى QueryProcessorHandler عند إنشائه.
        handler = QueryProcessorHandler(
            dataset_name=dataset_name,
            model_type=model_type,
            text_processor=processor  # <-- **هنا تم حل الخطأ**
        )
        
        handler_result = handler.process(request.query)
        
        return ProcessedQueryResponse(
            corrected_query_display=handler_result["corrected_query_display"],
            query_vector=handler_result["query_vector"]
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"An error occurred during query processing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
