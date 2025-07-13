from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .topic_modeling_handler import TopicModelingHandler
from utils.logger_config import logger

app = FastAPI(title="Topic Modeling Service")

class TopicModelRequest(BaseModel):
    dataset_name: str
    num_topics: int = 8

@app.post("/run-topic-modeling/")
async def run_topic_modeling(request: TopicModelRequest):
    handler = TopicModelingHandler()
    try:
        result = handler.run_lda_from_database(request.dataset_name, request.num_topics)
        return result
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error running topic modeling: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
