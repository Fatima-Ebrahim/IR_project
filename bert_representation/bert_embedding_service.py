# services/bert_embedding_service.py
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

import utils.config as config

from database.database_handler import DatabaseHandler
from bert_representation.bert_embedding_handler import BertEmbeddingHandler
from utils.logger_config import logger

# --- In-memory Job Store ---
# In a production system, you might use Redis or a database for this.
job_store: Dict[str, Dict[str, Any]] = {}

# --- Initialize Handlers ---
# Load the heavy model only once when the service starts.
try:
    embedding_handler = BertEmbeddingHandler()
except Exception as e:
    logger.error(f"Fatal error during startup: Could not initialize BertEmbeddingHandler. {e}")
    embedding_handler = None

# --- FastAPI App and Models ---
app = FastAPI(
    title="Asynchronous BERT Embedding Service",
    description="A service to generate BERT embeddings as a background job.",
    version="2.0.0"
)

class EmbeddingRequest(BaseModel):
    dataset_name: str

class JobCreationResponse(BaseModel):
    message: str
    job_id: str
    status_endpoint: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

# --- Background Task ---
def generate_embeddings_task(job_id: str, dataset_name: str):
    """The actual work that runs in the background."""
    job_store[job_id]["status"] = "running"
    db_handler = None
    try:
        db_handler = DatabaseHandler(config.MYSQL_CONFIG)
        db_handler.connect()
        documents = db_handler.get_processed_docs_for_indexing(dataset_name)
        if not documents:
            raise ValueError(f"No processed documents found for dataset '{dataset_name}'.")

        model_dir = os.path.join(config.OUTPUT_DIR, dataset_name, "bert")
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        matrix_path = os.path.join(model_dir, "matrix.joblib")
        doc_map_path = os.path.join(model_dir, "doc_ids_map.joblib")
  
        docs_generated, embed_dim = embedding_handler.generate_and_save_embeddings(
            documents, vectorizer_path, matrix_path, doc_map_path
        )

        job_store[job_id]["status"] = "completed"
        job_store[job_id]["details"] = {
            "documents_generated": docs_generated,
            "embedding_dimension": embed_dim,
            "output_directory": model_dir
        }
        logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error_message"] = str(e)
    finally:
        if db_handler:
            db_handler.disconnect()


@app.post("/create-embeddings", response_model=JobCreationResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_embeddings_job(request: EmbeddingRequest, background_tasks: BackgroundTasks):
   
    if embedding_handler is None:
        raise HTTPException(status_code=503, detail="Service is unavailable, model not loaded.")

    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status": "pending",
        "dataset_name": request.dataset_name,
        "created_at": datetime.now()
    }
    
    background_tasks.add_task(generate_embeddings_task, job_id, request.dataset_name)
    
    return JobCreationResponse(
        message="Embedding generation job has been accepted.",
        job_id=job_id,
        status_endpoint=f"/job-status/{job_id}"
    )

@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")
    
    return JobStatusResponse(job_id=job_id, **job)
