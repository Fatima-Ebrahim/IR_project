
import uuid
from datetime import datetime
from typing import Dict, Any, Literal, Optional

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from pydantic import BaseModel


from indexing.inverted_index_handler import InvertedIndexHandler
from database.database_handler import DatabaseHandler
import  utils.config as config  

app = FastAPI(
    title="Final Integrated Indexing Service",
    description="Builds indexes asynchronously using production settings.",
    version="5.0.0"
)


job_store: Dict[str, Dict[str, Any]] = {}

class IndexRequest(BaseModel):
    dataset_name: str

class IndexingDetails(BaseModel):
    documents_indexed: int
    vocabulary_size: int
    index_type: str

class JobCreationResponse(BaseModel):
    message: str
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    details: Optional[IndexingDetails] = None
    error_message: Optional[str] = None

def create_index_task(job_id: str, dataset_name: str, index_type: str = "inverted_index"):

    db_handler = None
    try:
        job_store[job_id]["status"] = "running"
        
      
        db_handler = DatabaseHandler(config.MYSQL_CONFIG)
        db_handler.connect()

        documents = db_handler.get_processed_docs_for_indexing(dataset_name)
        if not documents:
            raise ValueError(f"No processed documents found for '{dataset_name}'. Please process the text first.")

        index_handler = InvertedIndexHandler()
        docs_indexed, vocab_size = index_handler.build_index(documents)
        index_handler.save_index(dataset_name, index_type)
        
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["details"] = {
            "documents_indexed": docs_indexed,
            "vocabulary_size": vocab_size,
            "index_type": index_type
        }
    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error_message"] = str(e)
    finally:
        if db_handler:
            db_handler.disconnect()

@app.post("/create-index", response_model=JobCreationResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_index(request: IndexRequest, background_tasks: BackgroundTasks):
    for job_id, job_info in job_store.items():
        if job_info["dataset_name"] == request.dataset_name and job_info["status"] == "running":
            raise HTTPException(status_code=409, detail="Indexing job already in progress.")

    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending", "dataset_name": request.dataset_name, "created_at": datetime.now()}
    
    background_tasks.add_task(create_index_task, job_id, request.dataset_name)
    
    return JobCreationResponse(message="Indexing job has been accepted.", job_id=job_id)

@app.get("/index-job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    return JobStatusResponse(job_id=job_id, **job)