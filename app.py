# app.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List

import utils.config as config
from database.database_handler import DatabaseHandler
from data_loader.data_loader_handler import DataLoaderHandler
from handlers.text_processing_handler import TextProcessingHandler
from utils.logger_config import logger

# Import dataset management router for user uploads
from dataset_management.dataset_management_handler import router as dataset_router

# --- App Initialization ---
app = FastAPI(
    title="Advanced Information Retrieval System API",
    description="An SOA-based API to load and process text datasets. Users can now upload their own datasets.",
    version="3.0.0"
)

# Include dataset management router
app.include_router(dataset_router)

# --- Dependency Injection ---
def get_db_handler():
    """Dependency for the Database Handler."""
    db = DatabaseHandler(config.MYSQL_CONFIG)
    try:
        db.connect()
        yield db
    finally:
        db.disconnect()

# --- Request Models ---
class DataRequest(BaseModel):
    dataset_name: str

# --- API Endpoints ---
@app.post("/load-data", status_code=202)
def load_data_endpoint(request: DataRequest, background_tasks: BackgroundTasks, db: DatabaseHandler = Depends(get_db_handler)):
    """
    Endpoint to trigger loading a dataset in the background.
    """
    dataset_name = request.dataset_name
    if dataset_name not in config.DATASET_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found in configuration.")

    logger.info(f"Received request to load dataset: '{dataset_name}'. Task added to background.")
    
    loader_handler = DataLoaderHandler(config.DATASET_CONFIGS, config.DATASETS_BASE_DIR, config.BATCH_SIZE)
    background_tasks.add_task(loader_handler.load_dataset, dataset_name)
    
    return {"message": f"Data loading for '{dataset_name}' has been started in the background."}

@app.post("/process-data", status_code=202)
def process_data_endpoint(request: DataRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the sequential text processing pipeline for a dataset in the background.
    """
    dataset_name = request.dataset_name
    logger.info(f"Received request to process dataset: '{dataset_name}'. Task added to background.")

    processing_handler = TextProcessingHandler(config.SYMPSPELL_DICT_PATH)
    
    background_tasks.add_task(
        processing_handler.run_processing_pipeline,
        dataset_name,
        config.BATCH_SIZE
    )

    return {"message": f"Sequential text processing for '{dataset_name}' has been started in the background."}


@app.get("/", tags=["Health Check"])
def read_root():
    return {
        "status": "API is running and ready!",
        "features": [
            "Upload custom datasets via /api/v1/datasets/upload",
            "List datasets via /api/v1/datasets",
            "Check dataset status via /api/v1/datasets/{name}/status",
            "Delete datasets via /api/v1/datasets/{name}",
            "Trigger processing via /api/v1/datasets/{name}/process"
        ]
    }