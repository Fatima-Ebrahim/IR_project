# services/data_loader_service.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

import utils.config as config
from database.data_loader_handler import DataLoaderHandler
from utils.logger_config import logger

app = FastAPI(
    title="Data Loader Microservice",
    description="A dedicated microservice to load datasets into the database.",
    version="1.1.0"
)

class DataRequest(BaseModel):
    dataset_name: str

@app.post("/load-data", status_code=202)
def load_data_endpoint(request: DataRequest, background_tasks: BackgroundTasks):
    """
    Triggers loading a dataset in the background.
    The background task will manage its own database connection.
    """
    dataset_name = request.dataset_name
    if dataset_name not in config.DATASET_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found in configuration.")

    logger.info(f"Received request to load dataset: '{dataset_name}'. Task added to background.")
    
    # لم نعد نمرر اتصال قاعدة البيانات هنا
    loader_handler = DataLoaderHandler(config.DATASET_CONFIGS, config.DATASETS_BASE_DIR, config.BATCH_SIZE)
    background_tasks.add_task(loader_handler.load_dataset, dataset_name)
    
    return {"message": f"Data loading for '{dataset_name}' has been started in the background."}

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "Data Loader Service is running!"}
