# # app.py
# from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
# from pydantic import BaseModel
# from typing import List

# import config
# from handlers.database_handler import DatabaseHandler
# from handlers.data_loader_handler import DataLoaderHandler
# # ❗️ CORRECTED IMPORT: Use TextProcessingHandler, not a non-existent service
# from handlers.text_processing_handler import TextProcessingHandler
# from utils.logger_config import logger

# # --- App Initialization ---
# app = FastAPI(
#     title="Advanced Information Retrieval System API",
#     description="An SOA-based API to load and process text datasets.",
#     version="2.0.0"
# )

# # --- Dependency Injection ---
# def get_db_handler():
#     """Dependency for the Database Handler."""
#     db = DatabaseHandler(config.MYSQL_CONFIG)
#     try:
#         db.connect()
#         yield db
#     finally:
#         db.disconnect()

# # --- Request Models ---
# class DataRequest(BaseModel):
#     dataset_name: str

# # --- API Endpoints ---
# @app.post("/load-data", status_code=202)
# def load_data_endpoint(request: DataRequest, background_tasks: BackgroundTasks, db: DatabaseHandler = Depends(get_db_handler)):
#     """
#     Endpoint to trigger loading a dataset in the background.
#     """
#     dataset_name = request.dataset_name
#     if dataset_name not in config.DATASET_CONFIGS:
#         raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found in configuration.")

#     logger.info(f"Received request to load dataset: '{dataset_name}'. Task added to background.")
    
#     # This was changed in the original file content provided in the prompt, so I'm updating it here to match
#     loader_handler = DataLoaderHandler(config.DATASET_CONFIGS, config.DATASETS_BASE_DIR, config.BATCH_SIZE)
#     background_tasks.add_task(loader_handler.load_dataset, dataset_name)
    
#     return {"message": f"Data loading for '{dataset_name}' has been started in the background."}

# # ✅ --- THIS IS THE CORRECTED FUNCTION --- ✅
# @app.post("/process-data", status_code=202)
# def process_data_endpoint(request: DataRequest, background_tasks: BackgroundTasks):
#     """
#     Endpoint to trigger the sequential text processing pipeline for a dataset in the background.
#     """
#     dataset_name = request.dataset_name
#     logger.info(f"Received request to process dataset: '{dataset_name}'. Task added to background.")

#     # Use the correct handler class
#     processing_handler = TextProcessingHandler(config.SYMPSPELL_DICT_PATH)
    
#     # Call the updated function with the correct arguments
#     # The handler now manages its own DB connection, so we don't pass `db`.
#     # We also removed the CPU_CORES argument.
#     background_tasks.add_task(
#         processing_handler.run_processing_pipeline,
#         dataset_name,
#         config.BATCH_SIZE
#     )

#     return {"message": f"Sequential text processing for '{dataset_name}' has been started in the background."}


# @app.get("/", tags=["Health Check"])
# def read_root():
#     return {"status": "API is running and ready!"}