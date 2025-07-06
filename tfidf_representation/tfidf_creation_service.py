# services/tfidf_creation_service.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

import utils.config as config
from database.database_handler import DatabaseHandler
from tfidf_representation.tfidf_handler import TfIdfHandler # <-- استيراد الـ handler المعدّل
from utils.logger_config import logger

app = FastAPI(
    title="TF-IDF Creation Microservice",
    description="A service to create TF-IDF by processing RAW text on-the-fly.",
    version="2.0.0"
)

class RepresentationRequest(BaseModel):
    dataset_name: str

def create_tfidf_task(dataset_name: str):
    """The background task for creating TF-IDF representation from raw text."""
    db_handler = None
    try:
        db_handler = DatabaseHandler(config.MYSQL_CONFIG)
        db_handler.connect()
        db_handler.setup_tables()

        # **هنا التغيير**: جلب النصوص الخام بدلاً من المعالجة
        logger.info(f"Fetching RAW documents for '{dataset_name}'...")
        # تأكد من وجود هذه الدالة في database_handler.py
        raw_docs = db_handler.get_raw_docs_for_indexing(dataset_name)

        if not raw_docs:
            logger.warning(f"No raw documents found for '{dataset_name}'. Aborting.")
            return

        tfidf_handler = TfIdfHandler()
        tfidf_handler.build_representation(raw_docs) # تمرير النصوص الخام
        tfidf_handler.save_representation(dataset_name)
        
        logger.info(f"Successfully created TF-IDF representation for '{dataset_name}'.")

    except Exception as e:
        logger.error(f"An error occurred during TF-IDF creation for '{dataset_name}': {e}", exc_info=True)
    finally:
        if db_handler:
            db_handler.disconnect()

@app.post("/create-tfidf", status_code=202)
def create_tfidf_endpoint(request: RepresentationRequest, background_tasks: BackgroundTasks):
    """
    Triggers the TF-IDF creation process in the background using raw text.
    """
    logger.info(f"Received request to create TF-IDF for dataset: '{request.dataset_name}'.")
    background_tasks.add_task(create_tfidf_task, request.dataset_name)
    return {"message": f"TF-IDF creation from raw text for '{request.dataset_name}' has been started."}

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "TF-IDF Creation Service is running!"}

# --- How to run this service ---
# uvicorn services.tfidf_creation_service:app --reload --port 8003
