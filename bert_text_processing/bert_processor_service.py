# services/bert_processor_service.py
from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Annotated
from tqdm import tqdm

import utils.config as config
from .bert_processing_handler import BertPreprocessingHandler
from database.database_handler import DatabaseHandler
from utils.logger_config import logger


bert_processor_instance = None

def run_bert_processing_task(dataset_name: str, batch_size: int, processor: BertPreprocessingHandler):
    db_handler = None
    try:
        db_handler = DatabaseHandler(config.MYSQL_CONFIG)
        db_handler.connect()
        db_handler.setup_tables()

        logger.info(f"🚀 Starting processing for dataset: '{dataset_name}'")
        total_processed_count = 0
        total_remaining_count = db_handler.count_unprocessed_docs(dataset_name)
        logger.info(f"📊 Total unprocessed documents: {total_remaining_count}")

        while total_remaining_count > 0:
            docs_to_process = db_handler.get_unprocessed_docs(dataset_name, batch_size)

            if not docs_to_process:
                logger.info("✅ No more unprocessed documents. Processing completed.")
                break

            logger.info(f"⚙️ Processing batch of {len(docs_to_process)} documents (remaining: {total_remaining_count})...")

            texts = [doc['raw_text'] for doc in docs_to_process]
            processed_texts = [processor.process_single_text(text) for text in tqdm(texts, desc="🔬 Processing")]

            updates = [(processed, docs_to_process[i]['id']) for i, processed in enumerate(processed_texts)]
            updated_count = db_handler.bulk_update_processed_text(updates)
            total_processed_count += updated_count

            logger.info(f"✅ Batch complete. {updated_count} documents updated. Total processed so far: {total_processed_count}")
            total_remaining_count = db_handler.count_unprocessed_docs(dataset_name)

        logger.info(f"🎉 All documents processed for dataset '{dataset_name}'. Final total: {total_processed_count}")

    except Exception as e:
        logger.error(f"❌ Critical error during processing for '{dataset_name}': {e}", exc_info=True)
    finally:
        if db_handler:
            db_handler.disconnect()
@asynccontextmanager
async def lifespan(app: FastAPI):
   
    global bert_processor_instance
    logger.info("🚀 Starting BERT Processor Microservice...")
    bert_processor_instance = BertPreprocessingHandler()
    logger.info("✅ BertPreprocessingHandler initialized successfully.")
    yield
    logger.info("🛑 Shutting down BERT Processor Microservice...")

app = FastAPI(
    title="BERT Processor Microservice",
    description="Microservice for document text preprocessing optimized for BERT models.",
    version="1.0.0",
    lifespan=lifespan
)

class DataRequest(BaseModel):
    dataset_name: str

def get_bert_processor() -> BertPreprocessingHandler:
    return bert_processor_instance

BertProcessorDependency = Annotated[BertPreprocessingHandler, Depends(get_bert_processor)]

@app.post("/process-bert-data", status_code=202)
def process_bert_data_endpoint(
    request: DataRequest,
    background_tasks: BackgroundTasks,
    processor: BertProcessorDependency
):
    
    dataset_name = request.dataset_name
    logger.info(f"📥 Received BERT processing request for dataset: '{dataset_name}'")

    background_tasks.add_task(
        run_bert_processing_task,
        dataset_name,
        config.BATCH_SIZE,
        processor
    )

    return {"message": f"✅ BERT processing for dataset '{dataset_name}' has been started in the background."}

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "🟢 BERT Processor is running!"}
