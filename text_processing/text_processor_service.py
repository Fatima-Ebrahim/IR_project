from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Annotated
from tqdm import tqdm

import utils.config as config
from text_processing.text_processing_handler import TextProcessingHandler
from database.database_handler import DatabaseHandler
from utils.logger_config import logger

text_processor_instance = None

def run_database_processing_task(dataset_name: str, batch_size: int, processor: TextProcessingHandler):
    
    db_handler = None
    try:
        db_handler = DatabaseHandler(config.MYSQL_CONFIG)
        db_handler.connect()
        db_handler.setup_tables()

        logger.info(f"🔁 Starting background processing for dataset: '{dataset_name}'")
        total_processed_count = 0

        while True:
            
            docs_to_process = db_handler.get_unprocessed_docs(dataset_name, batch_size)
            remaining = db_handler.count_unprocessed_docs(dataset_name)

            if not docs_to_process:
                logger.info("✅ No more unprocessed documents. Processing complete.")
                break

            logger.info(f"⚙️ Processing batch of {len(docs_to_process)} documents ({remaining} remaining)...")
            texts = [doc['raw_text'] for doc in docs_to_process]

           
            processed_texts = [processor._process_single_text(text) for text in tqdm(texts, desc="🔬 Processing")]

            updates = [(processed, docs_to_process[i]['id']) for i, processed in enumerate(processed_texts)]
            updated_count = db_handler.bulk_update_processed_text(updates)
            total_processed_count += updated_count

            logger.info(f"✅ Updated {updated_count} documents. Total processed so far: {total_processed_count}")

        logger.info("🎉 Background processing task completed successfully.")

    except Exception as e:
        logger.error(f"❌ An error occurred during processing of dataset '{dataset_name}': {e}", exc_info=True)
    finally:
        if db_handler:
            db_handler.disconnect()
            logger.info("🔌 Database connection closed.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global text_processor_instance
    logger.info("🚀 Starting Text Processor Microservice...")
    text_processor_instance = TextProcessingHandler()
    logger.info("✅ TextProcessingHandler initialized successfully.")
    yield
    logger.info("🛑 Shutting down Text Processor Microservice...")

app = FastAPI(
    title="Text Processor Microservice",
    description="Microservice for full document text processing using NLTK and SymSpell.",
    version="4.0.0",
    lifespan=lifespan
)


class DataRequest(BaseModel):
    dataset_name: str


def get_text_processor() -> TextProcessingHandler:
    return text_processor_instance

ProcessorDependency = Annotated[TextProcessingHandler, Depends(get_text_processor)]

@app.post("/process-data", status_code=202)
def process_data_endpoint(
    request: DataRequest,
    background_tasks: BackgroundTasks,
    processor: ProcessorDependency
):
    
    dataset_name = request.dataset_name
    logger.info(f"📥 Received processing request for dataset: '{dataset_name}'")

    background_tasks.add_task(
        run_database_processing_task,
        dataset_name,
        config.BATCH_SIZE,
        processor
    )

    return {"message": f"✅ Text processing for dataset '{dataset_name}' has been started in the background."}

@app.get("/health", tags=["Health Check"])
def health_check():
    
    return {"status": "🟢 Text Processor is running!"}
