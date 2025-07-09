# # services/text_processor_service.py
# from fastapi import FastAPI, BackgroundTasks, Depends
# from pydantic import BaseModel
# from contextlib import asynccontextmanager
# from typing import Annotated
# from tqdm import tqdm

# import utils.config as config
# # استيراد المكونات الجديدة من المعالج
# from text_processing.text_processing_handler import TextProcessor, process_text_pipeline
# from database.database_handler import DatabaseHandler
# from utils.logger_config import logger

# # متغير عام سيحتوي على الكائن الوحيد لمعالج النصوص
# text_processor_instance = None

# def run_database_processing_task(dataset_name: str, batch_size: int, processor: TextProcessor):
#     """
#     (جديد)
#     هذه الدالة تعمل في الخلفية وتقوم بتنظيم عملية المعالجة بأكملها:
#     الاتصال بقاعدة البيانات، جلب البيانات، معالجتها، ثم تحديثها.
#     """
#     db_handler = None
#     try:
#         db_handler = DatabaseHandler(config.MYSQL_CONFIG)
#         db_handler.connect()
#         db_handler.setup_tables()

#         logger.info(f"Starting background processing pipeline for dataset: '{dataset_name}'")
#         total_processed_count = 0
#         while True:
#             docs_to_process = db_handler.get_unprocessed_docs(dataset_name, batch_size)
#             if not docs_to_process:
#                 logger.info("No more unprocessed documents found. Pipeline complete.")
#                 break

#             logger.info(f"Processing a batch of {len(docs_to_process)} documents...")
            
#             # معالجة كل نص باستخدام دالة المعالجة الجديدة
#             processed_texts = [process_text_pipeline(doc['raw_text'], processor) for doc in tqdm(docs_to_process, desc="Processing Batch")]

#             # تجهيز البيانات للتحديث
#             updates = [(processed, docs_to_process[i]['id']) for i, processed in enumerate(processed_texts)]
            
#             updated_count = db_handler.bulk_update_processed_text(updates)
#             total_processed_count += updated_count
#             logger.info(f"Successfully updated {updated_count} documents. Total processed: {total_processed_count}")
    
#     except Exception as e:
#         logger.error(f"A critical error occurred during background processing for '{dataset_name}': {e}", exc_info=True)
#     finally:
#         if db_handler:
#             db_handler.disconnect()
#             logger.info("MySQL connection closed.")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     يقوم بإنشاء كائن معالج النصوص مرة واحدة عند بدء تشغيل الخدمة.
#     """
#     global text_processor_instance
#     logger.info("Service is starting up...")
#     text_processor_instance = TextProcessor()
#     logger.info("TextProcessor initialized successfully.")
#     yield
#     logger.info("Service is shutting down.")

# app = FastAPI(
#     title="Text Processor Microservice",
#     description="A simplified microservice for text processing.",
#     version="3.0.0",
#     lifespan=lifespan
# )

# class DataRequest(BaseModel):
#     dataset_name: str

# def get_text_processor() -> TextProcessor:
#     """
#     دالة حقن التبعية التي تعيد الكائن المشترك.
#     """
#     return text_processor_instance

# ProcessorDependency = Annotated[TextProcessor, Depends(get_text_processor)]

# @app.post("/process-data", status_code=202)
# def process_data_endpoint(
#     request: DataRequest,
#     background_tasks: BackgroundTasks,
#     processor: ProcessorDependency
# ):
#     """
#     يبدأ عملية معالجة النصوص في الخلفية.
#     """
#     dataset_name = request.dataset_name
#     logger.info(f"Received request for dataset: '{dataset_name}'. Task added to background.")
    
#     # إضافة مهمة معالجة قاعدة البيانات إلى الخلفية
#     background_tasks.add_task(
#         run_database_processing_task,
#         dataset_name,
#         config.BATCH_SIZE,
#         processor # تمرير كائن المعالج المشترك
#     )

#     return {"message": f"Text processing for '{dataset_name}' has been started in the background."}

# @app.get("/health", tags=["Health Check"])
# def health_check():
#     return {"status": "Simplified Text Processor Service is running!"}
#todo الكود يلي قبل شغال تمام مع النسخة يلي عليها اورانج من الهاندلر 
    # services/text_processor_service.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from utils import config
# --- (التعديل) استيراد الكلاس الجديد بالاسم الصحيح ---
from text_processing.text_processing_handler import TextProcessingHandler
from database.database_handler import DatabaseHandler
from utils.logger_config import logger

app = FastAPI(
    title="Text Processor Microservice",
    description="A microservice for advanced text processing.",
    version="2.0.0"
)

class DataRequest(BaseModel):
    dataset_name: str

def run_processing_task(dataset_name: str):
    """
    The background task for processing a dataset.
    It creates its own handler instances.
    """
    try:
        # --- (التعديل) إنشاء نسخة من الكلاس الجديد ---
        # لم نعد بحاجة لتمرير المعالج، المهمة تنشئه بنفسها
        processing_handler = TextProcessingHandler() # لا يحتاج لمسار القاموس لأنه يقرأه من config
        processing_handler.run_corpus_processing(
            dataset_name=dataset_name,
            batch_size=config.BATCH_SIZE,
            num_cores=config.CPU_CORES
        )
    except Exception as e:
        logger.error(f"Failed to run background processing task for {dataset_name}: {e}", exc_info=True)


@app.post("/process-data", status_code=202)
def process_data_endpoint(request: DataRequest, background_tasks: BackgroundTasks):
    """
    Triggers the text processing pipeline in the background.
    """
    dataset_name = request.dataset_name
    logger.info(f"Received request to process dataset: '{dataset_name}'. Task added to background.")
    
    # --- (التعديل) استدعاء الدالة الجديدة للمهمة الخلفية ---
    background_tasks.add_task(run_processing_task, dataset_name)

    return {"message": f"Text processing for '{dataset_name}' has been started in the background."}

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "Text Processor Service is running!"}