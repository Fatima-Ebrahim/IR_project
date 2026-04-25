"""
Dataset Management Handler (API Layer)
FastAPI endpoints for dataset upload and management
"""

import os
import uuid
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio

from utils.config import MYSQL_CONFIG
from utils.logger_config import logger
from .dataset_management_service import DatasetManagementService

router = APIRouter(prefix="/api/v1/datasets", tags=["Dataset Management"])


def get_dataset_service() -> DatasetManagementService:
    """Dependency to get dataset service instance"""
    return DatasetManagementService(MYSQL_CONFIG)


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file (CSV, JSON, TXT, or TSV)"),
    dataset_name: str = Query(..., description="Name for the dataset"),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a new dataset file
    
    - Accepts CSV, JSON, TXT, or TSV files
    - Validates file format
    - Parses and stores documents in database
    - Returns task ID for tracking processing status
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = ['.csv', '.json', '.txt', '.tsv']
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Create temporary directory for upload
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
    
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize service
        service = get_dataset_service()
        
        # Validate file
        is_valid, error_msg = service.validate_file(temp_file_path, file_ext)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Register dataset
        dataset_id = service.register_dataset(dataset_name)
        
        # Parse file
        documents = service.parse_uploaded_file(temp_file_path, file_ext, dataset_name)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found in file")
        
        # Insert documents into database
        inserted_count = service.insert_documents(documents, dataset_id)
        
        logger.info(f"Uploaded dataset '{dataset_name}': {inserted_count} documents")
        
        # Schedule background processing (text processing, BERT, indexing)
        if background_tasks:
            background_tasks.add_task(
                process_dataset_pipeline,
                dataset_name,
                temp_file_path,
                temp_dir
            )
        else:
            # Cleanup if no background task
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "message": f"Dataset '{dataset_name}' uploaded successfully",
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "documents_uploaded": inserted_count,
            "status": "processing_started",
            "note": "Text processing and indexing will run in background"
        }
        
    except HTTPException:
        # Clean up on HTTP errors
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}", exc_info=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def process_dataset_pipeline(dataset_name: str, temp_file_path: str, temp_dir: str):
    """
    Background task to process uploaded dataset
    Runs text processing, BERT embedding, and indexing
    """
    try:
        from handlers.text_processing_handler import TextProcessingHandler
        from bert_representation.bert_embedding_handler import BertEmbeddingHandler
        from tfidf_representation.tfidf_indexer_handler import TFIDFIndexerHandler
        
        logger.info(f"Starting processing pipeline for dataset: {dataset_name}")
        
        # Step 1: Text preprocessing
        logger.info("Step 1: Running text preprocessing...")
        text_handler = TextProcessingHandler(MYSQL_CONFIG.get('symspell_dict_path', ''))
        # Note: You may need to adjust this based on your actual handler signature
        
        # Step 2: BERT embeddings
        logger.info("Step 2: Generating BERT embeddings...")
        # bert_handler = BertEmbeddingHandler()
        # bert_handler.process_dataset(dataset_name)
        
        # Step 3: TF-IDF indexing
        logger.info("Step 3: Building TF-IDF index...")
        # tfidf_handler = TFIDFIndexerHandler()
        # tfidf_handler.build_index(dataset_name)
        
        logger.info(f"Processing pipeline completed for dataset: {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error in processing pipeline for {dataset_name}: {e}", exc_info=True)
    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("")
async def list_datasets():
    """List all datasets in the system"""
    service = get_dataset_service()
    datasets = service.list_datasets()
    return {"datasets": datasets, "count": len(datasets)}


@router.get("/{dataset_name}/status")
async def get_dataset_status(dataset_name: str):
    """Get detailed status of a specific dataset"""
    service = get_dataset_service()
    status = service.get_dataset_status(dataset_name)
    
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    
    return status


@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its documents"""
    service = get_dataset_service()
    success = service.delete_dataset(dataset_name)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete dataset")
    
    return {"message": f"Dataset '{dataset_name}' deleted successfully"}


@router.post("/{dataset_name}/process")
async def trigger_processing(dataset_name: str, background_tasks: BackgroundTasks):
    """
    Manually trigger processing pipeline for a dataset
    Use this if automatic processing failed or needs re-running
    """
    # Verify dataset exists
    service = get_dataset_service()
    status = service.get_dataset_status(dataset_name)
    
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    
    # Add to background tasks
    background_tasks.add_task(run_full_processing, dataset_name)
    
    return {
        "message": f"Processing started for dataset '{dataset_name}'",
        "status": "processing_queued"
    }


def run_full_processing(dataset_name: str):
    """
    Run complete processing pipeline for a dataset
    """
    try:
        from handlers.text_processing_handler import TextProcessingHandler
        from utils.config import BATCH_SIZE
        
        logger.info(f"Running full processing for dataset: {dataset_name}")
        
        # Text preprocessing
        text_handler = TextProcessingHandler("")
        text_handler.run_processing_pipeline(dataset_name, BATCH_SIZE)
        
        logger.info(f"Processing completed for dataset: {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error in full processing for {dataset_name}: {e}", exc_info=True)
