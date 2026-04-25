"""
Dataset Management Service
Handles user-uploaded datasets including file parsing, validation, and storage
"""

import os
import uuid
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import tempfile
import shutil

from database.database_handler import DatabaseHandler
from utils.logger_config import logger


class DatasetManagementService:
    """Service for managing user-uploaded datasets"""
    
    ALLOWED_EXTENSIONS = {'.csv', '.json', '.txt', '.tsv'}
    SUPPORTED_FORMATS = ['csv', 'json', 'txt', 'tsv']
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.db_handler: Optional[DatabaseHandler] = None
    
    def connect_db(self):
        """Initialize database connection"""
        if not self.db_handler:
            self.db_handler = DatabaseHandler(self.db_config)
            self.db_handler.connect()
            self.db_handler.setup_tables()
    
    def disconnect_db(self):
        """Close database connection"""
        if self.db_handler:
            self.db_handler.disconnect()
            self.db_handler = None
    
    def validate_file(self, file_path: str, file_extension: str) -> Tuple[bool, str]:
        """
        Validate uploaded file format and content
        
        Args:
            file_path: Path to the uploaded file
            file_extension: File extension
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if file_extension.lower() not in self.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type: {file_extension}. Allowed: {self.SUPPORTED_FORMATS}"
        
        try:
            if file_extension.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=10)
                if len(df.columns) < 2:
                    return False, "CSV must have at least 2 columns (doc_id and text)"
                if 'text' not in df.columns.str.lower() and df.shape[1] < 2:
                    logger.warning("CSV doesn't have 'text' column, assuming second column is text")
                    
            elif file_extension.lower() == '.json':
                df = pd.read_json(file_path, nrows=10)
                if len(df.columns) < 2:
                    return False, "JSON must have at least 2 fields (doc_id and text)"
                    
            elif file_extension.lower() in ['.txt', '.tsv']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline() for _ in range(10)]
                    if not first_lines:
                        return False, "File is empty"
                        
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
        
        return True, ""
    
    def parse_uploaded_file(self, file_path: str, file_extension: str, 
                           dataset_name: str) -> List[Tuple]:
        """
        Parse uploaded file into document tuples
        
        Args:
            file_path: Path to the uploaded file
            file_extension: File extension
            dataset_name: Name of the dataset
            
        Returns:
            List of document tuples (doc_id, raw_text, processed_text, dataset_id)
        """
        documents = []
        
        try:
            if file_extension.lower() == '.csv':
                df = pd.read_csv(file_path, on_bad_lines='skip')
                # Assume first column is doc_id, second is text (or look for 'text' column)
                text_col = None
                id_col = None
                
                for col in df.columns:
                    if col.lower() == 'text' or col.lower() == 'content':
                        text_col = col
                    elif col.lower() == 'doc_id' or col.lower() == 'id':
                        id_col = col
                
                if not text_col:
                    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                if not id_col:
                    id_col = df.columns[0]
                
                for idx, row in df.iterrows():
                    doc_id = str(row[id_col]) if id_col else f"{dataset_name}_{idx}"
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    if text.strip():
                        documents.append((doc_id, text.strip(), "", None))
                        
            elif file_extension.lower() == '.json':
                df = pd.read_json(file_path)
                text_col = None
                id_col = None
                
                for col in df.columns:
                    if col.lower() == 'text' or col.lower() == 'content':
                        text_col = col
                    elif col.lower() == 'doc_id' or col.lower() == 'id':
                        id_col = col
                
                if not text_col:
                    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                if not id_col:
                    id_col = df.columns[0]
                
                for idx, row in df.iterrows():
                    doc_id = str(row[id_col]) if id_col else f"{dataset_name}_{idx}"
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    if text.strip():
                        documents.append((doc_id, text.strip(), "", None))
                        
            elif file_extension.lower() in ['.txt', '.tsv']:
                delimiter = '\t' if file_extension.lower() == '.tsv' else None
                with open(file_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        
                        if delimiter:
                            parts = line.split(delimiter, 1)
                            if len(parts) >= 2:
                                doc_id, text = parts[0].strip(), parts[1].strip()
                            else:
                                doc_id, text = f"{dataset_name}_{idx}", parts[0].strip()
                        else:
                            doc_id, text = f"{dataset_name}_{idx}", line
                        
                        if text:
                            documents.append((doc_id, text, "", None))
                            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise
        
        return documents
    
    def register_dataset(self, dataset_name: str) -> int:
        """
        Register a new dataset in the database
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset ID
        """
        self.connect_db()
        try:
            dataset_id = self.db_handler.get_or_create_dataset_id(dataset_name)
            logger.info(f"Registered dataset '{dataset_name}' with ID {dataset_id}")
            return dataset_id
        finally:
            self.disconnect_db()
    
    def insert_documents(self, documents: List[Tuple], dataset_id: int) -> int:
        """
        Insert documents into the database
        
        Args:
            documents: List of document tuples
            dataset_id: Dataset ID
            
        Returns:
            Number of inserted documents
        """
        self.connect_db()
        try:
            # Add dataset_id to each document tuple
            docs_with_dataset = []
            for doc in documents:
                doc_id, raw_text, processed_text, _ = doc
                docs_with_dataset.append((doc_id, raw_text, processed_text, dataset_id))
            
            inserted = self.db_handler.bulk_insert_documents(docs_with_dataset, [], "generic")
            logger.info(f"Inserted {inserted} documents for dataset ID {dataset_id}")
            return inserted
        finally:
            self.disconnect_db()
    
    def get_dataset_status(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get status of a dataset including document counts and processing status
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset status information
        """
        self.connect_db()
        try:
            dataset_id = self.db_handler.get_or_create_dataset_id(dataset_name)
            
            # Get total document count
            query = "SELECT COUNT(*) as total FROM documents WHERE dataset_id = %s"
            self.db_handler.cursor.execute(query, (dataset_id,))
            total_docs = self.db_handler.cursor.fetchone()['total']
            
            # Get processed count
            query_processed = """
                SELECT COUNT(*) as processed FROM documents 
                WHERE dataset_id = %s AND processed_text IS NOT NULL AND processed_text != ''
            """
            self.db_handler.cursor.execute(query_processed, (dataset_id,))
            processed_docs = self.db_handler.cursor.fetchone()['processed']
            
            # Get BERT processed count
            query_bert = """
                SELECT COUNT(*) as bert_processed FROM documents 
                WHERE dataset_id = %s AND bert_processed_text IS NOT NULL AND bert_processed_text != ''
            """
            self.db_handler.cursor.execute(query_bert, (dataset_id,))
            bert_processed = self.db_handler.cursor.fetchone()['bert_processed']
            
            return {
                "dataset_name": dataset_name,
                "dataset_id": dataset_id,
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "bert_processed_documents": bert_processed,
                "processing_progress": (processed_docs / total_docs * 100) if total_docs > 0 else 0,
                "bert_processing_progress": (bert_processed / total_docs * 100) if total_docs > 0 else 0,
                "ready_for_search": processed_docs > 0 and bert_processed > 0
            }
        except Exception as e:
            logger.error(f"Error getting dataset status: {e}")
            return {"error": str(e)}
        finally:
            self.disconnect_db()
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets in the system
        
        Returns:
            List of dataset information dictionaries
        """
        self.connect_db()
        try:
            query = """
                SELECT d.id, d.name, COUNT(doc.id) as doc_count
                FROM datasets d
                LEFT JOIN documents doc ON d.id = doc.dataset_id
                GROUP BY d.id, d.name
            """
            self.db_handler.cursor.execute(query)
            results = self.db_handler.cursor.fetchall()
            
            datasets = []
            for row in results:
                datasets.append({
                    "id": row['id'],
                    "name": row['name'],
                    "document_count": row['doc_count']
                })
            
            return datasets
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []
        finally:
            self.disconnect_db()
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset and all its documents
        
        Args:
            dataset_name: Name of the dataset to delete
            
        Returns:
            True if successful, False otherwise
        """
        self.connect_db()
        try:
            dataset_id = self.db_handler.get_or_create_dataset_id(dataset_name)
            
            # Delete documents first (foreign key constraint)
            delete_docs = "DELETE FROM documents WHERE dataset_id = %s"
            self.db_handler.cursor.execute(delete_docs, (dataset_id,))
            
            # Delete dataset
            delete_dataset = "DELETE FROM datasets WHERE id = %s"
            self.db_handler.cursor.execute(delete_dataset, (dataset_id,))
            
            self.db_handler.connection.commit()
            logger.info(f"Deleted dataset '{dataset_name}' and all its documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            self.db_handler.connection.rollback()
            return False
        finally:
            self.disconnect_db()
