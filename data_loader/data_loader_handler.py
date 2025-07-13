
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any

import utils.config as config
from database.database_handler import DatabaseHandler
from utils.logger_config import logger
import utils.parsers as parsers

class DataLoaderHandler:
   
    def __init__(self, dataset_configs: Dict[str, Any], base_dir: str, batch_size: int):
        self.dataset_configs = dataset_configs
        self.base_dir = base_dir
        self.batch_size = batch_size

    def load_dataset(self, dataset_name: str):
        db_handler = None
        try:
            db_handler = DatabaseHandler(config.MYSQL_CONFIG)
            db_handler.connect()

            db_handler.setup_tables()

            if dataset_name not in self.dataset_configs:
                logger.error(f"Dataset '{dataset_name}' is not defined in the configuration.")
                return

            dataset_config = self.dataset_configs[dataset_name]
            file_path = os.path.join(self.base_dir, dataset_config['file_name'])
            
            if not os.path.exists(file_path):
                logger.error(f"Data file not found at: {file_path}")
                return

            logger.info(f"Starting to load dataset '{dataset_name}' from '{file_path}'...")
            
            parser_func = getattr(parsers, dataset_config['parser_func'])
            dataset_id = db_handler.get_or_create_dataset_id(dataset_name)
            
            if file_path.endswith(".csv"):
                self._load_from_csv(file_path, parser_func, db_handler, dataset_id)
            else:
                self._load_from_tsv(file_path, dataset_config, parser_func, db_handler, dataset_id)
        
        except Exception as e:
            logger.error(f"A critical error occurred during data loading for '{dataset_name}': {e}", exc_info=True)
        finally:
            if db_handler:
                db_handler.disconnect()

    def _load_from_csv(self, file_path: str, parser_func, db_handler: DatabaseHandler, dataset_id: int):
       
        docs_batch, total_inserted = [], 0
        try:
            for chunk in pd.read_csv(file_path, chunksize=self.batch_size, on_bad_lines='skip'):
                for _, row in chunk.iterrows():
                    doc_data, _ = parser_func(row)
                    if doc_data:
                        docs_batch.append(doc_data + (dataset_id,))
                
                if docs_batch:
                    inserted = db_handler.bulk_insert_documents(docs_batch, [], "generic")
                    total_inserted += inserted
                    logger.info(f"Inserted {inserted} documents. Total so far: {total_inserted}")
                    docs_batch = []
        except Exception as e:
            logger.error(f"Failed to process CSV file {file_path}: {e}")
        logger.info(f"Finished loading '{os.path.basename(file_path)}'. Total new documents: {total_inserted}")

    def _load_from_tsv(self, file_path: str, config_dict: dict, parser_func, db_handler: DatabaseHandler, dataset_id: int):
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        questions_map = {}
        if config_dict['metadata_type'] == 'qa_answer':
            logger.info("Pre-processing to map questions to topics for antique_qa...")
            for line in lines:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2 and parts[0].endswith('_0'):
                    questions_map[parts[0].split('_')[0]] = parts[1].strip()

        docs_batch, meta_batch, total_inserted = [], [], 0
        for line in tqdm(lines, desc=f"Loading {os.path.basename(file_path)}"):
            doc_data, meta_data = parser_func(line, questions_map=questions_map)
            if doc_data:
                docs_batch.append(doc_data + (dataset_id,))
                if meta_data:
                    meta_batch.append(meta_data)
            
            if len(docs_batch) >= self.batch_size:
                inserted = db_handler.bulk_insert_documents(docs_batch, meta_batch, config_dict['metadata_type'])
                total_inserted += inserted
                docs_batch, meta_batch = [], []
        
        if docs_batch:
            inserted = db_handler.bulk_insert_documents(docs_batch, meta_batch, config_dict['metadata_type'])
            total_inserted += inserted

        logger.info(f"Finished loading '{os.path.basename(file_path)}'. Total new documents: {total_inserted}")
