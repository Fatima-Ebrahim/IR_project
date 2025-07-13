import mysql.connector
from mysql.connector import errorcode
from typing import List, Dict, Any, Tuple
from utils.logger_config import logger

class DatabaseHandler:
   
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
        self.cursor = None

    def connect(self):
        if self.connection and self.connection.is_connected():
            return
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info("Successfully connected to MySQL database.")
        except mysql.connector.Error as err:
            logger.error(f"Failed to connect to MySQL: {err}")
            raise

    def disconnect(self):
        if self.cursor: self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed.")

    def setup_tables(self):
        tables = {
            "datasets": """
                CREATE TABLE IF NOT EXISTS `datasets` (
                  `id` INT AUTO_INCREMENT PRIMARY KEY,
                  `name` VARCHAR(255) NOT NULL UNIQUE
                ) ENGINE=InnoDB
            """,
            "documents": """
                CREATE TABLE IF NOT EXISTS `documents` (
                  `id` INT AUTO_INCREMENT PRIMARY KEY,
                  `doc_id` VARCHAR(255) NOT NULL,
                  `raw_text` LONGTEXT,
                  `processed_text` LONGTEXT,
                  `bert_processed_text` LONGTEXT,  -- Added new column for BERT
                  `dataset_id` INT,
                  `last_processed_at` TIMESTAMP NULL DEFAULT NULL,
                  `bert_processed_at` TIMESTAMP NULL DEFAULT NULL,  -- Added new column
                  FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`id`) ON DELETE CASCADE,
                  UNIQUE KEY `doc_dataset_unique` (`doc_id`, `dataset_id`)
                ) ENGINE=InnoDB
            """,
            "metadata_qa": """
                CREATE TABLE IF NOT EXISTS `metadata_qa` (
                  `document_id` INT PRIMARY KEY,
                  `topic_id` VARCHAR(255),
                  `question_text` LONGTEXT,
                  FOREIGN KEY (`document_id`) REFERENCES `documents`(`id`) ON DELETE CASCADE
                ) ENGINE=InnoDB
            """
        }
        logger.info("Ensuring all tables exist with the correct schema...")
        for name, ddl in tables.items():
            self.cursor.execute(ddl)
        self.connection.commit()
        logger.info("Tables setup complete.")

    def get_or_create_dataset_id(self, dataset_name: str) -> int:
        self.cursor.execute("SELECT id FROM datasets WHERE name = %s", (dataset_name,))
        result = self.cursor.fetchone()
        if result:
            return result['id']
        else:
            self.cursor.execute("INSERT INTO datasets (name) VALUES (%s)", (dataset_name,))
            self.connection.commit()
            return self.cursor.lastrowid

    def bulk_insert_documents(self, documents: List[Tuple], metadata: List[Tuple], metadata_type: str) -> int:
        if not documents: return 0
        doc_sql = "INSERT INTO documents (doc_id, raw_text, processed_text, dataset_id) VALUES (%s, %s, %s, %s)"
        meta_sql_map = {'qa_answer': "INSERT INTO metadata_qa (document_id, topic_id, question_text) VALUES (%s, %s, %s)"}
        inserted_count = 0
        try:
            self.connection.start_transaction()
            for i, doc_data in enumerate(documents):
                try:
                    self.cursor.execute(doc_sql, doc_data)
                    doc_pk = self.cursor.lastrowid
                    meta_sql = meta_sql_map.get(metadata_type)
                    if meta_sql and i < len(metadata):
                        meta_values = (doc_pk,) + metadata[i]
                        self.cursor.execute(meta_sql, meta_values)
                    inserted_count += 1
                except mysql.connector.Error as err:
                    if err.errno != errorcode.ER_DUP_ENTRY:
                        logger.warning(f"Skipping insert for doc '{doc_data[0]}': {err}")
            self.connection.commit()
        except mysql.connector.Error as err:
            logger.error(f"Transaction failed, rolling back. Error: {err}")
            self.connection.rollback()
        return inserted_count

    def get_unprocessed_docs(self, dataset_name: str, batch_size: int) -> List[Dict[str, Any]]:
        dataset_id = self.get_or_create_dataset_id(dataset_name)
        query = """
            SELECT id, raw_text FROM documents 
            WHERE dataset_id = %s 
            AND (processed_text IS NULL OR processed_text = '') 
            AND last_processed_at IS NULL 
            LIMIT %s
        """
        self.cursor.execute(query, (dataset_id, batch_size))
        return self.cursor.fetchall()

    def bulk_update_processed_text(self, updates: List[Tuple[str, int]]) -> int:
        if not updates: return 0
        query = "UPDATE documents SET processed_text = %s, last_processed_at = NOW() WHERE id = %s"
        try:
            self.cursor.executemany(query, updates)
            self.connection.commit()
            return self.cursor.rowcount
        except mysql.connector.Error as err:
            logger.error(f"Bulk update failed with a database error: {err}")
            self.connection.rollback()
            return 0

    def get_raw_docs_for_indexing(self, dataset_name: str) -> List[Dict[str, Any]]:
        logger.info(f"Fetching RAW documents for dataset: {dataset_name}")
        dataset_id = self.get_or_create_dataset_id(dataset_name)
        query = "SELECT id, raw_text FROM documents WHERE dataset_id = %s AND raw_text IS NOT NULL AND raw_text != ''"
        self.cursor.execute(query, (dataset_id,))
        docs = self.cursor.fetchall()
        logger.info(f"Found {len(docs)} raw documents.")
        return docs

    def get_processed_docs_for_indexing(self, dataset_name: str) -> List[Dict[str, Any]]:
        query = """
            SELECT id, bert_processed_text AS processed_text 
            FROM documents 
            WHERE dataset_id = (SELECT id FROM datasets WHERE name = %s) 
            AND bert_processed_text IS NOT NULL 
            AND bert_processed_text != ''
            """
        self.cursor.execute(query, (dataset_name,))
        docs = self.cursor.fetchall()
        print(f"📄 Found {len(docs)} BERT-processed documents for indexing.")
        return docs
    
    def find_documents_by_ids(self, doc_ids: List[int]) -> Dict[str, Dict[str, Any]]:
        if not doc_ids:
            return {}

        try:
            format_strings = ','.join(['%s'] * len(doc_ids))
            query = f"SELECT id, doc_id, raw_text FROM documents WHERE id IN ({format_strings})"
            self.cursor.execute(query, tuple(doc_ids))
            results = self.cursor.fetchall()
            return {str(row['id']): row for row in results}
        except mysql.connector.Error as err:
            logger.error(f"Database error in find_documents_by_ids: {err}")
            return {}

    def get_all_raw_docs(self, dataset_name: str) -> List[Dict[str, Any]]:
        logger.info(f"Fetching ALL RAW documents for dataset: {dataset_name}")
        dataset_id = self.get_or_create_dataset_id(dataset_name)
        query = "SELECT id, doc_id, raw_text FROM documents WHERE dataset_id = %s AND raw_text IS NOT NULL AND raw_text != ''"
        self.cursor.execute(query, (dataset_id,))
        docs = self.cursor.fetchall()
        logger.info(f"Found {len(docs)} total raw documents for dataset '{dataset_name}'.")
        return docs

    def count_unprocessed_docs(self, dataset_name: str) -> int:
        dataset_id = self.get_or_create_dataset_id(dataset_name)
        query = """
            SELECT COUNT(*) AS count FROM documents 
            WHERE dataset_id = %s 
            AND (processed_text IS NULL OR processed_text = '') 
            AND last_processed_at IS NULL
        """
        self.cursor.execute(query, (dataset_id,))
        result = self.cursor.fetchone()
        return result['count'] if result else 0
    def create_topic_tables(self):
            
            topic_tables = {
                "topics": """
                    CREATE TABLE IF NOT EXISTS `topics` (
                        id INT PRIMARY KEY,
                        keywords TEXT
                    ) ENGINE=InnoDB
                """,
                "document_topics": """
                    CREATE TABLE IF NOT EXISTS `document_topics` (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        document_id INT,
                        topic_id INT,
                        probability FLOAT,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB
                """
            }
            for ddl in topic_tables.values():
                self.cursor.execute(ddl)
            self.connection.commit()

    def save_lda_topics(self, topic_keywords: List[Tuple[int, str]]):
        self.cursor.executemany("INSERT INTO topics (id, keywords) VALUES (%s, %s) ON DUPLICATE KEY UPDATE keywords=VALUES(keywords)", topic_keywords)
        self.connection.commit()

    def save_document_topics(self, topics: List[Tuple[int, int, float]]):
        
        self.cursor.executemany(
            "INSERT INTO document_topics (document_id, topic_id, probability) VALUES (%s, %s, %s)",
            topics
        )
        self.connection.commit()
        
    def get_unprocessed_bert_docs(self, dataset_name: str, batch_size: int) -> List[Dict[str, Any]]:
        dataset_id = self.get_or_create_dataset_id(dataset_name)
        query = """
            SELECT id, raw_text FROM documents 
            WHERE dataset_id = %s 
            AND (bert_processed_text IS NULL OR bert_processed_text = '') 
            AND bert_processed_at IS NULL 
            LIMIT %s
        """
        self.cursor.execute(query, (dataset_id, batch_size))
        return self.cursor.fetchall()

    def bulk_update_bert_processed_text(self, updates: List[Tuple[str, int]]) -> int:
        if not updates: return 0
        query = "UPDATE documents SET bert_processed_text = %s, bert_processed_at = NOW() WHERE id = %s"
        try:
            self.cursor.executemany(query, updates)
            self.connection.commit()
            return self.cursor.rowcount
        except mysql.connector.Error as err:
            logger.error(f"Bulk BERT update failed with a database error: {err}")
            self.connection.rollback()
            return 0

    def count_unprocessed_bert_docs(self, dataset_name: str) -> int:
        dataset_id = self.get_or_create_dataset_id(dataset_name)
        query = """
            SELECT COUNT(*) AS count FROM documents 
            WHERE dataset_id = %s 
            AND (bert_processed_text IS NULL OR bert_processed_text = '') 
            AND bert_processed_at IS NULL
        """
        self.cursor.execute(query, (dataset_id,))
        result = self.cursor.fetchone()
        return result['count'] if result else 0