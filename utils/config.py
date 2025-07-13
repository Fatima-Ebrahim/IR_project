import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'database': os.getenv('DB_NAME', 'ir_test_2')
    # 'database': os.getenv('DB_NAME', 'information_retrieval_project')
}

BATCH_SIZE = 100000
CPU_CORES = os.cpu_count() or 1

BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_BASE_DIR = os.path.join(BASE_PROJECT_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "saved_models")

SYMPSPELL_DICT_PATH = os.path.join(DATASETS_BASE_DIR, "symspell_dictionary", "frequency_dictionary_en_82_765.txt")

DATASET_CONFIGS = {
    "antique": {
        "file_name": "antique/collection.tsv",
        "parser_func": "parse_qa_line",
        "id_pattern": r'^\d+_\d+\t',
        "metadata_type": "qa_answer"
    },
    "antique_qa": {
        "file_name": "antique_qa/collection.tsv",
        "parser_func": "parse_qa_line",
        "id_pattern": r'^\d+_\d+\t',
        "metadata_type": "qa_answer"
    },
    "wikIR1k": {
        "file_name": "wikIR1k/documents.csv",
        "parser_func": "parse_wikir_csv_line",
        "id_pattern": None,
        "metadata_type": "generic"
    }
}

GATEWAY_URL = "http://127.0.0.1:8000"
TFIDF_SEARCH_URL = "http://127.0.0.1:8031"
BERT_SEARCH_URL = "http://127.0.0.1:8034"
HYBRID_SEARCH_URL = "http://127.0.0.1:8035"
QUERY_PREPROCESSOR_URL = "http://127.0.0.1:8044/preprocess-query"
# QUERY_PREPROCESSOR_BERT_URL = "http://127.0.0.1:8008/preprocess-bert-query"