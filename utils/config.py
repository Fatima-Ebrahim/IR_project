# config.py
import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'database': os.getenv('DB_NAME', 'ir_test_2')
}

# --- الإعدادات العامة للمشروع ---
BATCH_SIZE = 500
CPU_CORES = os.cpu_count() or 1 # عدد أنوية المعالج للمعالجة المتوازية

# --- مسارات المشروع ---
# المسار الأساسي الذي يحتوي على مجلدات المشروع
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# مسار مجلد مجموعات البيانات
# DATASETS_BASE_DIR = "E:/information-retrieval-project/datasets" # <--- ❗️ عدّل هذا المسار
DATASETS_BASE_DIR = "E:/information_test/datasets" # <--- ❗️ عدّل هذا المسار
# مسار قاموس التصحيح الإملائي
# قم بتنزيل القاموس وضعه في مسار معروف
SYMPSPELL_DICT_PATH = os.path.join(DATASETS_BASE_DIR, "symspell_dictionary", "frequency_dictionary_en_82_765.txt") # <--- ❗️ تأكد من صحة هذا المسار

# --- تعريف مجموعات البيانات ---
# هذا القاموس يحدد كل مجموعة بيانات وكيفية تحليلها
# 'parser_func': اسم الدالة المستخدمة للتحليل من ملف utils/parsers.py
# 'id_pattern': تعبير نمطي للتعرف على بداية مستند جديد في الملف
# 'metadata_type': نوع البيانات الوصفية (إن وجدت)
DATASET_CONFIGS = {
    "antique_qa": {
        "file_name": "antique/collection.tsv",
        "parser_func": "parse_qa_line",
        "id_pattern": r'^\d+_\d+\t',
        "metadata_type": "qa_answer"
    },
    # "clinical_trials": {
    #     "file_name": "clinicaltrials/collection.tsv",
    #     "parser_func": "parse_generic_line",
    #     "id_pattern": r'^NCT\d{7,}\t',
    #     "metadata_type": "generic"
    # },
    "wikIR1k": {
        "file_name": "wikIR1k/documents.csv", # نفترض أنه ملف CSV
        "parser_func": "parse_wikir_csv_line",
        "id_pattern": None, # لا نحتاج نمطًا لملفات CSV
        "metadata_type": "generic"
    },
    # "wiki": {
    #     "file_name": "wiki_test.csv", # نفترض أنه ملف CSV
    #     "parser_func": "parse_wikir_csv_line",
    #     "id_pattern": None, # لا نحتاج نمطًا لملفات CSV
    #     "metadata_type": "generic"
    # }
}
# OUTPUT_DIR = "E:/information-retrieval-project/saved_models"
OUTPUT_DIR = "E:/information_test/saved_models"
# --- Service URLs ---
# This is the new, centralized location for all service addresses
GATEWAY_URL = "http://127.0.0.1:8000"
TFIDF_SEARCH_URL = "http://127.0.0.1:8031"
BERT_SEARCH_URL = "http://127.0.0.1:8034"
HYBRID_SEARCH_URL = "http://127.0.0.1:8035"
# Add other service URLs here if needed