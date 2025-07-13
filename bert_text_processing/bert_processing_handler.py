
import re
from typing import Dict
from utils.logger_config import logger

class BertPreprocessingHandler:
    def __init__(self):
        logger.info("Initializing BERT Preprocessing Handler")

    @staticmethod
    def remove_urls(text: str) -> str:
      
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    @staticmethod
    def remove_special_chars(text: str) -> str:
     
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        return text.encode('ascii', 'ignore').decode('ascii')

    @staticmethod
    def normalize_whitespace(text: str) -> str:
      
        return ' '.join(text.split())

    def preprocess_text(self, text: str) -> str:
       
        if not text or not isinstance(text, str):
            return ""

        text = self.remove_urls(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        return text.lower()

    def preprocess_corpus(self, corpus: Dict[str, str]) -> Dict[str, str]:
        
        return {
            doc_id: self.preprocess_text(text)
            for doc_id, text in corpus.items()
        }

    def process_single_text(self, text: str) -> str:
       
        return self.preprocess_text(text)