# handlers/mlm_expansion_handler.py
from transformers import pipeline, logging as hf_logging
from nltk.corpus import stopwords
import string
from typing import List, Set

from utils.logger_config import logger

hf_logging.set_verbosity_error()

class MlmExpansionHandler:
   
    def __init__(self, model_name: str = 'bert-base-uncased'):
        logger.info(f"Initializing MLMQueryExpander with model: {model_name}")
        try:
            self.mlm_pipeline = pipeline('fill-mask', model=model_name)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Failed to load MLM model '{model_name}'. Error: {e}")
            raise RuntimeError(f"Could not initialize MLM model: {e}")

    def expand(self, query: str, top_k: int = 5) -> dict:
        
        if not query:
            return {"expanded_query": "", "expansion_terms": []}

        templates = [
            f"{query} is related to [MASK].",
            f"The concept of {query} involves [MASK].",
            f"{query} and [MASK] are often discussed together."
        ]
        
        expansion_terms: Set[str] = set()
        
        for template in templates:
            try:
                results = self.mlm_pipeline(template, top_k=top_k)
                for res in results:
                    term = res['token_str'].strip().lower()
                    
                    if (term and term not in query.lower().split() and
                        term not in self.stop_words and
                        not all(char in string.punctuation for char in term)):
                        expansion_terms.add(term)
            except Exception as e:
                logger.warning(f"MLM pipeline failed for template '{template}'. Error: {e}")
                continue
        
        original_terms = set(query.lower().split())
        final_terms = original_terms.union(expansion_terms)
        expanded_query = " ".join(sorted(list(final_terms)))
        
        logger.info(f"Original Query: '{query}' -> Expanded Query: '{expanded_query}'")

        return {
            "expanded_query": expanded_query,
            "expansion_terms": list(expansion_terms)
        }
