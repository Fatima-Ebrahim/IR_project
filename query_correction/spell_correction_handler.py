import re
import os
from symspellpy import SymSpell
from utils import config
from utils.logger_config import logger

class SpellCorrectionHandler:
    def __init__(self):
        self.sym_spell = self._setup_symspell(config.SYMPSPELL_DICT_PATH)
        
    def _setup_symspell(self, path: str) -> SymSpell:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        if not os.path.exists(path):
            logger.warning(f"SymSpell dictionary not found at {path}")
            return sym_spell
        
        if not sym_spell.load_dictionary(path, term_index=0, count_index=1, encoding='utf-8'):
            logger.warning(f"Failed to load SymSpell dictionary from {path}")
        
        return sym_spell

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def correct_spelling(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return text
            
        cleaned_text = self.clean_text(text)
        is_upper = text.isupper()
        is_title = text.istitle()
        
        suggestions = self.sym_spell.lookup_compound(
            cleaned_text, 
            max_edit_distance=2,
            transfer_casing=True
        )
        
        if suggestions:
            corrected_text = suggestions[0].term
            if is_upper:
                corrected_text = corrected_text.upper()
            elif is_title:
                corrected_text = corrected_text.title()
            return corrected_text
        
        return cleaned_text