import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger
from symspellpy import SymSpell
from typing import List, Set
from multiprocessing import Pool
from tqdm import tqdm

from utils import config
from database.database_handler import DatabaseHandler
from utils.logger_config import logger

worker_tagger = None

def worker_initializer():
    global worker_tagger
    logger.info(f"Initializing worker process [PID: {os.getpid()}]...")
    packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            nltk.data.find(
                f'tokenizers/{package}' if package == 'punkt' else
                f'taggers/{package}' if 'tagger' in package else
                f'corpora/{package}'
            )
        except LookupError:
            logger.info(f"Worker [PID: {os.getpid()}] downloading NLTK package: {package}...")
            nltk.download(package, quiet=True)
    
    worker_tagger = PerceptronTagger()
    logger.info(f"Worker process [PID: {os.getpid()}] initialized successfully.")


class TextProcessingHandler:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        negation_words = {'no', 'not', 'nor', 'never'}
        self.stop_words: Set[str] = set(stopwords.words('english'))
        additional_stopwords = {
        'not', 'because', 'so', 'even', 'only', 'why', 'well', 'yet', 'would',
        'should', 'every', 'but', 'yeah', 'this', 'ok', 'oh', 'wow', 'me',
        'debate', 'theme', 'round', 'person', 'rebuttal',
        'accept', 'response', 'con', 'pro', 'side',
        'say', 'make', 'get', 'like', 'one', 'another', 'might', 'also', 'yes',
        'hey', 'you', 'we', 'i', 'it', 'that', 'if', 'what', 'just', 'one', 'come',
        'go', 'get', 'got', 'make', 'could', 'also', 'still', 'though', 'rather', 'like',
        'YeahNo', 'gotta', 'whatever', 'somewhat', 'whats', 'becauses', 'becaused', 'u'}
        self.stop_words = self.stop_words.union(additional_stopwords).difference(negation_words)
        self.pos_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
        self.sym_spell = self._setup_symspell(config.SYMPSPELL_DICT_PATH)

    def _setup_symspell(self, path: str) -> SymSpell:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        if not os.path.exists(path):
            logger.warning(f"SymSpell dictionary not found at {path}. Spell correction will be skipped.")
            return sym_spell
        if not sym_spell.load_dictionary(path, term_index=0, count_index=1, encoding='utf-8'):
            logger.warning(f"SymSpell dictionary at {path} could not be loaded.")
        return sym_spell

    def _get_wordnet_pos(self, tag: str) -> str:
        return self.pos_map.get(tag[0].upper(), wordnet.NOUN)

    def get_spell_corrected_text(self, text: str) -> str:

        if not isinstance(text, str) or not text:
            return ""
        
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        
        if suggestions:
            corrected_text = suggestions[0].term
            cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', corrected_text).strip()
            return cleaned_text
        
        return text
    def remove_urls(self, text: str) -> str:
        return re.sub(r'http\S+|www\S+|https\S+', '', text)

    def remove_special_chars(self, text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', ' ', text)

    def remove_short_words(self, text: str, min_len=2) -> str:
        return ' '.join([word for word in text.split() if len(word) >= min_len])

    def remove_punctuation(self, tokens: list) -> list:
        return [word for word in tokens if word not in string.punctuation]


    def remove_stopwords(self, tokens: list) -> list:
        return [
            word.lower() for word in tokens if word.lower() not in self.stop_words and word.isalpha()
            ]
    def _process_single_text(self, text: str) -> str:
        global worker_tagger
        if not isinstance(text, str) or not text:
                return ""

        text = text.lower()
        text = self.remove_urls(text)
        text = self.remove_special_chars(text)
        text = self.remove_short_words(text, min_len=2)

        tokens = word_tokenize(text)
        tokens = self.remove_punctuation(tokens)
        tokens = self.remove_stopwords(tokens)

        if worker_tagger is None:
            worker_initializer()
        pos_tagged = worker_tagger.tag(tokens)

        lemmatized = [
            self.lemmatizer.lemmatize(token, self._get_wordnet_pos(pos))
            for token, pos in pos_tagged
        ]
        lemmatized = self.remove_stopwords(lemmatized)

        return ' '.join(lemmatized)

   
    def run_corpus_processing(self, dataset_name: str, batch_size: int, num_cores: int):

        db_handler = None
        try:
            db_handler = DatabaseHandler(config.MYSQL_CONFIG)
            db_handler.connect()
            db_handler.setup_tables()

            logger.info(f"🚀 Starting full text processing pipeline for dataset: '{dataset_name}' using {num_cores} core(s)")
            total_processed_count = 0
            total_remaining_count = db_handler.count_unprocessed_docs(dataset_name)
            logger.info(f"📊 Total unprocessed documents: {total_remaining_count}")

            while total_remaining_count > 0:
                docs_to_process = db_handler.get_unprocessed_docs(dataset_name, batch_size)

                if not docs_to_process:
                    logger.info("✅ No more unprocessed documents found. Processing completed.")
                    break

                logger.info(f"⚙️ Processing batch of {len(docs_to_process)} documents (remaining: {total_remaining_count})...")

                texts = [doc['raw_text'] for doc in docs_to_process]

                with Pool(num_cores, initializer=worker_initializer) as pool:
                    processed_texts = list(tqdm(
                        pool.imap(self._process_single_text, texts),
                        total=len(texts),
                        desc="🔬 Processing Batch"
                    ))

                updates = [(processed, docs_to_process[i]['id']) for i, processed in enumerate(processed_texts)]
                updated_count = db_handler.bulk_update_processed_text(updates)
                total_processed_count += updated_count

                logger.info(f"✅ Batch complete. {updated_count} documents updated. Total processed so far: {total_processed_count}")

                total_remaining_count = db_handler.count_unprocessed_docs(dataset_name)

            logger.info(f"🎉 All documents processed for dataset '{dataset_name}'. Final total: {total_processed_count}")
        
        except Exception as e:
            logger.error(f"❌ Critical error during processing for '{dataset_name}': {e}", exc_info=True)

        finally:
            if db_handler:
                db_handler.disconnect()
