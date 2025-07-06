# # handlers/text_processing_handler.py
# import nltk 
# from nltk import PorterStemmer, WordNetLemmatizer 
# from nltk.corpus import stopwords 
# import re 
# from typing import Set 

# # Ensure NLTK packages are available 
# try: 
#     stopwords.words('english') 
# except LookupError: 
#     # It's better to run the download script separately, 
#     # but this is a fallback.
#     nltk.download('punkt', quiet=True) 
#     nltk.download('stopwords', quiet=True) 
#     nltk.download('wordnet', quiet=True)

# class TextProcessor: 
#     """Handles various text processing tasks like cleaning, normalization, and tokenization.""" 
#     def __init__(self): 
#         self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
#         self.stop_words: Set[str] = set(stopwords.words('english'))
#         self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

#     def basic_clean(self, text: str) -> str: 
#         """Removes non-alphanumeric characters and extra whitespace.""" 
#         text = re.sub(r'\W', ' ', str(text))
#         text = re.sub(r'\s+', ' ', text)
#         return text.strip() 

#     def normalization(self, text: str) -> str: 
#         """Converts text to lowercase.""" 
#         return text.lower()

#     def remove_stopwords(self, text: str) -> str: 
#         """Removes common English stopwords from the text.""" 
#         words = self.tokenizer.tokenize(text) 
#         return ' '.join([word for word in words if word not in self.stop_words])

#     def lemmatization(self, text: str) -> str: 
#         """Reduces words to their base or dictionary form (lemma).""" 
#         words = self.tokenizer.tokenize(text) 
#         return ' '.join([self.lemmatizer.lemmatize(word) for word in words])

#     def remove_urls(self, text: str) -> str: 
#         """Removes URLs from the text.""" 
#         return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
     
#     def remove_punctuation(self, text: str) -> str: 
#         """Removes punctuation from the text.""" 
#         return re.sub(r'[^\w\s]', '', text)


# def process_text_pipeline(text: str, processor: TextProcessor) -> str: 
#     """ 
#     Applies a full sequence of text processing steps. 
#     """ 
#     if not text or not isinstance(text, str): 
#         return ""

#     processed_text = processor.remove_urls(text)
#     processed_text = processor.basic_clean(processed_text)
#     processed_text = processor.normalization(processed_text)
#     processed_text = processor.remove_stopwords(processed_text)
#     processed_text = processor.lemmatization(processed_text)
#     processed_text = processor.remove_punctuation(processed_text)

#     return processed_text
#////////////////////////////////////////////////////////////////
# handlers/text_processing_handler.py
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')
# import re
# import logging  # (جديد) إضافة مكتبة التسجيل الأساسية
# from typing import Set, List
# import nltk
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import TreebankWordTokenizer
# from spellchecker import SpellChecker
# from nltk.tag.perceptron import PerceptronTagger
# # --- شرح التعديلات ---
# # 1.  تم فصل تحميل بيانات NLTK. يجب تشغيل سكربت check_dependencies.py مرة واحدة.
# # 2.  تمت إضافة مكتبة 'pyspellchecker' لتصحيح الأخطاء الإملائية.
# # 3.  تم تحسين الـ Lemmatization باستخدام Part-of-Speech (POS) tagging.
# # 4.  تمت إضافة خطوات جديدة مثل تصحيح الإملاء وإزالة الكلمات القصيرة.
# # 5.  تم إعادة هيكلة الـ pipeline ليكون أكثر كفاءة ومنطقية.
# # 6.  (جديد) تمت إضافة رسالة تسجيل لتأكيد معالجة كل نص على حدة.

# def get_wordnet_pos(treebank_tag: str) -> str:
#     """
#     Maps Treebank POS tags to WordNet POS tags.
#     This is crucial for accurate lemmatization.
#     """
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         # Default to noun if the tag is not recognized
#         return wordnet.NOUN

# class TextProcessor:
#     """
#     Handles advanced text processing tasks like cleaning, normalization,
#     spell correction, and accurate lemmatization.
#     """
#     def __init__(self):
#         """
#         Initializes all necessary components. This is done only once when the service starts.
#         """
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words: Set[str] = set(stopwords.words('english'))
#         self.tokenizer = TreebankWordTokenizer()
#         # Initialize the spell checker for English
#         self.spell_checker = SpellChecker(language='en')
#         self.tagger = PerceptronTagger()
#     def remove_urls(self, text: str) -> str:
#         """Removes URLs from the text."""
#         return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

#     def remove_punctuation_and_non_alpha(self, text: str) -> str:
#         """Removes punctuation and non-alphabetic characters."""
#         # Keep only letters and spaces
#         text = re.sub(r'[^a-zA-Z\s]', ' ', text)
#         # Remove extra whitespace
#         text = re.sub(r'\s+', ' ', text)
#         return text.strip()

#     def lemmatize_with_pos(self, tokens: List[str]) -> List[str]:
#         """
#         Performs lemmatization on a list of tokens using POS tagging for higher accuracy.
#         """
#         # Get POS tags for each token
#         pos_tags = self.tagger.tag(tokens)
#         # Lemmatize each word with its corresponding POS tag
#         lemmatized_tokens = [
#             self.lemmatizer.lemmatize(word, get_wordnet_pos(pos))
#             for word, pos in pos_tags
#         ]
#         return lemmatized_tokens

# def process_text_pipeline(text: str, processor: TextProcessor) -> str:
#     """
#     Applies a full, optimized sequence of text processing steps.
#     This function is designed to be efficient and accurate.
#     """
#     if not text or not isinstance(text, str):
#         return ""

#     # --- Pipeline Stage 1: Initial Cleaning ---
#     # Remove URLs and convert to lowercase
#     processed_text = processor.remove_urls(text)
#     processed_text = processed_text.lower()
    
#     # Remove punctuation and any non-alphabetic characters
#     processed_text = processor.remove_punctuation_and_non_alpha(processed_text)

#     # --- Pipeline Stage 2: Tokenization and Word-level Processing ---
#     # Tokenize the cleaned text into words
#     words = processor.tokenizer.tokenize(processed_text)

#     # Correct spelling for the list of words.
#     # The library can handle a list directly, which is efficient.
#     misspelled = processor.spell_checker.unknown(words)
#     corrected_words = [
#         processor.spell_checker.correction(word) if word in misspelled and processor.spell_checker.correction(word) is not None else word
#         for word in words
#     ]

#     # --- Pipeline Stage 3: Filtering ---
#     # Remove stopwords and short words (less than 3 chars) in a single pass
#     filtered_words = [
#         word for word in corrected_words
#         if word not in processor.stop_words and len(word) >= 3
#     ]

#     if not filtered_words:
#         return ""

#     # --- Pipeline Stage 4: Lemmatization (The most accurate step) ---
#     # Perform lemmatization using POS tagging on the filtered words
#     lemmatized_words = processor.lemmatize_with_pos(filtered_words)

#     # --- Final Stage: Rejoin to a single string ---
#     final_text = ' '.join(lemmatized_words)
    
#     # (جديد) إضافة رسالة سجل لتأكيد اكتمال معالجة نص واحد
#     # هذه الرسالة ستظهر في طرفية الخادم (uvicorn) إذا تم إعداد التسجيل بشكل صحيح
#     logging.info("Successfully processed one text entry.")
    
#     return final_text


#/////////////////////////////////////////////

# import re
# import logging
# from typing import Set, List
# from functools import lru_cache

# import nltk
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import TreebankWordTokenizer
# from nltk.tag.perceptron import PerceptronTagger
# from spellchecker import SpellChecker


# def get_wordnet_pos(treebank_tag: str) -> str:
#     """
#     Maps Treebank POS tags to WordNet POS tags.
#     This is crucial for accurate lemmatization.
#     """
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN  # Default fallback


# class TextProcessor:
#     """
#     Handles advanced text processing tasks like cleaning, normalization,
#     spell correction, and accurate lemmatization.
#     """

#     def __init__(self):
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words: Set[str] = set(stopwords.words('english'))
#         self.tokenizer = TreebankWordTokenizer()
#         self.spell_checker = SpellChecker(language='en')
#         self.tagger = PerceptronTagger()

#     def remove_urls(self, text: str) -> str:
#         return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

#     def remove_punctuation_and_non_alpha(self, text: str) -> str:
#         text = re.sub(r'[^a-zA-Z\s]', ' ', text)
#         return re.sub(r'\s+', ' ', text).strip()

#     @lru_cache(maxsize=10000)
#     def correct_word(self, word: str) -> str:
#         corrected = self.spell_checker.correction(word)
#         return corrected if corrected else word

#     def lemmatize_with_pos(self, tokens: List[str]) -> List[str]:
#         pos_tags = self.tagger.tag(tokens)
#         return [
#             self.lemmatizer.lemmatize(word, get_wordnet_pos(pos))
#             for word, pos in pos_tags
#         ]


# def process_text_pipeline(text: str, processor: TextProcessor) -> str:
#     """
#     Applies a full, optimized sequence of text processing steps.
#     """
#     if not text or not isinstance(text, str):
#         return ""

#     # Stage 1: Clean text
#     processed_text = processor.remove_urls(text)
#     processed_text = processed_text.lower()
#     processed_text = processor.remove_punctuation_and_non_alpha(processed_text)

#     # Stage 2: Tokenize
#     words = processor.tokenizer.tokenize(processed_text)

#     # Stage 3: Spell correction (with cache)
#     misspelled = processor.spell_checker.unknown(words)
#     corrected_words = [
#         processor.correct_word(word) if word in misspelled else word
#         for word in words
#     ]

#     # Stage 4: Filter stopwords and short tokens
#     filtered_words = [
#         word for word in corrected_words
#         if word not in processor.stop_words and len(word) >= 3
#     ]
#     if not filtered_words:
#         return ""

#     # Stage 5: Lemmatization with POS
#     lemmatized_words = processor.lemmatize_with_pos(filtered_words)

#     # Stage 6: Final join
#     final_text = ' '.join(lemmatized_words)

#     logging.info("Successfully processed one text entry.")
#     return final_text

# ////////////////////
# # 
# handlers/text_processing_handler.py
import re
import logging
from typing import Set, List
from functools import lru_cache

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag.perceptron import PerceptronTagger
from spellchecker import SpellChecker

# --- شرح التعديلات ---
# 1.  تم الاحتفاظ بتحسينات الأداء: استخدام @lru_cache وتهيئة PerceptronTagger مسبقاً.
# 2.  تمت إعادة آلية التحميل التلقائي لموارد NLTK لضمان استقرار الخدمة.
# 3.  تم تنظيم الكود ليعكس أفضل الممارسات من كلا الإصدارين.

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Maps Treebank POS tags to WordNet POS tags for accurate lemmatization.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default fallback

class TextProcessor:
    """
    Handles advanced text processing tasks with performance optimizations and robustness.
    """
    def __init__(self):
        """
        Initializes components and ensures all necessary NLTK resources are available.
        """
        # (مهم) إعادة آلية التحقق والتحميل التلقائي لضمان عدم حدوث خطأ LookupError
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            logging.warning("NLTK's 'averaged_perceptron_tagger' not found. Downloading...")
            nltk.download('averaged_perceptron_tagger', quiet=True)
            logging.info("Successfully downloaded 'averaged_perceptron_tagger'.")

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words: Set[str] = set(stopwords.words('english'))
        self.tokenizer = TreebankWordTokenizer()
        self.spell_checker = SpellChecker(language='en')
        # تهيئة الـ Tagger مرة واحدة لتحسين الأداء
        self.tagger = PerceptronTagger()

    def remove_urls(self, text: str) -> str:
        """Removes URLs from the text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_punctuation_and_non_alpha(self, text: str) -> str:
        """Removes punctuation and non-alphabetic characters."""
        # text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    # استخدام التخزين المؤقت (cache) لتسريع تصحيح الكلمات المكررة
    @lru_cache(maxsize=10000)
    def correct_word(self, word: str) -> str:
        """Corrects a single word using a cache to avoid redundant lookups."""
        corrected = self.spell_checker.correction(word)
        return corrected if corrected else word

    def lemmatize_with_pos(self, tokens: List[str]) -> List[str]:
        """Performs lemmatization using the pre-loaded POS tagger."""
        pos_tags = self.tagger.tag(tokens)
        return [
            self.lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]

def process_text_pipeline(text: str, processor: TextProcessor) -> str:
    """
    Applies a full, optimized sequence of text processing steps.
    """
    if not text or not isinstance(text, str):
        return ""

    # Stage 1: Clean text
    processed_text = processor.remove_urls(text)
    processed_text = processed_text.lower()
    processed_text = processor.remove_punctuation_and_non_alpha(processed_text)

    # Stage 2: Tokenize
    words = processor.tokenizer.tokenize(processed_text)

    # Stage 3: Spell correction (with cache)
    misspelled = processor.spell_checker.unknown(words)
    corrected_words = [
        processor.correct_word(word) if word in misspelled else word
        for word in words
    ]

    # Stage 4: Filter stopwords and short tokens
    filtered_words = [
        word for word in corrected_words
        if word not in processor.stop_words and len(word) >= 3
    ]
    if not filtered_words:
        return ""

    # Stage 5: Lemmatization with POS
    lemmatized_words = processor.lemmatize_with_pos(filtered_words)

    # Stage 6: Final join
    final_text = ' '.join(lemmatized_words)

    # هذه الرسالة هي مؤشر التقدم في بيئة الخادم
    logging.info("Successfully processed one text entry.")
    return final_text
# هي يلي فوق شغالة تمام بس بتروح الارقام
# ///////////////
# # handlers/text_processing_handler.py
# import re
# from typing import Set, List
# import nltk
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import TreebankWordTokenizer
# from nltk.tag import PerceptronTagger
# from spellchecker import SpellChecker

# def _ensure_nltk_data():
#     """
#     Ensures all necessary NLTK data packages are downloaded.
#     """
#     required_packages = {
#         'tokenizers/punkt': 'punkt',
#         'corpora/stopwords': 'stopwords',
#         'corpora/wordnet': 'wordnet',
#         'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
#     }
#     for path, pkg_id in required_packages.items():
#         try:
#             nltk.data.find(path)
#         except LookupError:
#             print(f"NLTK package '{pkg_id}' not found. Downloading...")
#             nltk.download(pkg_id, quiet=True)

# # Ensure data is available when the module is imported
# _ensure_nltk_data()

# def get_wordnet_pos(treebank_tag: str) -> str:
#     """
#     Maps Treebank POS tags to the format expected by the WordNetLemmatizer.
#     """
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN

# class TextProcessor:
#     """
#     Handles state-of-the-art text processing tasks including cleaning,
#     spell correction, and POS-tagged lemmatization.
#     """
#     def __init__(self):
#         """Initializes all necessary components for processing."""
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words: Set[str] = set(stopwords.words('english'))
#         self.tokenizer = TreebankWordTokenizer()
#         self.tagger = PerceptronTagger()
#         self.spell_checker = SpellChecker()

#     def process(self, text: str) -> str:
#         """
#         Applies the full, high-quality processing pipeline to a single text.
#         This is the main method to be called for processing.
#         """
#         if not text or not isinstance(text, str):
#             return ""

#         # 1. Clean URLs and convert to lowercase
#         text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#         text = text.lower()

#         # 2. Refined cleaning: Keep letters and numbers, remove other symbols
#         text = re.sub(r'[^a-z0-9\s]', ' ', text)
#         text = re.sub(r'\s+', ' ', text).strip()

#         # 3. Tokenize the text
#         tokens = self.tokenizer.tokenize(text)

#         # 4. Spell Correction
#         misspelled = self.spell_checker.unknown(tokens)
#         corrected_tokens = [
#             self.spell_checker.correction(word) if word in misspelled and self.spell_checker.correction(word) is not None else word
#             for word in tokens
#         ]

#         # 5. Filter out stopwords and short tokens
#         filtered_tokens = [
#             word for word in corrected_tokens
#             if word not in self.stop_words and len(word) > 2
#         ]

#         if not filtered_tokens:
#             return ""

#         # 6. Part-of-Speech (POS) Tagging
#         pos_tags = self.tagger.tag(filtered_tokens)

#         # 7. Accurate Lemmatization using POS tags
#         lemmatized_tokens = [
#             self.lemmatizer.lemmatize(word, get_wordnet_pos(pos))
#             for word, pos in pos_tags
#         ]

#         return ' '.join(lemmatized_tokens)

# # You can keep this function if you want a separate entry point,
# # but it's often cleaner to call the class method directly.
# def process_text_pipeline(text: str, processor: TextProcessor) -> str:
#     """
#     A wrapper function that applies the full sequence of text processing steps
#     using an instance of the AdvancedTextProcessor.
#     """
#     return processor.process(text)

