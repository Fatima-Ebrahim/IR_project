
import os
import pandas as pd
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from utils import config
from database.database_handler import DatabaseHandler
from utils.logger_config import logger

class TopicModelingHandler:
    def __init__(self):
        self.output_base_dir = os.path.join(config.OUTPUT_DIR, "topic_modeling")

    def run_lda_from_database(self, dataset_name: str, num_topics: int = 8):
        logger.info(f"🔍 بدء تحليل المواضيع LDA لمجموعة البيانات: {dataset_name}")

        db = DatabaseHandler(config.MYSQL_CONFIG)
        db.connect()
        docs = db.get_processed_docs_for_indexing(dataset_name)
        db.disconnect()

        if not docs:
            raise ValueError("❌ لا توجد مستندات معالجة لهذه المجموعة.")

        df = pd.DataFrame(docs)
        df = df[df['processed_text'].notnull() & (df['processed_text'] != '')]
        processed_docs = df['processed_text'].astype(str).apply(lambda x: x.split()).tolist()

        dictionary = Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            chunksize=100,
            passes=10,
            per_word_topics=True
        )

        logger.info("📈 تقييم النموذج...")
        coherence_model = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        logger.info(f"✅ Coherence Score: {coherence:.4f}")

        logger.info("📊 إنشاء ملف التفاعل LDA...")
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        db_name = config.MYSQL_CONFIG.get("database", "default")
        dataset_output_dir = os.path.join(self.output_base_dir, db_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        vis_path = os.path.join(dataset_output_dir, "lda_visualization.html")
        pyLDAvis.save_html(vis_data, vis_path)

        logger.info("📝 حفظ توزيع المواضيع في قاعدة البيانات...")
        topic_distributions = []
        for i, row in enumerate(corpus):
            topic_probs = lda_model.get_document_topics(row, minimum_probability=0.0)
            topic_probs_dict = {f"topic_{t}": prob for t, prob in topic_probs}
            topic_probs_dict["dominant_topic"] = max(topic_probs, key=lambda x: x[1])[0]
            topic_probs_dict["doc_id"] = int(df.iloc[i]['id'])
            topic_distributions.append(topic_probs_dict)

        db.connect()
        db.insert_document_topics(topic_distributions)
        db.disconnect()

        return {
            "message": f"Topic modeling completed successfully for dataset '{dataset_name}'.",
            "topics_table": "lda_topics",
            "distribution_table": "document_topics",
            "coherence": coherence,
            "visualization_html_path": vis_path
        }
