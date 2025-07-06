# utils/parsers.py
import pandas as pd

def parse_qa_line(block: str, questions_map: dict) -> tuple:
    """
    يحلل سطراً من مجموعة بيانات الأسئلة والأجوبة (antique_qa).
    """
    try:
        parts = block.strip().split('\t', 1)
        if len(parts) != 2: return None, None
        doc_id, raw_text = parts[0], parts[1].strip().replace('%', '%%')

        topic_id = doc_id.split('_')[0]
        question_text = questions_map.get(topic_id, "").replace('%', '%%')

        doc_tuple = (doc_id, raw_text, "")
        meta_tuple = (topic_id, question_text)
        return doc_tuple, meta_tuple
    except Exception:
        return None, None

def parse_generic_line(block: str, **kwargs) -> tuple:
    """
    محلل عام للبيانات التي تحتوي على معرف ونص فقط (مثل clinical_trials).
    """
    try:
        parts = block.strip().split('\t', 1)
        if len(parts) != 2: return None, None
        doc_id, raw_text = parts[0], parts[1].strip().replace('%', '%%')
        
        doc_tuple = (doc_id, raw_text, "")
        meta_tuple = () # لا توجد بيانات وصفية
        return doc_tuple, meta_tuple
    except Exception:
        return None, None

def parse_wikir_csv_line(row: pd.Series, **kwargs) -> tuple:
    """
    يحلل سطراً من ملف CSV الخاص بـ wikIR1k.
    يفترض أن الأعمدة هي 'id_right' و 'text_right'.
    """
    try:
        # تأكد من أن أسماء الأعمدة تتطابق مع ملفك
        doc_id = str(row['id_right'])
        raw_text = str(row['text_right']).replace('%', '%%')

        doc_tuple = (doc_id, raw_text, "")
        meta_tuple = ()
        return doc_tuple, meta_tuple
    except (KeyError, AttributeError):
        return None, None
