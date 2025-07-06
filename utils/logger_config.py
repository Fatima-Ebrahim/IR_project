# utils/logger_config.py
import logging
import sys

def setup_logger():
    """
    Sets up a centralized logger for the application.
    """
    logger = logging.getLogger("IR_Project")
    logger.setLevel(logging.INFO)

    # منع تكرار الرسائل إذا تم استدعاء هذه الدالة عدة مرات
    if logger.hasHandlers():
        logger.handlers.clear()

    # إعداد شكل الرسالة
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # إعداد معالج لطباعة الرسائل في نافذة الأوامر
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

# إنشاء نسخة واحدة من الـ logger لاستخدامها في كل المشروع
logger = setup_logger()
