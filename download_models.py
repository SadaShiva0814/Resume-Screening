import spacy
from sentence_transformers import SentenceTransformer
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download():
    logger.info(f"Downloading spaCy model: en_core_web_sm")
    try:
        spacy.cli.download("en_core_web_sm")
    except Exception as e:
        logger.error(f"Failed to download spaCy model: {e}")

    logger.info(f"Downloading SentenceTransformer model: {Config.EMBEDDING_MODEL}")
    try:
        SentenceTransformer(Config.EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Failed to download SentenceTransformer: {e}")

if __name__ == "__main__":
    download()
