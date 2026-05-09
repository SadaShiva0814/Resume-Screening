import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'resume-screening-secret-key-2026')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database (Persistent JSON Document Store — no external server needed)
    DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    
    # Hugging Face Detection
    IS_HF = os.getenv('SPACE_ID') is not None
    if IS_HF:
        print(">>> Detected Hugging Face Space Environment")
    
    # Upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}
    
    # ML Model
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = 384
    
    # Section Weights (Recalibrated for Sum-to-100% logic)
    DEFAULT_WEIGHTS = {
        'skills': 50,
        'experience': 35,
        'projects': 15
    }
    
    # Feedback Learning
    LEARNING_RATE = 0.05          # How fast weights adapt
    MIN_FEEDBACK_FOR_BIAS = 5    # Min feedback signals before using preference bias
    INITIAL_ALPHA = 1.0           # Start with pure JD matching (no preference bias)
    ALPHA_DECAY = 0.05            # How fast to blend in preference bias
    
    # Kaggle Dataset
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    KAGGLE_CSV = os.path.join(DATA_DIR, 'Resume.csv')
    
    # Tesseract
    # In Docker (Linux), tesseract is usually just in the PATH.
    # We check common paths for local Mac dev (Homebrew) and Linux.
    TESSERACT_CMD = os.getenv('TESSERACT_CMD')
    if not TESSERACT_CMD:
        if os.path.exists('/usr/bin/tesseract'):
            TESSERACT_CMD = '/usr/bin/tesseract'
        elif os.path.exists('/opt/homebrew/bin/tesseract'):
            TESSERACT_CMD = '/opt/homebrew/bin/tesseract'
        else:
            TESSERACT_CMD = 'tesseract' # Assume it's in PATH
