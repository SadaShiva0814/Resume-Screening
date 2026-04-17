"""
AI-Based Resume Screening System
Flask Application Entry Point

This application uses NLP and ML (Sentence-BERT) to semantically
understand and rank resumes against job descriptions, replacing
traditional keyword-based ATS filtering.
"""

import os
import logging
from flask import Flask
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_app():
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # Register blueprints
    from routes.screening import screening_bp
    from routes.results import results_bp
    from routes.feedback import feedback_bp
    from routes.analytics import analytics_bp
    from routes.multirole import multirole_bp
    from routes.ats_search import ats_bp
    
    app.register_blueprint(screening_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(multirole_bp)
    app.register_blueprint(ats_bp)
    
    # Template context processors
    @app.context_processor
    def inject_db_status():
        from database.db import is_lite_mode
        return dict(is_lite_mode=is_lite_mode())
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        from flask import render_template, request, jsonify
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'API endpoint not found'}), 404
        return render_template('error.html', message='Page not found'), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        from flask import render_template, request, jsonify
        error_msg = str(getattr(e, 'original_exception', e))
        logger.error(f"Internal Server Error: {error_msg}")
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': f'Internal server error: {error_msg}'}), 500
        return render_template('error.html', message=f'Internal server error: {error_msg}'), 500
    
    logger.info("="*60)
    logger.info("AI Resume Screening System — Started")
    logger.info(f"Embedding Model: {Config.EMBEDDING_MODEL}")
    logger.info(f"MongoDB: {Config.MONGO_DB_NAME}")
    logger.info("="*60)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5001)
