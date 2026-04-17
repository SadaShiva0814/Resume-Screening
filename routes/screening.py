"""
Screening Routes — Resume Upload & Screening API

Handles:
- Resume file upload (multiple files)
- Job description input
- Triggering the screening pipeline
- Session management
"""

import os
import uuid
import logging
from flask import Blueprint, request, jsonify, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from config import Config

screening_bp = Blueprint('screening', __name__)
logger = logging.getLogger(__name__)


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@screening_bp.route('/')
def index():
    """Landing page."""
    return render_template('index.html')


@screening_bp.route('/upload', methods=['GET'])
def upload_page():
    """Upload page for resumes and job description."""
    from database.db import get_all_categories
    categories = get_all_categories()
    
    # Map to Title Case for UI if they are all lowercase
    ui_categories = []
    for cat in categories:
        # Extra safety check to ensure cat is a string
        if not isinstance(cat, str):
            cat = str(cat)
            
        if cat.lower() == 'general':
            continue # Already handled or want it at top
        
        # Apply title case if it looks like a lowercase string
        if cat.islower():
            ui_categories.append(cat.title())
        else:
            ui_categories.append(cat)
    
    return render_template('upload.html', categories=ui_categories)


@screening_bp.route('/api/screen', methods=['POST'])
def screen():
    """
    Main screening endpoint — now async with background threading.
    
    Accepts multipart form data with:
    - files[]: Resume files (PDF, DOCX, TXT)
    - job_description: The job description text
    - job_category: Optional job category
    - weights: Optional JSON of section weights
    """
    import json
    import threading
    from database.db import (create_session, update_session_status)
    
    # Validate inputs
    job_description = request.form.get('job_description', '').strip()
    if not job_description:
        return jsonify({'error': 'Job description is required'}), 400
    
    job_category = request.form.get('job_category', 'general').strip().lower()
    
    # Parse optional weights
    weights = None
    weights_json = request.form.get('weights', '')
    if weights_json:
        try:
            weights = json.loads(weights_json)
        except json.JSONDecodeError:
            pass
    
    # Collect resume files
    files = request.files.getlist('files[]')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'At least one resume file is required'}), 400
    
    # Create session
    session_id = create_session(
        job_description=job_description,
        job_category=job_category,
        num_resumes=len(files),
        weights=weights or Config.DEFAULT_WEIGHTS
    )
    
    # Save uploaded files and prepare inputs
    resume_inputs = []
    session_upload_dir = os.path.join(Config.UPLOAD_FOLDER, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)
    
    for f in files:
        if f and f.filename and _allowed_file(f.filename):
            filename = secure_filename(f.filename)
            unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
            file_path = os.path.join(session_upload_dir, unique_name)
            f.save(file_path)
            
            resume_inputs.append({
                'file_path': file_path,
                'file_name': filename
            })
    
    if not resume_inputs:
        update_session_status(session_id, 'failed')
        return jsonify({'error': 'No valid resume files uploaded'}), 400
    
    # Launch background thread for the heavy AI pipeline
    def _run_pipeline(app, sid, inputs, jd, w, cat):
        with app.app_context():
            from pipeline.ranker import screen_resumes
            from pipeline.evaluator import compute_all_metrics
            from database.db import (save_candidates, update_session_status,
                                      save_analytics, update_session_progress)
            try:
                def on_progress(done, total):
                    update_session_progress(sid, done)
                
                candidates = screen_resumes(
                    resume_inputs=inputs,
                    job_description=jd,
                    weights=w,
                    job_category=cat,
                    progress_callback=on_progress
                )
                
                if not candidates:
                    update_session_status(sid, 'failed')
                    return
                
                save_candidates(sid, candidates)
                update_session_status(sid, 'completed', num_resumes=len(candidates))
                
                try:
                    resume_texts = [c.get('raw_text', '') for c in candidates]
                    metrics = compute_all_metrics(candidates, jd, resume_texts)
                    save_analytics(sid, metrics)
                except Exception as e:
                    logger.warning(f"Could not compute metrics: {e}")
                
                logger.info(f"Session {sid} complete: {len(candidates)} candidates ranked")
                
            except Exception as e:
                logger.error(f"Screening failed for session {sid}: {e}", exc_info=True)
                update_session_status(sid, 'failed')
    
    from flask import current_app
    app = current_app._get_current_object()
    t = threading.Thread(target=_run_pipeline, args=(app, session_id, resume_inputs, job_description, weights, job_category))
    t.daemon = True
    t.start()
    
    logger.info(f"Session {session_id} launched in background with {len(resume_inputs)} resumes")
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'redirect': f'/status/{session_id}'
    })


@screening_bp.route('/status/<session_id>')
def status_page(session_id):
    """Render the live progress page for a session."""
    from database.db import get_session
    session = get_session(session_id)
    if not session:
        return render_template('error.html', message='Session not found'), 404
    # If already completed, skip status page entirely
    if session.get('status') == 'completed':
        # Check for custom redirect (multi-role tracker sessions)
        if session.get('redirect_url'):
            return redirect(session['redirect_url'])
        # Check if this is a multi-role child
        if session.get('multi_session_id'):
            return redirect(f"/multi/results/{session['multi_session_id']}")
        return redirect(f"/results/{session_id}")
    return render_template('status.html', session=session)


@screening_bp.route('/api/status/<session_id>')
def api_status(session_id):
    """API: Get live progress for a screening session."""
    from database.db import get_session
    session = get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    status = session.get('status', 'processing')
    processed = session.get('processed_count', 0)
    total = session.get('num_resumes', 0)
    
    result = {
        'status': status,
        'processed_count': processed,
        'total_resumes': total,
        'progress_pct': round((processed / total) * 100) if total > 0 else 0
    }
    
    if status == 'completed':
        result['redirect'] = session.get('redirect_url', f'/results/{session_id}')
    elif status == 'failed':
        result['error'] = 'Screening pipeline failed. Check server logs.'
    
    return jsonify(result)


@screening_bp.route('/api/screen/dataset', methods=['POST'])
def screen_dataset():
    """
    Screen resumes from the Kaggle dataset (CSV text input).
    
    Used for testing and demonstration.
    """
    import json
    import pandas as pd
    from pipeline.ranker import screen_resumes
    from pipeline.evaluator import compute_all_metrics
    from database.db import (create_session, save_candidates,
                              update_session_status, save_analytics)
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400
    
    job_description = data.get('job_description', '').strip()
    if not job_description:
        return jsonify({'error': 'Job description is required'}), 400
    
    job_category = data.get('job_category', 'general')
    max_resumes = data.get('max_resumes', 50)
    category_filter = data.get('category_filter', None)
    
    # Load Kaggle dataset
    csv_path = Config.KAGGLE_CSV
    if not os.path.exists(csv_path):
        return jsonify({'error': f'Dataset not found at {csv_path}. Please download the Kaggle dataset.'}), 404
    
    df = pd.read_csv(csv_path)
    
    # Filter by category if specified
    if category_filter:
        df = df[df['Category'].str.lower() == category_filter.lower()]
    
    # Limit number of resumes
    if len(df) > max_resumes:
        df = df.sample(n=max_resumes, random_state=42)
    
    # Create session
    session_id = create_session(
        job_description=job_description,
        job_category=job_category,
        num_resumes=len(df)
    )
    
    try:
        # Prepare inputs from CSV — use HTML column for better structure
        resume_inputs = []
        for idx, row in df.iterrows():
            resume_text = str(row.get('Resume_str', row.get('Resume', '')))
            resume_html = str(row.get('Resume_html', ''))
            category = row.get('Category', 'Unknown')
            resume_inputs.append({
                'text': resume_text,
                'html': resume_html if resume_html != 'nan' else '',
                'file_name': f"{category}_{idx}"
            })
        
        # Run screening
        candidates = screen_resumes(
            resume_inputs=resume_inputs,
            job_description=job_description,
            job_category=job_category
        )
        
        if not candidates:
            update_session_status(session_id, 'failed')
            return jsonify({'error': 'No candidates could be processed'}), 500
        
        # Save results
        save_candidates(session_id, candidates)
        update_session_status(session_id, 'completed', num_resumes=len(candidates))
        
        # Compute metrics
        try:
            resume_texts = [c.get('raw_text', '') for c in candidates]
            metrics = compute_all_metrics(candidates, job_description, resume_texts)
            save_analytics(session_id, metrics)
        except Exception as e:
            logger.warning(f"Metrics computation failed: {e}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'candidates_processed': len(candidates),
            'redirect': f'/results/{session_id}'
        })
        
    except Exception as e:
        logger.error(f"Dataset screening failed: {e}", exc_info=True)
        update_session_status(session_id, 'failed')
        return jsonify({'error': str(e)}), 500
