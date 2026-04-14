"""
Multi-Role Routes — Multi-JD Screening & Results

Handles:
- Multi-role screening endpoint (screen against multiple JDs)
- Grouped results view
- Multi-session management
"""

import os
import uuid
import json
import logging
from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
from config import Config

multirole_bp = Blueprint('multirole', __name__)
logger = logging.getLogger(__name__)


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@multirole_bp.route('/api/screen/multi', methods=['POST'])
def screen_multi_role():
    """
    Screen resumes against multiple job descriptions simultaneously.
    
    Accepts multipart form data with:
    - files[]: Resume files (PDF, DOCX, TXT)
    - roles: JSON string — array of {title, description} objects
    - weights: Optional JSON of section weights
    """
    from pipeline.ranker import screen_resumes_multi_role
    from pipeline.evaluator import compute_all_metrics
    from database.db import (
        create_session, save_candidates, update_session_status,
        save_analytics, create_multi_session
    )
    
    # Parse roles
    roles_json = request.form.get('roles', '').strip()
    if not roles_json:
        return jsonify({'error': 'roles JSON is required'}), 400
    
    try:
        roles = json.loads(roles_json)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid roles JSON'}), 400
    
    if not isinstance(roles, list) or len(roles) < 2:
        return jsonify({'error': 'At least 2 roles are required for multi-role screening'}), 400
    
    # Validate each role
    for i, role in enumerate(roles):
        if not role.get('title', '').strip():
            return jsonify({'error': f'Role {i+1} is missing a title'}), 400
        if not role.get('description', '').strip():
            return jsonify({'error': f'Role {i+1} ("{role["title"]}") is missing a description'}), 400
    
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
    
    # Save uploaded files to a shared directory
    multi_upload_id = uuid.uuid4().hex[:12]
    upload_dir = os.path.join(Config.UPLOAD_FOLDER, f'multi_{multi_upload_id}')
    os.makedirs(upload_dir, exist_ok=True)
    
    resume_inputs = []
    for f in files:
        if f and f.filename and _allowed_file(f.filename):
            filename = secure_filename(f.filename)
            unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
            file_path = os.path.join(upload_dir, unique_name)
            f.save(file_path)
            resume_inputs.append({
                'file_path': file_path,
                'file_name': filename,
            })
    
    if not resume_inputs:
        return jsonify({'error': 'No valid resume files uploaded'}), 400
    
    try:
        # Build job_descriptions list for the pipeline
        job_descriptions = []
        for role in roles:
            job_descriptions.append({
                'title': role['title'].strip(),
                'text': role['description'].strip(),
                'category': role.get('category', role['title'].strip().lower().replace(' ', '_')),
            })
        
        logger.info(f"Starting multi-role screening: {len(resume_inputs)} resumes × {len(job_descriptions)} roles")
        
        # Run multi-role pipeline
        results = screen_resumes_multi_role(
            resume_inputs=resume_inputs,
            job_descriptions=job_descriptions,
            weights=weights,
        )
        
        # Save each role's results as a child session
        roles_data = []
        for jd in job_descriptions:
            title = jd['title']
            category = jd['category']
            role_candidates = results['roles'].get(title, [])
            
            # Create a child session for this role
            child_session_id = create_session(
                job_description=jd['text'],
                job_category=category,
                num_resumes=len(role_candidates),
                weights=weights or Config.DEFAULT_WEIGHTS,
            )
            
            if role_candidates:
                save_candidates(child_session_id, role_candidates)
                update_session_status(child_session_id, 'completed', num_resumes=len(role_candidates))
                
                # Compute analytics for this role
                try:
                    resume_texts = [c.get('raw_text', '') for c in role_candidates]
                    metrics = compute_all_metrics(role_candidates, jd['text'], resume_texts)
                    save_analytics(child_session_id, metrics)
                except Exception as e:
                    logger.warning(f"Metrics failed for role '{title}': {e}")
            else:
                update_session_status(child_session_id, 'completed', num_resumes=0)
            
            roles_data.append({
                'title': title,
                'category': category,
                'session_id': child_session_id,
                'candidate_count': len(role_candidates),
            })
        
        # Create the parent multi-session
        multi_session_id = create_multi_session(
            roles_data=roles_data,
            total_resumes=len(resume_inputs),
        )
        
        # Store versatile and unmatched info in the multi-session
        from database.db import get_db
        from bson import ObjectId
        db = get_db()
        db.multi_sessions.update_one(
            {'_id': ObjectId(multi_session_id)},
            {'$set': {
                'versatile_candidates': results['versatile'],
                'unmatched_count': len(results['unmatched']),
            }}
        )
        
        # Link child sessions back to the parent multi-session for UI navigation
        for rd in roles_data:
            db.sessions.update_one(
                {'_id': ObjectId(rd['session_id'])},
                {'$set': {'multi_session_id': str(multi_session_id)}}
            )
        
        total_routed = sum(r['candidate_count'] for r in roles_data)
        logger.info(f"Multi-role session {multi_session_id} complete: "
                    f"{total_routed} routed across {len(roles_data)} roles")
        
        return jsonify({
            'success': True,
            'multi_session_id': multi_session_id,
            'roles': roles_data,
            'versatile_count': len(results['versatile']),
            'unmatched_count': len(results['unmatched']),
            'redirect': f'/multi/results/{multi_session_id}',
        })
        
    except Exception as e:
        logger.error(f"Multi-role screening failed: {e}", exc_info=True)
        return jsonify({'error': f'Multi-role screening failed: {str(e)}'}), 500


@multirole_bp.route('/multi/results/<multi_session_id>')
def multi_results_page(multi_session_id):
    """Render grouped multi-role results."""
    from database.db import get_multi_session, get_candidates, get_session
    
    ms = get_multi_session(multi_session_id)
    if not ms:
        return render_template('error.html', message='Multi-role session not found'), 404
    
    # Load candidates for each role
    roles_with_candidates = []
    for role in ms.get('roles', []):
        child_session_id = role['session_id']
        candidates = get_candidates(child_session_id)
        session = get_session(child_session_id)
        roles_with_candidates.append({
            'title': role['title'],
            'category': role.get('category', ''),
            'session_id': child_session_id,
            'candidates': candidates,
            'candidate_count': len(candidates),
            'jd_preview': session['job_description'][:200] if session else '',
        })
    
    # Enrich versatile candidates with DB IDs and status
    versatile_enriched = ms.get('versatile_candidates', [])
    for v in versatile_enriched:
        primary = v.get('primary_role')
        for r in roles_with_candidates:
            if r['title'] == primary:
                for c in r['candidates']:
                    if c.get('file_name') == v.get('file_name'):
                        v['_id'] = c['_id']
                        v['session_id'] = r['session_id']
                        v['feedback_status'] = c.get('feedback_status')
                        break
                break
    
    return render_template(
        'multi_results.html',
        ms=ms,
        roles=roles_with_candidates,
        versatile=versatile_enriched,
    )


@multirole_bp.route('/api/multi/session/<multi_session_id>', methods=['DELETE'])
def api_delete_multi_session(multi_session_id):
    """Delete a multi-role session and all child data."""
    from database.db import delete_multi_session
    try:
        success = delete_multi_session(multi_session_id)
        if success:
            return jsonify({'success': True})
        return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting multi-session: {e}")
        return jsonify({'error': str(e)}), 500
