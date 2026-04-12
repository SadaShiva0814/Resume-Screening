"""
Feedback Routes — Recruiter Feedback Collection & Learning

Handles recruiter actions:
- Shortlisting a candidate
- Rejecting a candidate
- Re-ranking (moving a candidate up/down)

After feedback, triggers the learning engine to update model preferences.
"""

import logging
from flask import Blueprint, request, jsonify
from database.db import save_feedback, get_session_feedback, get_session
from pipeline.feedback_learner import process_feedback, get_learning_stats

feedback_bp = Blueprint('feedback', __name__)
logger = logging.getLogger(__name__)


@feedback_bp.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit recruiter feedback on a candidate.
    
    JSON body:
    {
        "session_id": "...",
        "candidate_id": "...",
        "action": "shortlisted" | "rejected" | "re-ranked",
        "new_rank": 3  (optional, for re-ranking)
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400
    
    session_id = data.get('session_id')
    candidate_id = data.get('candidate_id')
    action = data.get('action')
    new_rank = data.get('new_rank')
    
    if not all([session_id, candidate_id, action]):
        return jsonify({'error': 'session_id, candidate_id, and action are required'}), 400
    
    if action not in ('shortlisted', 'rejected', 're-ranked', 'undo'):
        return jsonify({'error': 'action must be: shortlisted, rejected, re-ranked, or undo'}), 400
    
    # Save feedback
    feedback_id = save_feedback(
        session_id=session_id,
        candidate_id=candidate_id,
        action=action,
        new_rank=new_rank
    )
    
    if not feedback_id:
        return jsonify({'error': 'Candidate not found'}), 404
    
    # Trigger learning
    session = get_session(session_id)
    job_category = session.get('job_category', 'general') if session else 'general'
    
    learning_result = None
    try:
        learning_result = process_feedback(session_id, job_category)
    except Exception as e:
        logger.warning(f"Learning update failed: {e}")
    
    return jsonify({
        'success': True,
        'feedback_id': feedback_id,
        'learning_update': learning_result,
        'message': f'Candidate {action} successfully. Model preferences updated.'
    })


@feedback_bp.route('/api/feedback/<session_id>')
def get_feedback(session_id):
    """Get all feedback for a session."""
    feedbacks = get_session_feedback(session_id)
    return jsonify({
        'session_id': session_id,
        'feedback': feedbacks,
        'total': len(feedbacks)
    })


@feedback_bp.route('/api/learning-stats/<job_category>')
def learning_stats(job_category):
    """Get learning stats for a job category."""
    stats = get_learning_stats(job_category)
    return jsonify(stats)


@feedback_bp.route('/api/learning-stats')
def all_learning_stats():
    """Get learning stats for all categories."""
    from database.db import get_db
    db = get_db()
    
    categories = db.learned_preferences.distinct('job_category')
    all_stats = {}
    for cat in categories:
        all_stats[cat] = get_learning_stats(cat)
    
    return jsonify({
        'categories': all_stats,
        'total_categories': len(categories)
    })
