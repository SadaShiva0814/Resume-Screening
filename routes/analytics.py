"""
Analytics Routes — Evaluation Metrics & Dashboard
"""

import logging
from flask import Blueprint, jsonify, render_template
from database.db import get_analytics, get_session, get_candidates

analytics_bp = Blueprint('analytics', __name__)
logger = logging.getLogger(__name__)


@analytics_bp.route('/analytics/<session_id>')
def analytics_page(session_id):
    """Render analytics dashboard for a session."""
    session = get_session(session_id)
    if not session:
        return render_template('error.html', message='Session not found'), 404
        
    # Retroactive lookup for older multi-sessions that lack the linked ID
    if not session.get('multi_session_id'):
        from database.db import get_db
        db = get_db()
        parent = db.multi_sessions.find_one({'roles.session_id': session_id})
        if parent:
            session['multi_session_id'] = str(parent['_id'])
    
    analytics = get_analytics(session_id)
    candidates = get_candidates(session_id)
    
    candidate_scores = {}
    for c in sorted(candidates, key=lambda x: x.get('overall_score', 0), reverse=True)[:10]:
        name = c.get('candidate_name', 'Unnamed')
        score_pct = float(c.get('overall_score', 0)) * 100
        # If there are duplicates, appending the score ensures they don't overwrite
        # We'll use name as key. Actually, we can just use the name directly.
        candidate_scores[name] = round(score_pct, 1)
        
    return render_template('analytics.html', session=session, analytics=analytics, candidate_scores=candidate_scores)


@analytics_bp.route('/api/analytics/<session_id>')
def api_analytics(session_id):
    """API: Get evaluation metrics for a session."""
    analytics = get_analytics(session_id)
    if not analytics:
        return jsonify({'error': 'Analytics not found for this session'}), 404
    
    return jsonify({'session_id': session_id, 'metrics': analytics})
