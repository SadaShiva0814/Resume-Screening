"""
Results Routes — View Rankings & Candidate Details
"""

import logging
from flask import Blueprint, jsonify, render_template
from database.db import get_candidates, get_candidate, get_session, get_all_sessions

results_bp = Blueprint('results', __name__)
logger = logging.getLogger(__name__)


@results_bp.route('/results/<session_id>')
def results_page(session_id):
    """Render the results page for a screening session."""
    session = get_session(session_id)
    if not session:
        return render_template('error.html', message='Session not found'), 404
    
    candidates = get_candidates(session_id)
    return render_template('results.html', session=session, candidates=candidates)


@results_bp.route('/api/results/<session_id>')
def api_results(session_id):
    """API: Get ranked results for a session."""
    session = get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    candidates = get_candidates(session_id)
    
    return jsonify({
        'session': session,
        'candidates': candidates,
        'total': len(candidates)
    })


@results_bp.route('/api/candidate/<candidate_id>')
def api_candidate_detail(candidate_id):
    """API: Get detailed info for a single candidate."""
    candidate = get_candidate(candidate_id)
    if not candidate:
        return jsonify({'error': 'Candidate not found'}), 404
    
    # Don't send raw embedding
    candidate.pop('embedding', None)
    
    return jsonify({'candidate': candidate})


@results_bp.route('/candidate/<candidate_id>')
def candidate_page(candidate_id):
    """Render detailed candidate page."""
    candidate = get_candidate(candidate_id)
    if not candidate:
        return render_template('error.html', message='Candidate not found'), 404
    
    # Get session info
    session = get_session(candidate.get('session_id', ''))
    
    # Remove embedding for frontend
    candidate.pop('embedding', None)
    
    return render_template('candidate.html', candidate=candidate, session=session)


@results_bp.route('/history')
def history_page():
    """Render screening history page."""
    sessions = get_all_sessions()
    return render_template('history.html', sessions=sessions)


@results_bp.route('/api/history')
def api_history():
    """API: Get all screening sessions."""
    sessions = get_all_sessions()
    return jsonify({'sessions': sessions, 'total': len(sessions)})


@results_bp.route('/api/session/<session_id>', methods=['DELETE'])
def api_delete_session(session_id):
    """API: Delete a screening session and all its data."""
    from database.db import delete_session
    try:
        success = delete_session(session_id)
        if success:
            return jsonify({'success': True, 'message': 'Session deleted successfully'})
        else:
            return jsonify({'error': 'Session not found or already deleted'}), 404
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500


@results_bp.route('/api/sessions/bulk', methods=['DELETE'])
def api_delete_bulk_sessions():
    """API: Bulk delete multiple screening sessions."""
    from flask import request
    from database.db import delete_session
    data = request.get_json()
    if not data or 'session_ids' not in data:
        return jsonify({'error': 'session_ids list required'}), 400
    
    session_ids = data['session_ids']
    if not isinstance(session_ids, list):
        return jsonify({'error': 'session_ids must be a list'}), 400
        
    deleted_count = 0
    try:
        for sid in session_ids:
            if delete_session(sid):
                deleted_count += 1
        return jsonify({'success': True, 'deleted_count': deleted_count})
    except Exception as e:
        logger.error(f"Error in bulk deletion: {e}")
        return jsonify({'error': str(e)}), 500
