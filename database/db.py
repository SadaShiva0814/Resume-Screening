"""
MongoDB Database Layer
Handles all database operations for the resume screening system.
Collections:
  - sessions: Screening sessions (JD + metadata)
  - candidates: Parsed resume data + embeddings + scores
  - feedback: Recruiter feedback (shortlist/reject/re-rank)
  - learned_preferences: Model's learned weights and preference vectors per category
  - analytics: Evaluation metrics per session
"""

from pymongo import MongoClient, DESCENDING
from datetime import datetime
import numpy as np

_client = None
_db = None
_is_lite_mode = False

class MockCollection:
    """Mock MongoDB collection that stores data in memory."""
    def __init__(self, name):
        self.name = name
        self.data = {}
        
    def insert_one(self, doc):
        from bson import ObjectId
        if '_id' not in doc:
            doc['_id'] = ObjectId()
        self.data[str(doc['_id'])] = doc
        class Result:
            def __init__(self, id): self.inserted_id = id
        return Result(doc['_id'])

    def insert_many(self, docs):
        for d in docs: self.insert_one(d)

    def find_one(self, query):
        if '_id' in query and isinstance(query['_id'], (str, bytes, object)):
             return self.data.get(str(query['_id']))
        for doc in self.data.values():
            match = True
            for k, v in query.items():
                if doc.get(k) != v: match = False; break
            if match: return doc
        return None

    def find(self, query=None):
        query = query or {}
        results = []
        for doc in self.data.values():
            match = True
            for k, v in query.items():
                if doc.get(k) != v: match = False; break
            if match: results.append(doc)
        
        class Cursor:
            def __init__(self, data): self._data = data
            def sort(self, field, direction=1):
                self._data.sort(key=lambda x: x.get(field) or 0, reverse=(direction == -1))
                return self
            def limit(self, n):
                self._data = self._data[:n]
                return self
            def __iter__(self): return iter(self._data)
            def __getitem__(self, i): return self._data[i]
        
        return Cursor(results)

    def update_one(self, query, update, upsert=False):
        doc = self.find_one(query)
        if doc:
            if '$set' in update:
                doc.update(update['$set'])
        elif upsert:
            new_doc = query.copy()
            if '$set' in update: new_doc.update(update['$set'])
            self.insert_one(new_doc)

    def delete_one(self, query):
        doc = self.find_one(query)
        if doc:
            del self.data[str(doc['_id'])]
            class Result:
                def __init__(self): self.deleted_count = 1
            return Result()
        class Result:
            def __init__(self): self.deleted_count = 0
        return Result()

    def delete_many(self, query):
        to_delete = []
        for id, doc in self.data.items():
            match = True
            for k, v in query.items():
                if doc.get(k) != v: match = False; break
            if match: to_delete.append(id)
        for id in to_delete: del self.data[id]

    def distinct(self, field):
        values = set()
        for doc in self.data.values():
            val = doc.get(field)
            if val is not None:
                values.add(val)
        return list(values)

    def count_documents(self, query=None):
        if not query:
            return len(self.data)
        return len([d for d in self.data.values()
                    if all(d.get(k) == v for k, v in query.items())])

    def create_index(self, *args, **kwargs): pass

class MockDB:
    """Mock MongoDB database."""
    def __init__(self):
        self._collections = {}
        import os, json
        seed_path = os.path.join(os.path.dirname(__file__), 'seed_data.json')
        if os.path.exists(seed_path):
            try:
                with open(seed_path, 'r') as f:
                    seed_data = json.load(f)
                from datetime import datetime
                for coll_name, docs in seed_data.items():
                    coll = MockCollection(coll_name)
                    for doc in docs:
                        for field in ['created_at', 'completed_at', 'timestamp', 'cached_at', 'last_updated']:
                            if field in doc and isinstance(doc[field], str):
                                try:
                                    doc[field] = datetime.fromisoformat(doc[field])
                                except Exception:
                                    pass
                        coll.data[str(doc['_id'])] = doc
                    self._collections[coll_name] = coll
                print(f">>> MockDB: Loaded {sum(len(c.data) for c in self._collections.values())} documents from seed data.")
            except Exception as e:
                print(f">>> MockDB: Failed to load seed data: {e}")

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]
    def __getattr__(self, name):
        return self[name]

def get_db():
    """Get or create MongoDB connection with fallback."""
    global _client, _db, _is_lite_mode
    if _db is None:
        from config import Config
        try:
            # Try connecting with a short timeout
            _client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=Config.MONGO_TIMEOUT_MS)
            # Check if the server is actually reachable
            _client.server_info()
            _db = _client[Config.MONGO_DB_NAME]
            _ensure_indexes()
            _is_lite_mode = False
        except Exception as e:
            from config import Config
            print(f">>> CRITICAL: Could not connect to MongoDB: {e}")
            print(">>> FALLBACK: Enabling 'Lite Mode' (In-Memory Database)")
            _is_lite_mode = True
            _db = MockDB()
    return _db

def is_lite_mode():
    """Check if the app is currently running without a real database."""
    get_db() # Ensure init
    return _is_lite_mode


def _ensure_indexes():
    """Create indexes for performant queries."""
    db = _db
    db.sessions.create_index([('created_at', DESCENDING)])
    db.candidates.create_index('session_id')
    db.candidates.create_index([('session_id', 1), ('rank', 1)])
    db.feedback.create_index('session_id')
    db.feedback.create_index('job_category')
    db.learned_preferences.create_index('job_category', unique=True)
    db.resume_cache.create_index('file_hash', unique=True)


# ===================== Cache Operations =====================

def get_cached_resume(file_hash):
    """Retrieve a pre-computed resume profile by hash."""
    return get_db().resume_cache.find_one({'file_hash': file_hash})

def cache_resume(file_hash, parsed_data, full_embedding, raw_text):
    """Store computed resume profile to avoid future re-parsing."""
    try:
        get_db().resume_cache.update_one(
            {'file_hash': file_hash},
            {'$set': {
                'parsed': parsed_data,
                'full_embedding': full_embedding,
                'raw_text': raw_text[:2000],
                'cached_at': datetime.utcnow()
            }},
            upsert=True
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to cache resume {file_hash[:8]}: {e}")

# ===================== Session Operations =====================

def create_session(job_description, job_category=None, num_resumes=0, weights=None):
    """Create a new screening session."""
    db = get_db()
    session = {
        'job_description': job_description,
        'job_category': job_category or 'general',
        'num_resumes': num_resumes,
        'processed_count': 0,
        'weights': weights,
        'status': 'processing',
        'created_at': datetime.utcnow(),
        'completed_at': None
    }
    result = db.sessions.insert_one(session)
    return str(result.inserted_id)

def update_session_progress(session_id, processed_count):
    """Update how many resumes have currently been analyzed."""
    from bson import ObjectId
    db = get_db()
    db.sessions.update_one(
        {'_id': ObjectId(session_id)},
        {'$set': {'processed_count': processed_count}}
    )


def update_session_status(session_id, status, num_resumes=None):
    """Update session status after processing."""
    from bson import ObjectId
    db = get_db()
    update = {'status': status}
    if status == 'completed':
        update['completed_at'] = datetime.utcnow()
    if num_resumes is not None:
        update['num_resumes'] = num_resumes
    db.sessions.update_one({'_id': ObjectId(session_id)}, {'$set': update})


def get_session(session_id):
    """Get a single session by ID."""
    from bson import ObjectId
    db = get_db()
    session = db.sessions.find_one({'_id': ObjectId(session_id)})
    if session:
        session['_id'] = str(session['_id'])
    return session


def get_all_sessions():
    """Get all sessions, most recent first."""
    db = get_db()
    sessions = list(db.sessions.find().sort('created_at', DESCENDING))
    for s in sessions:
        s['_id'] = str(s['_id'])
    return sessions


def delete_session(session_id):
    """Delete a session and all its associated candidates, feedback, and analytics."""
    from bson import ObjectId
    db = get_db()
    
    # Needs string form for cascading deletes since they use it as a string
    str_session_id = str(session_id)
    obj_id = ObjectId(session_id)
    
    # 1. Delete candidates
    db.candidates.delete_many({'session_id': str_session_id})
    
    # 2. Delete feedback
    db.feedback.delete_many({'session_id': str_session_id})
    
    # 3. Delete analytics
    db.analytics.delete_one({'session_id': str_session_id})
    
    # 4. Delete the session itself
    result = db.sessions.delete_one({'_id': obj_id})
    
    # Note: We don't delete learned_preferences because those are aggregated 
    # across the job role, not tied exclusively to a session.
    
    return result.deleted_count > 0

# ===================== Candidate Operations =====================

def save_candidates(session_id, candidates_data):
    """Save all ranked candidates for a session.
    
    candidates_data: list of dicts with keys:
        - candidate_name, rank, overall_score, percentile
        - section_scores, section_texts, strengths, gaps
        - embedding (list of floats), raw_text, file_name
    """
    db = get_db()
    for c in candidates_data:
        c['session_id'] = session_id
        c['feedback_status'] = None   # None | 'shortlisted' | 'rejected'
        c['created_at'] = datetime.utcnow()
    if candidates_data:
        db.candidates.insert_many(candidates_data)


def get_candidates(session_id, limit=None):
    """Get ranked candidates for a session."""
    from bson import ObjectId
    db = get_db()
    query = db.candidates.find(
        {'session_id': session_id}
    ).sort('rank', 1)
    if limit:
        query = query.limit(limit)
    candidates = list(query)
    for c in candidates:
        c['_id'] = str(c['_id'])
        # Don't send embedding to frontend (large array)
        c.pop('embedding', None)
    return candidates


def get_candidate(candidate_id):
    """Get a single candidate with full details."""
    from bson import ObjectId
    db = get_db()
    candidate = db.candidates.find_one({'_id': ObjectId(candidate_id)})
    if candidate:
        candidate['_id'] = str(candidate['_id'])
    return candidate


# ===================== Feedback Operations =====================

def save_feedback(session_id, candidate_id, action, original_rank=None, new_rank=None):
    """Save recruiter feedback on a candidate.
    
    action: 'shortlisted' | 'rejected' | 're-ranked'
    """
    from bson import ObjectId
    db = get_db()
    
    # Get candidate details for learning
    candidate = db.candidates.find_one({'_id': ObjectId(candidate_id)})
    if not candidate:
        return None
    
    if action == 'undo':
        db.feedback.delete_many({'session_id': session_id, 'candidate_id': candidate_id})
        db.candidates.update_one(
            {'_id': ObjectId(candidate_id)},
            {'$set': {'feedback_status': None}}
        )
        return "undo"

    # Get session for job category
    session = db.sessions.find_one({'_id': ObjectId(session_id)})
    
    feedback = {
        'session_id': session_id,
        'candidate_id': candidate_id,
        'action': action,
        'original_rank': original_rank or candidate.get('rank'),
        'new_rank': new_rank,
        'job_category': session.get('job_category', 'general') if session else 'general',
        'section_scores': candidate.get('section_scores', {}),
        'embedding': candidate.get('embedding', []),
        'overall_score': candidate.get('overall_score', 0),
        'timestamp': datetime.utcnow()
    }
    result = db.feedback.insert_one(feedback)
    
    # Update candidate's feedback status
    db.candidates.update_one(
        {'_id': ObjectId(candidate_id)},
        {'$set': {'feedback_status': action}}
    )
    
    return str(result.inserted_id)


def get_feedback_for_category(job_category):
    """Get all feedback for a job category (for learning)."""
    db = get_db()
    return list(db.feedback.find({'job_category': job_category}))


def get_session_feedback(session_id):
    """Get all feedback for a session."""
    db = get_db()
    feedbacks = list(db.feedback.find({'session_id': session_id}))
    for f in feedbacks:
        f['_id'] = str(f['_id'])
    return feedbacks


# ===================== Learned Preferences =====================

def get_learned_preferences(job_category):
    """Get the model's learned preferences for a job category."""
    db = get_db()
    prefs = db.learned_preferences.find_one({'job_category': job_category})
    if prefs:
        prefs['_id'] = str(prefs['_id'])
    return prefs


def save_learned_preferences(job_category, adjusted_weights, preference_centroid, alpha, feedback_count):
    """Save/update learned preferences for a job category."""
    db = get_db()
    prefs = {
        'job_category': job_category,
        'adjusted_weights': adjusted_weights,
        'preference_centroid': preference_centroid if isinstance(preference_centroid, list) else preference_centroid.tolist(),
        'alpha': alpha,
        'feedback_count': feedback_count,
        'last_updated': datetime.utcnow()
    }
    db.learned_preferences.update_one(
        {'job_category': job_category},
        {'$set': prefs},
        upsert=True
    )



def get_all_categories():
    """Get all unique job categories across sessions and learned preferences."""
    db = get_db()
    
    # Get from sessions
    cats1 = db.sessions.distinct('job_category')
    
    # Get from learned_preferences
    cats2 = db.learned_preferences.distinct('job_category')
    
    # Combine and clean safely (handle potential non-string values)
    all_cats = set()
    for c in cats1 + cats2:
        if c:
            # Safely convert to string and strip
            all_cats.add(str(c).strip())
    
    # Return sorted list
    return sorted(list(all_cats))


# ===================== Analytics Operations =====================

def save_analytics(session_id, metrics):
    """Save evaluation metrics for a session."""
    db = get_db()
    metrics['session_id'] = session_id
    metrics['created_at'] = datetime.utcnow()
    db.analytics.insert_one(metrics)


def get_analytics(session_id):
    """Get analytics for a session."""
    db = get_db()
    analytics = db.analytics.find_one({'session_id': session_id})
    if analytics:
        analytics['_id'] = str(analytics['_id'])
    return analytics


# ===================== Multi-Role Session Operations =====================

def create_multi_session(roles_data, total_resumes):
    """Create a parent multi-role session.
    
    Args:
        roles_data: list of dicts with keys: title, category, session_id, candidate_count
        total_resumes: total number of unique resumes screened
    
    Returns:
        str: the multi-session ID
    """
    db = get_db()
    multi_session = {
        'type': 'multi_role',
        'roles': roles_data,
        'total_resumes': total_resumes,
        'total_roles': len(roles_data),
        'status': 'completed',
        'created_at': datetime.utcnow(),
    }
    result = db.multi_sessions.insert_one(multi_session)
    return str(result.inserted_id)


def get_multi_session(multi_session_id):
    """Get a multi-role session with all its role metadata."""
    from bson import ObjectId
    db = get_db()
    ms = db.multi_sessions.find_one({'_id': ObjectId(multi_session_id)})
    if ms:
        ms['_id'] = str(ms['_id'])
    return ms


def get_all_multi_sessions():
    """Get all multi-role sessions, most recent first."""
    db = get_db()
    sessions = list(db.multi_sessions.find().sort('created_at', DESCENDING))
    for s in sessions:
        s['_id'] = str(s['_id'])
    return sessions


def delete_multi_session(multi_session_id):
    """Delete a multi-role session and all its child sessions."""
    from bson import ObjectId
    db = get_db()
    
    ms = get_multi_session(multi_session_id)
    if not ms:
        return False
    
    # Delete all child sessions
    for role in ms.get('roles', []):
        child_id = role.get('session_id')
        if child_id:
            delete_session(child_id)
    
    # Delete the parent multi-session
    result = db.multi_sessions.delete_one({'_id': ObjectId(multi_session_id)})
    return result.deleted_count > 0

