"""
JSON-Backed Document Store — Database Layer
Handles all database operations for the resume screening system.

This is a lightweight, file-backed document database that persists data
to JSON on disk. It provides a MongoDB-compatible API while requiring
zero external services — works locally and on Hugging Face Spaces.

Collections:
  - sessions: Screening sessions (JD + metadata)
  - candidates: Parsed resume data + embeddings + scores
  - feedback: Recruiter feedback (shortlist/reject/re-rank)
  - learned_preferences: Model's learned weights and preference vectors per category
  - analytics: Evaluation metrics per session
  - resume_cache: Cached parsed resumes by file hash
  - multi_sessions: Multi-role screening parent sessions
"""

import os
import json
import threading
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

_db = None
_DB_FILE = os.path.join(os.path.dirname(__file__), 'store.json')
_SEED_FILE = os.path.join(os.path.dirname(__file__), 'seed_data.json')

# ===================== ID Generation =====================

def _new_id():
    """Generate a new unique document ID (24-char hex string like MongoDB ObjectId)."""
    try:
        from bson import ObjectId
        return str(ObjectId())
    except ImportError:
        import uuid
        return uuid.uuid4().hex[:24]


def _to_id(value):
    """Normalize an ID value to string form."""
    return str(value)


# ===================== JSON Serialization =====================

class _Encoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime and numpy types."""
    def default(self, o):
        if isinstance(o, datetime):
            return {'__datetime__': o.isoformat()}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        try:
            # bson.ObjectId → string
            return str(o)
        except Exception:
            return super().default(o)


def _decode_hook(obj):
    """JSON decode hook that restores datetime objects."""
    if '__datetime__' in obj:
        try:
            return datetime.fromisoformat(obj['__datetime__'])
        except Exception:
            return obj['__datetime__']
    return obj


# ===================== Collection =====================

class Collection:
    """A single named collection of documents with MongoDB-like API."""

    def __init__(self, name, db_ref):
        self.name = name
        self._db = db_ref   # back-reference for triggering save
        self.data = {}       # { str_id: document_dict }

    # --- Write Ops ---

    def insert_one(self, doc):
        if '_id' not in doc:
            doc['_id'] = _new_id()
        doc['_id'] = _to_id(doc['_id'])
        self.data[doc['_id']] = doc
        self._db._save()

        class Result:
            def __init__(self, id):
                self.inserted_id = id
        return Result(doc['_id'])

    def insert_many(self, docs):
        for d in docs:
            if '_id' not in d:
                d['_id'] = _new_id()
            d['_id'] = _to_id(d['_id'])
            self.data[d['_id']] = d
        self._db._save()

    def update_one(self, query, update, upsert=False):
        doc = self.find_one(query)
        if doc:
            if '$set' in update:
                doc.update(update['$set'])
            self._db._save()
        elif upsert:
            new_doc = {}
            # Flatten query into the new doc
            for k, v in query.items():
                new_doc[k] = v
            if '$set' in update:
                new_doc.update(update['$set'])
            self.insert_one(new_doc)

    def delete_one(self, query):
        doc = self.find_one(query)
        if doc:
            del self.data[_to_id(doc['_id'])]
            self._db._save()
            class Result:
                def __init__(self):
                    self.deleted_count = 1
            return Result()
        class Result:
            def __init__(self):
                self.deleted_count = 0
        return Result()

    def delete_many(self, query):
        to_delete = []
        for id_key, doc in self.data.items():
            if self._matches(doc, query):
                to_delete.append(id_key)
        for id_key in to_delete:
            del self.data[id_key]
        if to_delete:
            self._db._save()

    # --- Read Ops ---

    def find_one(self, query):
        if '_id' in query:
            target = _to_id(query['_id'])
            doc = self.data.get(target)
            if doc:
                # Check remaining query fields
                rest = {k: v for k, v in query.items() if k != '_id'}
                if self._matches(doc, rest):
                    return doc
            return None
        for doc in self.data.values():
            if self._matches(doc, query):
                return doc
        return None

    def find(self, query=None):
        query = query or {}
        results = [doc for doc in self.data.values() if self._matches(doc, query)]

        class Cursor:
            def __init__(self, data):
                self._data = data

            def sort(self, field, direction=1):
                self._data.sort(
                    key=lambda x: x.get(field) if x.get(field) is not None else (datetime.min if direction == -1 else datetime.max),
                    reverse=(direction == -1)
                )
                return self

            def limit(self, n):
                self._data = self._data[:n]
                return self

            def __iter__(self):
                return iter(self._data)

            def __getitem__(self, i):
                return self._data[i]

            def __len__(self):
                return len(self._data)

        return Cursor(results)

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
        return sum(1 for d in self.data.values() if self._matches(d, query))

    def create_index(self, *args, **kwargs):
        pass  # No-op; indexes not needed for in-memory store

    # --- Internal ---

    @staticmethod
    def _matches(doc, query):
        """Check if a document matches a simple equality query."""
        for k, v in query.items():
            doc_val = doc.get(k)
            # Normalize ID comparison
            if k == '_id':
                if _to_id(doc_val) != _to_id(v):
                    return False
            elif doc_val != v:
                return False
        return True


# ===================== Document Store =====================

class JsonDocStore:
    """
    A persistent, file-backed document database.
    
    Data is stored in memory for fast access and automatically
    persisted to a JSON file on every write operation.
    Thread-safe via a lock.
    """

    def __init__(self, db_path):
        self._path = db_path
        self._collections = {}
        self._lock = threading.Lock()
        self._load()

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = Collection(name, self)
        return self._collections[name]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self[name]

    # --- Persistence ---

    def _load(self):
        """Load data from disk. Tries store.json first, then seed_data.json."""
        loaded = False

        # 1. Try the primary store
        if os.path.exists(self._path):
            try:
                with open(self._path, 'r') as f:
                    raw = json.load(f, object_hook=_decode_hook)
                self._hydrate(raw)
                doc_count = sum(len(c.data) for c in self._collections.values())
                logger.info(f"JsonDocStore: Loaded {doc_count} documents from {os.path.basename(self._path)}")
                loaded = True
            except Exception as e:
                logger.warning(f"JsonDocStore: Failed to load {self._path}: {e}")

        # 2. Fall back to seed data (first run or missing store)
        if not loaded and os.path.exists(_SEED_FILE):
            try:
                with open(_SEED_FILE, 'r') as f:
                    raw = json.load(f)
                # Seed data has ISO date strings, not our __datetime__ wrapper
                self._hydrate_seed(raw)
                doc_count = sum(len(c.data) for c in self._collections.values())
                logger.info(f"JsonDocStore: Bootstrapped from seed_data.json ({doc_count} documents)")
                # Persist immediately so future loads use store.json
                self._save()
                loaded = True
            except Exception as e:
                logger.warning(f"JsonDocStore: Failed to load seed data: {e}")

        if not loaded:
            logger.info("JsonDocStore: Starting with empty database")

    def _hydrate(self, raw):
        """Populate collections from deserialized JSON (store.json format)."""
        for coll_name, docs in raw.items():
            coll = Collection(coll_name, self)
            if isinstance(docs, list):
                for doc in docs:
                    doc_id = _to_id(doc.get('_id', _new_id()))
                    doc['_id'] = doc_id
                    coll.data[doc_id] = doc
            elif isinstance(docs, dict):
                # Already keyed by ID
                for doc_id, doc in docs.items():
                    doc['_id'] = _to_id(doc.get('_id', doc_id))
                    coll.data[doc['_id']] = doc
            self._collections[coll_name] = coll

    def _hydrate_seed(self, raw):
        """Populate collections from seed_data.json (ISO string dates)."""
        date_fields = ['created_at', 'completed_at', 'timestamp', 'cached_at', 'last_updated']
        for coll_name, docs in raw.items():
            coll = Collection(coll_name, self)
            for doc in docs:
                # Convert date strings
                for field in date_fields:
                    if field in doc and isinstance(doc[field], str):
                        try:
                            doc[field] = datetime.fromisoformat(doc[field])
                        except Exception:
                            pass
                doc_id = _to_id(doc.get('_id', _new_id()))
                doc['_id'] = doc_id
                coll.data[doc_id] = doc
            self._collections[coll_name] = coll

    def _save(self):
        """Persist all collections to disk (thread-safe)."""
        with self._lock:
            try:
                serializable = {}
                for name, coll in self._collections.items():
                    serializable[name] = list(coll.data.values())
                
                # Write atomically: write to temp file, then rename
                tmp_path = self._path + '.tmp'
                with open(tmp_path, 'w') as f:
                    json.dump(serializable, f, cls=_Encoder, separators=(',', ':'))
                os.replace(tmp_path, self._path)
            except Exception as e:
                logger.error(f"JsonDocStore: Failed to save: {e}")


# ===================== Public API =====================

def get_db():
    """Get or create the database instance."""
    global _db
    if _db is None:
        _db = JsonDocStore(_DB_FILE)
        _ensure_indexes()
    return _db


def is_lite_mode():
    """
    Check if the app is running without an external database.
    
    Always returns False now — the JSON store IS the primary database,
    not a fallback. Data is fully persistent.
    """
    get_db()  # Ensure init
    return False


def _ensure_indexes():
    """Create indexes (no-op for JSON store, kept for API compatibility)."""
    db = _db
    db.sessions.create_index([('created_at', -1)])
    db.candidates.create_index('session_id')
    db.candidates.create_index([('session_id', 1), ('rank', 1)])
    db.feedback.create_index('session_id')
    db.feedback.create_index('job_category')
    db.learned_preferences.create_index('job_category')
    db.resume_cache.create_index('file_hash')


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
        logger.error(f"Failed to cache resume {file_hash[:8]}: {e}")

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
    db = get_db()
    db.sessions.update_one(
        {'_id': session_id},
        {'$set': {'processed_count': processed_count}}
    )


def update_session_status(session_id, status, num_resumes=None):
    """Update session status after processing."""
    db = get_db()
    update = {'status': status}
    if status == 'completed':
        update['completed_at'] = datetime.utcnow()
    if num_resumes is not None:
        update['num_resumes'] = num_resumes
    db.sessions.update_one({'_id': session_id}, {'$set': update})


def get_session(session_id):
    """Get a single session by ID."""
    db = get_db()
    session = db.sessions.find_one({'_id': _to_id(session_id)})
    if session:
        session['_id'] = str(session['_id'])
    return session


def get_all_sessions():
    """Get all sessions, most recent first."""
    db = get_db()
    sessions = list(db.sessions.find().sort('created_at', -1))
    for s in sessions:
        s['_id'] = str(s['_id'])
    return sessions


def delete_session(session_id):
    """Delete a session and all its associated candidates, feedback, and analytics."""
    db = get_db()
    
    str_session_id = str(session_id)
    
    # 1. Delete candidates
    db.candidates.delete_many({'session_id': str_session_id})
    
    # 2. Delete feedback
    db.feedback.delete_many({'session_id': str_session_id})
    
    # 3. Delete analytics
    db.analytics.delete_one({'session_id': str_session_id})
    
    # 4. Delete the session itself
    result = db.sessions.delete_one({'_id': str_session_id})
    
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
    db = get_db()
    candidate = db.candidates.find_one({'_id': _to_id(candidate_id)})
    if candidate:
        candidate['_id'] = str(candidate['_id'])
    return candidate


# ===================== Feedback Operations =====================

def save_feedback(session_id, candidate_id, action, original_rank=None, new_rank=None):
    """Save recruiter feedback on a candidate.
    
    action: 'shortlisted' | 'rejected' | 're-ranked'
    """
    db = get_db()
    
    # Get candidate details for learning
    candidate = db.candidates.find_one({'_id': _to_id(candidate_id)})
    if not candidate:
        return None
    
    if action == 'undo':
        db.feedback.delete_many({'session_id': session_id, 'candidate_id': candidate_id})
        db.candidates.update_one(
            {'_id': _to_id(candidate_id)},
            {'$set': {'feedback_status': None}}
        )
        return "undo"

    # Get session for job category
    session = db.sessions.find_one({'_id': _to_id(session_id)})
    
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
        {'_id': _to_id(candidate_id)},
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
    db = get_db()
    ms = db.multi_sessions.find_one({'_id': _to_id(multi_session_id)})
    if ms:
        ms['_id'] = str(ms['_id'])
    return ms


def get_all_multi_sessions():
    """Get all multi-role sessions, most recent first."""
    db = get_db()
    sessions = list(db.multi_sessions.find().sort('created_at', -1))
    for s in sessions:
        s['_id'] = str(s['_id'])
    return sessions


def delete_multi_session(multi_session_id):
    """Delete a multi-role session and all its child sessions."""
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
    result = db.multi_sessions.delete_one({'_id': _to_id(multi_session_id)})
    return result.deleted_count > 0
