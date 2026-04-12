"""
Semantic Embedding Engine — Sentence-BERT Wrapper

This is the ML backbone of the resume screening system.
Uses Sentence-BERT (all-MiniLM-L6-v2) to encode text into
dense 384-dimensional vectors that capture SEMANTIC MEANING.

Why this works for semantic understanding:
- Trained on 1B+ sentence pairs for semantic textual similarity
- "Python developer with 3 years experience" and "Software engineer proficient in Python"
  will have HIGH cosine similarity despite different words
- This is what makes us DIFFERENT from keyword-based ATS

Key features:
- Section-aware encoding (encodes each resume section separately)
- Embedding caching for performance
- Batch processing for screening 100s of resumes efficiently
"""

import logging
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load the Sentence-BERT model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from config import Config
        logger.info(f"Loading Sentence-BERT model: {Config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(Config.EMBEDDING_MODEL)
        logger.info("Model loaded successfully")
    return _model


def encode_text(text, normalize=True):
    """
    Encode a single text string into a 384-dim embedding vector.
    
    Args:
        text: Input text string
        normalize: If True, L2-normalize the vector (for cosine similarity)
    
    Returns:
        numpy array of shape (384,)
    """
    if not text or not text.strip():
        return np.zeros(384)
    
    model = _get_model()
    
    # Truncate very long texts (model has ~256 token limit for best quality)
    # We keep the first 500 words as a good balance
    words = text.split()
    if len(words) > 500:
        text = ' '.join(words[:500])
    
    embedding = model.encode(text, normalize_embeddings=normalize, show_progress_bar=False)
    return np.array(embedding)


def encode_texts_batch(texts, normalize=True, batch_size=32):
    """
    Encode multiple texts in batch (much faster than one by one).
    
    Args:
        texts: List of text strings
        normalize: L2-normalize vectors
        batch_size: Batch size for encoding
    
    Returns:
        numpy array of shape (len(texts), 384)
    """
    model = _get_model()
    
    # Handle empty texts
    processed = []
    for text in texts:
        if not text or not text.strip():
            processed.append("empty")
        else:
            words = text.split()
            processed.append(' '.join(words[:500]) if len(words) > 500 else text)
    
    embeddings = model.encode(
        processed,
        normalize_embeddings=normalize,
        show_progress_bar=len(processed) > 10,
        batch_size=batch_size
    )
    
    return np.array(embeddings)


def encode_resume_sections(parsed_resume):
    """
    Encode each section of a parsed resume separately.
    
    This is the KEY innovation: instead of encoding the entire resume
    as one blob (losing section context), we encode each section
    independently. This allows section-weighted scoring.
    
    Args:
        parsed_resume: Output from section_parser.parse_resume()
    
    Returns:
        dict: {section_name: numpy array (384,)}
    """
    sections = parsed_resume.get('sections', {})
    section_embeddings = {}
    
    core_sections = ['summary', 'skills', 'experience', 'education', 'projects']
    
    for section_name in core_sections:
        section_text = sections.get(section_name, '')
        
        if section_text.strip():
            section_embeddings[section_name] = encode_text(section_text)
        else:
            section_embeddings[section_name] = np.zeros(384)
    
    return section_embeddings


def encode_resume_full(parsed_resume):
    """
    Also encode the full resume text as a single vector.
    Useful for overall similarity and clustering.
    
    Args:
        parsed_resume: Output from section_parser.parse_resume()
    
    Returns:
        numpy array of shape (384,)
    """
    sections = parsed_resume.get('sections', {})
    full_text = ' '.join(
        text for text in sections.values() if text.strip()
    )
    return encode_text(full_text)


def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    
    Returns value between -1 and 1 (usually 0 to 1 for normalized vectors).
    Higher = more semantically similar.
    """
    if np.all(vec_a == 0) or np.all(vec_b == 0):
        return 0.0
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def cosine_similarity_batch(query_vec, doc_vecs):
    """
    Compute cosine similarity between a query vector and batch of document vectors.
    
    Args:
        query_vec: numpy array of shape (384,)
        doc_vecs: numpy array of shape (N, 384)
    
    Returns:
        numpy array of shape (N,) with similarity scores
    """
    if len(doc_vecs) == 0:
        return np.array([])
    
    # Normalize
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    
    # Batch dot product
    similarities = np.dot(doc_norms, query_norm)
    return similarities
