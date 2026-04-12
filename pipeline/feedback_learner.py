"""
Feedback Learning Engine — Self-Improving Model

This module implements the LEARNING LOOP that makes the model
improve after each screening round.

Mechanism:
1. Recruiter shortlists/rejects candidates after viewing rankings
2. We analyze the feedback signals to learn:
   a. Which SECTIONS matter most for this job category (weight adaptation)
   b. What a "good candidate" looks like (preference centroid)
   c. Where to draw the shortlist cutoff (threshold calibration)
3. Next screening round automatically uses these learned preferences

Approach: Semi-supervised Online Learning
- Starts as fully unsupervised (no labels)
- Progressively learns from implicit human feedback
- Similar to RLHF (Reinforcement Learning from Human Feedback) in LLMs
- No retraining needed — just vector averaging and weight adjustment
"""

import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def process_feedback(session_id, job_category='general'):
    """
    Process all feedback for a session and update learned preferences.
    
    Called after recruiter provides feedback on candidates.
    Analyzes shortlisted vs rejected candidates to learn:
    1. Optimal section weights for this job category
    2. Preference centroid vector
    3. Alpha blending parameter
    
    Args:
        session_id: The screening session ID
        job_category: Job category for this session
    
    Returns:
        dict with updated preferences
    """
    from config import Config
    from database.db import (
        get_feedback_for_category, get_learned_preferences,
        save_learned_preferences
    )
    
    # Get all feedback for this job category (across ALL sessions)
    all_feedback = get_feedback_for_category(job_category)
    
    if not all_feedback:
        logger.info(f"No feedback yet for category '{job_category}'")
        return None
    
    # Separate shortlisted vs rejected
    shortlisted = [f for f in all_feedback if f['action'] == 'shortlisted']
    rejected = [f for f in all_feedback if f['action'] == 'rejected']
    
    total_feedback = len(shortlisted) + len(rejected)
    logger.info(f"Processing feedback for '{job_category}': "
               f"{len(shortlisted)} shortlisted, {len(rejected)} rejected")
    
    # ====== 1. Adaptive Section Weights ======
    adjusted_weights = _compute_adaptive_weights(shortlisted, rejected, Config)
    
    # ====== 2. Preference Centroid Vector ======
    preference_centroid = _compute_preference_centroid(shortlisted)
    
    # ====== 3. Alpha Blending ======
    # Load existing preferences
    existing_prefs = get_learned_preferences(job_category)
    current_alpha = existing_prefs['alpha'] if existing_prefs else Config.INITIAL_ALPHA
    
    # Gradually reduce alpha (blend in preference bias) as feedback accumulates
    if total_feedback >= Config.MIN_FEEDBACK_FOR_BIAS:
        new_alpha = max(0.5, current_alpha - Config.ALPHA_DECAY)
    else:
        new_alpha = Config.INITIAL_ALPHA
    
    # ====== Save updated preferences ======
    save_learned_preferences(
        job_category=job_category,
        adjusted_weights=adjusted_weights,
        preference_centroid=preference_centroid,
        alpha=new_alpha,
        feedback_count=total_feedback
    )
    
    logger.info(f"Updated preferences for '{job_category}': "
               f"alpha={new_alpha:.2f}, feedback_count={total_feedback}")
    
    return {
        'job_category': job_category,
        'adjusted_weights': adjusted_weights,
        'alpha': new_alpha,
        'feedback_count': total_feedback,
        'shortlisted_count': len(shortlisted),
        'rejected_count': len(rejected),
    }


def _compute_adaptive_weights(shortlisted, rejected, Config):
    """
    Compute optimal section weights based on feedback.
    
    Strategy:
    - For shortlisted candidates, analyze which sections had highest scores
    - For rejected candidates, analyze which sections had lowest scores
    - Shift weights toward sections that DIFFERENTIATE good vs bad candidates
    
    This is inspired by discriminant analysis:
    weight_boost = mean(shortlisted_section_score) - mean(rejected_section_score)
    """
    default_weights = Config.DEFAULT_WEIGHTS.copy()
    learning_rate = Config.LEARNING_RATE
    
    if not shortlisted:
        return default_weights
    
    sections = ['skills', 'experience', 'education', 'projects', 'summary']
    
    # Compute mean section scores for shortlisted
    shortlisted_means = {}
    for section in sections:
        scores = [f['section_scores'].get(section, 0) 
                  for f in shortlisted if f.get('section_scores')]
        shortlisted_means[section] = np.mean(scores) if scores else 0.5
    
    # Compute mean section scores for rejected
    rejected_means = {}
    if rejected:
        for section in sections:
            scores = [f['section_scores'].get(section, 0) 
                      for f in rejected if f.get('section_scores')]
            rejected_means[section] = np.mean(scores) if scores else 0.5
    else:
        rejected_means = {s: 0.5 for s in sections}
    
    # Compute discriminative power of each section
    discriminant = {}
    for section in sections:
        # How much does this section differentiate shortlisted from rejected?
        diff = shortlisted_means[section] - rejected_means[section]
        discriminant[section] = diff
    
    # Adjust weights based on discriminative power
    adjusted = {}
    for section in sections:
        base_weight = default_weights.get(section, 0.2)
        adjustment = learning_rate * discriminant[section]
        adjusted[section] = max(0.02, base_weight + adjustment)  # Floor at 2%
    
    # Normalize to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}
    
    logger.info(f"Adaptive weights: {adjusted}")
    logger.info(f"Discriminant scores: {discriminant}")
    
    return adjusted


def _compute_preference_centroid(shortlisted):
    """
    Compute the preference centroid — the average embedding of
    all shortlisted candidates.
    
    This represents "what a good candidate looks like" for this
    job category, learned from recruiter behavior.
    """
    embeddings = []
    for f in shortlisted:
        emb = f.get('embedding', [])
        if emb and len(emb) > 0:
            embeddings.append(np.array(emb))
    
    if not embeddings:
        return np.zeros(384).tolist()
    
    # Compute centroid (mean of all shortlisted embeddings)
    centroid = np.mean(embeddings, axis=0)
    
    # Normalize
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    
    return centroid.tolist()


def get_learning_stats(job_category='general'):
    """
    Get statistics about the model's learning progress
    for a given job category.
    
    Returns:
        dict with learning stats for display/analytics
    """
    from database.db import get_learned_preferences, get_feedback_for_category
    from config import Config
    
    prefs = get_learned_preferences(job_category)
    feedback = get_feedback_for_category(job_category)
    
    if not prefs:
        return {
            'job_category': job_category,
            'has_learned': False,
            'feedback_count': 0,
            'message': 'No feedback collected yet. The model is using default weights.',
        }
    
    shortlisted_count = len([f for f in feedback if f['action'] == 'shortlisted'])
    rejected_count = len([f for f in feedback if f['action'] == 'rejected'])
    
    # Compare adjusted vs default weights
    default_w = Config.DEFAULT_WEIGHTS
    adjusted_w = prefs['adjusted_weights']
    
    weight_changes = {}
    for section in default_w:
        change = adjusted_w.get(section, default_w[section]) - default_w[section]
        weight_changes[section] = round(change, 4)
    
    # Determine which sections gained / lost importance
    gained = [s for s, c in weight_changes.items() if c > 0.02]
    lost = [s for s, c in weight_changes.items() if c < -0.02]
    
    return {
        'job_category': job_category,
        'has_learned': True,
        'feedback_count': prefs['feedback_count'],
        'shortlisted_count': shortlisted_count,
        'rejected_count': rejected_count,
        'alpha': prefs['alpha'],
        'adjusted_weights': adjusted_w,
        'default_weights': default_w,
        'weight_changes': weight_changes,
        'sections_gained_importance': gained,
        'sections_lost_importance': lost,
        'message': (f"Model has learned from {prefs['feedback_count']} feedback signals. "
                    f"Preference blend: {(1-prefs['alpha'])*100:.0f}% learned + "
                    f"{prefs['alpha']*100:.0f}% JD-based."),
    }
