"""
Ranking Engine — Final Candidate Ranking Pipeline

Orchestrates the full screening pipeline:
1. Extract text from all resumes
2. Parse sections with NLP
3. Encode with Sentence-BERT
4. Compute weighted semantic similarity
5. Incorporate learned preferences (if available)
6. Rank candidates
7. Generate strengths/gaps analysis

This is the main entry point that ties everything together.
"""

import os
import re
import logging
import numpy as np
from datetime import datetime

from models.embedder import encode_text, cosine_similarity
from pipeline.extractor import extract_text, extract_text_from_string
from pipeline.section_parser import parse_resume
from pipeline.semantic_scorer import compute_section_scores, compute_weighted_score, analyze_strengths_gaps

logger = logging.getLogger(__name__)


def screen_resumes(resume_inputs, job_description, weights=None, job_category='general'):
    """
    Main screening function — ranks all resumes against a job description.
    
    Args:
        resume_inputs: List of dicts, each with:
            - 'file_path': path to resume file (PDF/DOCX/TXT)
            OR
            - 'text': raw resume text (for Kaggle dataset)
            - 'file_name': original filename or identifier
        job_description: str, the job description text
        weights: dict, section weights (default from config)
        job_category: str, for learned preferences lookup
    
    Returns:
        list of ranked candidate dicts
    """
    from config import Config
    
    if weights is None:
        weights = Config.DEFAULT_WEIGHTS.copy()
    
    # ====== Step 1: Check for learned preferences ======
    preference_centroid = None
    alpha = Config.INITIAL_ALPHA
    
    try:
        from database.db import get_learned_preferences
        prefs = get_learned_preferences(job_category)
        if prefs and prefs.get('feedback_count', 0) >= Config.MIN_FEEDBACK_FOR_BIAS:
            weights = prefs['adjusted_weights']
            preference_centroid = np.array(prefs['preference_centroid'])
            alpha = prefs['alpha']
            logger.info(f"Using learned preferences for '{job_category}' "
                       f"(feedback_count={prefs['feedback_count']}, alpha={alpha:.2f})")
    except Exception as e:
        logger.warning(f"Could not load learned preferences: {e}")
    
    # ====== Step 2: Encode the Job Description ======
    logger.info("Encoding job description...")
    jd_embedding = encode_text(job_description)
    
    # ====== Step 3: Process each resume ======
    candidates = []
    total = len(resume_inputs)
    
    for i, resume_input in enumerate(resume_inputs):
        try:
            logger.info(f"Processing resume {i+1}/{total}: {resume_input.get('file_name', 'unknown')}")
            
            # Extract text
            if 'text' in resume_input and resume_input['text']:
                raw_text = extract_text_from_string(resume_input['text'])
            elif 'file_path' in resume_input:
                raw_text = extract_text(resume_input['file_path'])
            else:
                logger.warning(f"Skipping resume {i+1}: no text or file_path")
                continue
            
            if not raw_text or len(raw_text.strip()) < 30:
                logger.warning(f"Skipping resume {i+1}: insufficient text extracted")
                continue
            
            # Parse sections — prefer HTML-based extraction for Kaggle data
            html_content = resume_input.get('html', '')
            html_header = None  # Will hold job title/name from HTML
            
            if html_content and len(html_content) > 100:
                # Use structured HTML for section extraction (much more accurate)
                from pipeline.extractor import extract_sections_from_html
                html_sections = extract_sections_from_html(html_content)
                
                # Extract the header (name/job title) from HTML sections
                html_header = html_sections.pop('header', None)
                
                if html_sections and sum(1 for v in html_sections.values() if v.strip()) >= 2:
                    # Good HTML extraction — use these sections
                    parsed = parse_resume(raw_text, html_header=html_header)
                    # Override sections with HTML-extracted ones (better structure)
                    for section_name, section_text in html_sections.items():
                        if section_text.strip():
                            parsed['sections'][section_name] = section_text
                    # Re-extract skills from the now-better skills section
                    from pipeline.section_parser import _extract_skills, _get_nlp, _format_section_text
                    parsed['extracted_skills'] = _extract_skills(
                        parsed['sections'].get('skills', ''),
                        raw_text, _get_nlp()
                    )
                    # Rebuild display_sections after overriding
                    for section_name, section_text in parsed['sections'].items():
                        if section_text and section_text.strip():
                            parsed['display_sections'][section_name] = _format_section_text(section_text, section_name)
                else:
                    parsed = parse_resume(raw_text, html_header=html_header)
            else:
                parsed = parse_resume(raw_text)
            
            # Compute section-wise semantic scores
            scoring = compute_section_scores(parsed, jd_embedding, job_description)
            section_scores = scoring['section_scores']
            full_embedding = scoring['full_embedding']
            
            # Compute weighted score
            jd_score = compute_weighted_score(section_scores, weights)
            
            # ====== Apply Hard Requirement Penalties (Stack Guard & Seniority) ======
            from pipeline.ontology import identify_jd_anchor
            jd_anchor = identify_jd_anchor(job_description)
            
            # Check for stack mismatch (Anchor Language Guard)
            has_anchor = any(jd_anchor in s.lower() for s in parsed['extracted_skills'])
            
            if jd_anchor != 'general':
                if has_anchor:
                    # PROACTIVE BONUS for having the right stack
                    jd_score += 0.08
                    logger.info(f"Applied Anchor Affinity Bonus to {parsed['contact']['name']} (Anchor: {jd_anchor.upper()})")
                else:
                    # Apply heavy stack mismatch penalty
                    jd_score *= 0.60 # 40% reduction for wrong stack
                    logger.info(f"Applied Stack-Guard penalty to {parsed['contact']['name']} (Missing Anchor: {jd_anchor.upper()})")
            
            years_exp = parsed.get('years_of_experience', 0)
            
            # --- OVERQUALIFICATION GUARD ---
            # Extract upper bound of JD experience (e.g., "0 to 3 years" -> 3)
            jd_max_match = re.search(r'(?:up|to|max|maximum|limit)\s*(?:of\b)?\s*(\d+)\s*(?:years?|yrs?)', job_description, re.IGNORECASE)
            if not jd_max_match:
                # Try "2-4 years" pattern
                jd_max_match = re.search(r'\d+\s*[-–to]\s*(\d+)\s*(?:years?|yrs?)', job_description, re.IGNORECASE)
            
            if jd_max_match:
                jd_max_target = float(jd_max_match.group(1))
                # If candidate exceeds target by more than 10 years (User specified threshold)
                if years_exp > (jd_max_target + 10):
                    jd_score *= 0.80 # 20% overqualification penalty
                    logger.info(f"Applied Overqualification penalty to {parsed['contact']['name']} ({years_exp} yrs vs Max {jd_max_target})")

            # --- UNDERQUALIFICATION / SENIORITY PENALTY ---
            # 1. Look for explicit years requirement (e.g., "5+ years", "minimum of 3 years")
            min_years_required = 0
            year_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\b\s*(?:of\s+)?(?:experience|exp)?\b', job_description, re.IGNORECASE)
            if year_match:
                min_years_required = float(year_match.group(1))
            
            # 2. Check for professional role keywords
            is_pro_role = bool(re.search(r'\b(senior|lead|architect|principal|staff)\b', job_description, re.IGNORECASE))
            
            # Default minimum for Pro roles if not explicit
            if is_pro_role and min_years_required == 0:
                min_years_required = 5.0
            
            # 3. Apply Penalties
            seniority_penalty_applied = 1.0
            if (is_pro_role or min_years_required > 2) and years_exp < 1.0:
                # Case A: Near-zero experience for Pro/Required-Exp roles (Critical Mismatch)
                seniority_penalty_applied = 0.40  # 60% reduction for entry-level applying for Senior
                logger.info(f"Applied CRITICAL seniority penalty (60%) to {parsed['contact']['name']} ({years_exp} yrs vs {min_years_required} required)")
            elif min_years_required > 0 and years_exp < min_years_required:
                # Case B: Significant experience gap
                gap_ratio = years_exp / min_years_required
                # Non-linear penalty: harsher for very small ratios
                seniority_penalty_applied = 0.65 + (pow(gap_ratio, 1.5) * 0.35)
                logger.info(f"Applied dynamic seniority penalty to {parsed['contact']['name']} ({years_exp} yrs vs {min_years_required} required, penalty={seniority_penalty_applied:.2f})")

            jd_score *= seniority_penalty_applied
            
            # [0.18 -> 1.05] maps to [0.05 -> 0.98]
            # Expanded upper bound from 0.75 to 1.05 to prevent score bunching at the 98% cap.
            # This ensures that only candidates with BOTH perfect skills and correct seniority 
            # achieve top-tier scores.
            calibrated_score = (jd_score - 0.18) / (1.05 - 0.18)
            calibrated_score = max(0.05, min(0.98, calibrated_score))
            
            # ====== Blend with preference bias (if available) ======
            final_score = calibrated_score
            if preference_centroid is not None:
                pref_score = cosine_similarity(full_embedding, preference_centroid)
                pref_score = max(0.0, min(1.0, pref_score))
                # Preference score is already calibrated usually, but we blend it here
                final_score = alpha * calibrated_score + (1 - alpha) * pref_score
            
            # Analyze strengths and gaps
            strengths, gaps = analyze_strengths_gaps(parsed, job_description, section_scores)
            
            # Build candidate record
            candidate_role = parsed.get('contact', {}).get('role', '')
            
            candidate = {
                'file_name': resume_input.get('file_name', f'resume_{i+1}'),
                'candidate_name': parsed['contact']['name'] or f"Candidate {i+1}",
                'role': candidate_role,
                'email': parsed['contact']['email'],
                'phone': parsed['contact']['phone'],
                'linkedin': parsed['contact'].get('linkedin', ''),
                'github': parsed['contact'].get('github', ''),
                'overall_score': round(final_score, 4),
                'jd_similarity_score': round(jd_score, 4),
                'section_scores': {k: round(v, 4) for k, v in section_scores.items()},
                'section_texts': parsed.get('display_sections', parsed['sections']),  # Use formatted text for display
                'extracted_skills': parsed['extracted_skills'],
                'years_of_experience': parsed['years_of_experience'],
                'education_level': parsed['education_level'],
                'strengths': strengths,
                'gaps': gaps,
                'embedding': full_embedding.tolist(),
                'raw_text': raw_text[:2000],  # Store first 2000 chars for reference
            }
            
            candidates.append(candidate)
            
        except Exception as e:
            logger.error(f"Error processing resume {i+1}: {e}", exc_info=True)
            continue
    
    if not candidates:
        logger.warning("No candidates could be processed!")
        return []
    
    # ====== Step 4: Rank candidates ======
    candidates.sort(key=lambda c: c['overall_score'], reverse=True)
    
    # Assign ranks and percentiles
    total_candidates = len(candidates)
    for rank, candidate in enumerate(candidates, 1):
        candidate['rank'] = rank
        candidate['percentile'] = round((1 - (rank - 1) / total_candidates) * 100, 1)
    
    logger.info(f"Screening complete: {total_candidates} candidates ranked")
    return candidates


def screen_resumes_from_files(file_paths, job_description, weights=None, job_category='general'):
    """
    Convenience function: screen resumes from file paths.
    
    Args:
        file_paths: List of file paths to resume files
        job_description: str
        weights: dict (optional)
        job_category: str
    
    Returns:
        list of ranked candidate dicts
    """
    resume_inputs = [
        {'file_path': fp, 'file_name': os.path.basename(fp)}
        for fp in file_paths
    ]
    return screen_resumes(resume_inputs, job_description, weights, job_category)


def screen_resumes_from_texts(resume_texts, job_description, weights=None, job_category='general'):
    """
    Convenience function: screen resumes from raw text (e.g., Kaggle CSV).
    
    Args:
        resume_texts: List of (text, label/filename) tuples
        job_description: str
        weights: dict (optional)
        job_category: str
    
    Returns:
        list of ranked candidate dicts
    """
    resume_inputs = [
        {'text': text, 'file_name': label}
        for text, label in resume_texts
    ]
    return screen_resumes(resume_inputs, job_description, weights, job_category)


def screen_resumes_multi_role(resume_inputs, job_descriptions, weights=None):
    """
    Multi-Role Matching — Screen resumes against multiple JDs simultaneously.
    
    Each resume is scored against every JD. Candidates are routed to the
    role where they score highest. Candidates scoring well on 2+ roles
    are flagged as "versatile".
    
    Args:
        resume_inputs: List of resume dicts (same format as screen_resumes)
        job_descriptions: List of dicts, each with:
            - 'title': role name (e.g. "Backend Engineer")  
            - 'text': full JD text
            - 'category': job category for feedback learning (optional)
        weights: dict (optional, shared section weights)
    
    Returns:
        dict with:
            - 'roles': {role_title: list of ranked candidate dicts}
            - 'versatile': list of candidates appearing in 2+ roles
            - 'unmatched': list of candidates below threshold on all roles
            - 'role_meta': list of {title, category, candidate_count}
    """
    VERSATILE_THRESHOLD = 0.10   # Within 10% of best score → versatile
    UNMATCHED_THRESHOLD = 0.30   # Below 30% on all roles → unmatched
    
    if not job_descriptions:
        logger.warning("No job descriptions provided for multi-role screening")
        return {'roles': {}, 'versatile': [], 'unmatched': [], 'role_meta': []}
    
    # ====== Step 1: Screen all resumes against each JD independently ======
    role_results = {}   # {role_title: [ranked candidates]}
    
    for jd in job_descriptions:
        title = jd['title']
        text = jd['text']
        category = jd.get('category', title.lower().replace(' ', '_'))
        
        logger.info(f"Multi-Role: Screening {len(resume_inputs)} resumes for '{title}'")
        
        candidates = screen_resumes(
            resume_inputs=resume_inputs,
            job_description=text,
            weights=weights,
            job_category=category
        )
        
        role_results[title] = candidates
    
    # ====== Step 2: Build per-candidate score matrix ======
    # Map: file_name → {role_title: score, ...}
    candidate_scores = {}   # file_name → {role: score}
    candidate_data = {}     # file_name → {role: full candidate dict}
    
    for role_title, candidates in role_results.items():
        for c in candidates:
            fname = c['file_name']
            if fname not in candidate_scores:
                candidate_scores[fname] = {}
                candidate_data[fname] = {}
            candidate_scores[fname][role_title] = c['overall_score']
            candidate_data[fname][role_title] = c
    
    # ====== Step 3: Route candidates to best-fit role ======
    routed = {title: [] for title in role_results}   # role → [candidates]
    versatile = []
    unmatched = []
    assigned_to = {}   # file_name → best role title
    
    for fname, scores in candidate_scores.items():
        if not scores:
            continue
        
        best_role = max(scores, key=scores.get)
        best_score = scores[best_role]
        
        # Check if unmatched (below threshold on ALL roles)
        if best_score < UNMATCHED_THRESHOLD:
            # Use the best-scoring version for the unmatched list
            unmatched_candidate = candidate_data[fname][best_role].copy()
            unmatched_candidate['best_role'] = best_role
            unmatched_candidate['all_role_scores'] = {r: round(s, 4) for r, s in scores.items()}
            unmatched.append(unmatched_candidate)
            continue
        
        # Assign to best role
        best_candidate = candidate_data[fname][best_role].copy()
        best_candidate['all_role_scores'] = {r: round(s, 4) for r, s in scores.items()}
        routed[best_role].append(best_candidate)
        assigned_to[fname] = best_role
        
        # Check for versatility (scored close to best on other roles too)
        other_strong_roles = []
        for role, score in scores.items():
            if role != best_role and (best_score - score) <= VERSATILE_THRESHOLD and score >= UNMATCHED_THRESHOLD:
                other_strong_roles.append({'role': role, 'score': round(score, 4)})
        
        if other_strong_roles:
            versatile.append({
                'file_name': fname,
                'candidate_name': best_candidate.get('candidate_name', fname),
                'primary_role': best_role,
                'primary_score': round(best_score, 4),
                'other_roles': other_strong_roles,
                'all_role_scores': {r: round(s, 4) for r, s in scores.items()},
            })
    
    # ====== Step 4: Re-rank within each role ======
    for role_title in routed:
        role_candidates = routed[role_title]
        role_candidates.sort(key=lambda c: c['overall_score'], reverse=True)
        for rank, c in enumerate(role_candidates, 1):
            c['rank'] = rank
            c['percentile'] = round((1 - (rank - 1) / max(len(role_candidates), 1)) * 100, 1)
    
    # Build role metadata
    role_meta = []
    for jd in job_descriptions:
        title = jd['title']
        role_meta.append({
            'title': title,
            'category': jd.get('category', title.lower().replace(' ', '_')),
            'candidate_count': len(routed.get(title, [])),
        })
    
    logger.info(f"Multi-Role complete: {len(job_descriptions)} roles, "
                f"{sum(len(v) for v in routed.values())} routed, "
                f"{len(versatile)} versatile, {len(unmatched)} unmatched")
    
    return {
        'roles': routed,
        'versatile': versatile,
        'unmatched': unmatched,
        'role_meta': role_meta,
    }


def keyword_baseline_ranking(resume_texts_list, job_description):
    """
    Simple TF-IDF keyword-based ranking (baseline for comparison).
    
    Used to demonstrate the IMPROVEMENT of our semantic approach
    over traditional keyword matching ATS systems.
    
    Returns:
        list of dicts with keyword_score and rank
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    
    # Combine JD with all resumes
    all_texts = [job_description] + resume_texts_list
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Cosine similarity of each resume against JD
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    
    similarities = sklearn_cosine(jd_vector, resume_vectors).flatten()
    
    # Create ranked list
    results = []
    for i, score in enumerate(similarities):
        results.append({
            'index': i,
            'keyword_score': round(float(score), 4)
        })
    
    results.sort(key=lambda x: x['keyword_score'], reverse=True)
    
    for rank, result in enumerate(results, 1):
        result['keyword_rank'] = rank
    
    return results
