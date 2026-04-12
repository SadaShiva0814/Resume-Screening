"""
Semantic Scorer — Section-Weighted Similarity Engine

Computes semantic similarity between each resume section and the
job description, then produces a weighted final score.

This is where the MAGIC happens:
- Each section gets its own similarity score against the JD
- Weights determine how much each section matters
- Strengths and gaps are identified through semantic comparison
- The system UNDERSTANDS context, not just keywords
"""

import numpy as np
import logging
from models.embedder import encode_text, encode_resume_sections, cosine_similarity
from pipeline.semantic_hardening import clean_negated_content
from pipeline.ontology import SKILL_ONTOLOGY, standardize_skill

logger = logging.getLogger(__name__)

def compute_section_scores(parsed_resume, jd_embedding, jd_text=""):
    """
    Compute semantic similarity between each resume section and the JD.
    Uses "Signal Isolation" (Max Paragraph Similarity) to prevent 
    distractor content from diluting strong matches.
    """
    sections = parsed_resume.get('sections', {})
    
    # Extract key skills from JD for overlap bonus
    jd_skills_raw = _extract_jd_requirements(jd_text) if jd_text else []
    # Standardize JD skills using ontology
    jd_skills = [standardize_skill(s) for s in jd_skills_raw]
    
    resume_skills_raw = [s.lower() for s in parsed_resume.get('extracted_skills', [])]
    resume_skills = [standardize_skill(s) for s in resume_skills_raw]
    
    section_scores = {}
    
    for section_name, section_text in sections.items():
        if not section_text.strip():
            section_scores[section_name] = 0.0
            continue
            
        # 1. Clean negations from text
        cleaned_text = clean_negated_content(section_text)
        
        # Encode all paragraphs and find the maximum similarity
        # 2. Split into paragraphs for Signal Isolation
        paragraphs = [p for p in cleaned_text.split('\n\n') if len(p.strip()) > 20]
        if not paragraphs:
            paragraphs = [cleaned_text]
            
        para_scores = []
        for p in paragraphs:
            p_vec = encode_text(p)
            similarity = cosine_similarity(p_vec, jd_embedding)
            
            # --- SIGNAL DENSITY CALCULATION ---
            # How many JD requirements are found in this SPECIFIC paragraph?
            p_lower = p.lower()
            p_matches = sum(1 for s in jd_skills if s.lower() in p_lower)
            
            # Density Bonus: High concentration of skills = boost. 
            # Low concentration = dilution penalty.
            density_factor = 0.8 # Base
            if len(jd_skills) > 0:
                density_pct = p_matches / len(jd_skills)
                # Boost based on density (cap at 1.4x)
                density_factor = 0.8 + min(0.6, density_pct * 2.0)
                
            # Weighted paragraph score
            para_scores.append(similarity * density_factor)
        
        # Base score is the BEST weighted match found in the section
        score = max(para_scores) if para_scores else 0.0
        
        # 3. Global Skill-Overlap Bonus (using Ontological matching)
        if section_name == 'skills' and jd_skills:
            matches = sum(1 for s in jd_skills if s.lower() in [rs.lower() for rs in resume_skills])
            # Higher bonus for more hits across the whole skills section
            # Increased floor: Any match gives significant foundational credit
            if matches > 0:
                bonus = 0.10 + min(0.35, (matches / len(jd_skills)) * 0.55)
            else:
                bonus = 0
            score += bonus
            logger.info(f"Ontology Skill bonus applied to {section_name}: +{bonus:.2f} ({matches} hits)")
            
        # Clamp to [0, 1.2] range
        section_scores[section_name] = max(0.0, min(1.3, score)) 
    
    # Full resume embedding remains as baseline
    full_text = ' '.join(text for text in sections.values() if text.strip())
    full_embedding = encode_text(full_text)
    
    return {
        'section_scores': section_scores,
        'full_embedding': full_embedding,
    }


def normalize_weights(weights):
    """
    Ensure weights sum to 1.0 (100%).
    If they don't, normalize them proportionally.
    If the user has provided Skills/Experience but no Projects, 
    the remainder is automatically assigned to Projects.
    """
    if not weights:
        return {}
    
    # Create a copy to avoid mutating original
    working_weights = weights.copy()
    
    # "Automatic Remainder" logic: If 'projects' is missing or 0, and total < 100
    total = sum(v for v in working_weights.values() if v > 0)
    if 'projects' not in working_weights or working_weights['projects'] == 0:
        if total < 100:
            working_weights['projects'] = 100 - total
            total = 100
    
    # Normalize to 1.0
    if total > 0:
        return {k: v / total for k, v in working_weights.items()}
    
    return working_weights


def compute_weighted_score(section_scores, weights):
    """
    Compute the final weighted score from section scores.
    Weights are automatically normalized to sum to 1.0.
    """
    # Normalize weights before computing (Sum to 100 logic)
    norm_weights = normalize_weights(weights)
    
    total_score = 0.0
    total_weight = 0.0
    
    for section, weight in norm_weights.items():
        if section in section_scores and section_scores[section] > 0:
            score = section_scores[section]
            total_score += score * weight
            total_weight += weight
        else:
            # Section missing — don't include in total_weight to avoid dragging down
            # the average, but apply a small "missing info" penalty to the total weight 
            # if it's a core section.
            if section in ['skills', 'experience']:
                # Include a fraction of the weight as a penalty (implicit 0.2 score)
                total_score += 0.2 * weight
                total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return total_score / total_weight


def analyze_strengths_gaps(parsed_resume, jd_text, section_scores):
    """
    Identify candidate strengths and gaps through SEMANTIC analysis.
    
    This goes beyond keyword matching:
    - Compare JD requirements against resume skills semantically
    - Flag sections where the candidate excels vs falls short
    - Provide actionable insights for the recruiter
    
    Returns:
        tuple: (strengths: list[str], gaps: list[str])
    """
    strengths = []
    gaps = []
    
    sections = parsed_resume.get('sections', {})
    extracted_skills = parsed_resume.get('extracted_skills', [])
    years_exp = parsed_resume.get('years_of_experience', 0)
    edu_level = parsed_resume.get('education_level', 'unknown')
    
    # Analyze section scores
    score_threshold_high = 0.55   # Above this = strength
    score_threshold_low = 0.30    # Below this = gap
    
    section_labels = {
        'skills': 'Technical Skills',
        'experience': 'Work Experience',
        'education': 'Educational Background',
        'projects': 'Projects',
        'summary': 'Professional Profile'
    }
    
    for section, score in section_scores.items():
        label = section_labels.get(section, section.title())
        if score >= score_threshold_high:
            strengths.append(f"Strong {label} alignment (score: {score:.0%})")
        elif score < score_threshold_low and sections.get(section, '').strip():
            gaps.append(f"Weak {label} match (score: {score:.0%})")
        elif not sections.get(section, '').strip():
            gaps.append(f"No {label} information found in resume")
    
    # Skills-based analysis
    if extracted_skills:
        if len(extracted_skills) >= 10:
            strengths.append(f"Diverse skill set ({len(extracted_skills)} skills detected)")
        
        # Check for JD skill mentions in resume skills
        _analyze_skill_coverage(jd_text, extracted_skills, strengths, gaps)
    else:
        gaps.append("Could not extract specific skills from resume")
    
    # Experience analysis
    if years_exp > 0:
        if years_exp >= 5:
            strengths.append(f"Significant experience ({years_exp:.0f}+ years)")
        elif years_exp >= 2:
            strengths.append(f"Moderate experience ({years_exp:.0f} years)")
        else:
            strengths.append(f"Entry-level ({years_exp:.0f} year(s) experience)")
    
    # Education analysis
    edu_rank = {'phd': 5, 'masters': 4, 'bachelors': 3, 'diploma': 2, 'high_school': 1}
    if edu_level in edu_rank and edu_rank[edu_level] >= 3:
        strengths.append(f"Strong educational background ({edu_level.replace('_', ' ').title()})")
    elif edu_level == 'unknown':
        gaps.append("Education level could not be determined")
    
    return strengths[:8], gaps[:6]  # Cap for clean display


def _analyze_skill_coverage(jd_text, resume_skills, strengths, gaps):
    """
    Semantically check how well resume skills cover JD requirements.
    Uses embedding similarity for near-match detection.
    """
    if not jd_text:
        return
    
    jd_lower = jd_text.lower()
    
    # Quick check: which resume skills appear in JD (directly or semantically)
    matched_skills = []
    for skill in resume_skills[:20]:  # Top-20 skills
        if skill.lower() in jd_lower:
            matched_skills.append(skill)
    
    if matched_skills:
        skills_str = ', '.join(matched_skills[:5])
        more = f" (+{len(matched_skills)-5} more)" if len(matched_skills) > 5 else ""
        strengths.append(f"Key JD skills present: {skills_str}{more}")
    
    # Check for commonly required skills missing from resume
    common_jd_skills = _extract_jd_requirements(jd_text)
    missing = [s for s in common_jd_skills if s.lower() not in 
               [rs.lower() for rs in resume_skills]]
    
    if missing:
        missing_str = ', '.join(missing[:4])
        gaps.append(f"JD skills not found in resume: {missing_str}")


def _extract_jd_requirements(jd_text):
    """Extract key skill requirements from a job description."""
    import re
    
    # Common patterns in JDs
    patterns = [
        r'(?:required|must have|essential|mandatory)[\s:]*(.+?)(?:\n|$)',
        r'(?:experience (?:with|in))[\s:]*(.+?)(?:\n|$)',
        r'(?:proficien(?:t|cy) (?:in|with))[\s:]*(.+?)(?:\n|$)',
    ]
    
    from pipeline.section_parser import KNOWN_SKILLS
    
    jd_lower = jd_text.lower()
    found_skills = []
    for skill in KNOWN_SKILLS:
        if skill.lower() in jd_lower:
            found_skills.append(skill.title() if len(skill) > 3 else skill.upper())
    
    return found_skills[:10]
