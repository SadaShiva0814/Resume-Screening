"""
Semantic Hardening — Syntactic & Domain Analysis

Ensures that the semantic engine is not fooled by negative
qualifiers (e.g., "No experience with Java") or domain distractors.
"""

import re
import logging

logger = logging.getLogger(__name__)

# List of negation patterns in engineering contexts
NEGATION_PATTERNS = [
    r'\bno(?:t)?\s*(?:experience|knowledge|proficien(?:t|cy))\s+(?:with|in|of)\s+([a-zA-Z0-9+#.]+)',
    r'\black\s+of\s+([a-zA-Z0-9+#.]+)',
    r'\blimited\s*(?:knowledge|exposure)?\s+(?:to|with)\s+([a-zA-Z0-9+#.]+)',
    r'\bnever\s+(?:worked|used)\s+(?:with|in)\s+([a-zA-Z0-9+#.]+)',
    r'\b(?!.*(?:\d{1,2}|present|current|till)).*?\b([a-zA-Z0-9+#.]+)', # Skills without any dates/projects (very weak match)
]

def clean_negated_content(text):
    """
    Identify and remove sentences that express a lack of experience.
    
    Example: 
    - Input: "I have worked with Python. No experience in Java."
    - Output: "I have worked with Python. [NEGATION_REMOVED]"
    """
    if not text:
        return ""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    
    for sent in sentences:
        is_negated = False
        sent_lower = sent.lower()
        
        # Check classic negation triggers
        triggers = ['no experience', 'lack of', 'limited knowledge', 'not proficient', 'never worked']
        if any(trigger in sent_lower for trigger in triggers):
            is_negated = True
            logger.info(f"Syntactic Negation detected: {sent}")
        
        if not is_negated:
            cleaned.append(sent)
        else:
            cleaned.append("[NEGATION_REMOVED]")
            
    return ' '.join(cleaned)

def flag_distractor_content(text, jd_category):
    """
    Identify if text belongs to a conflicting engineering domain.
    
    Example:
    - JD Category: 'Backend Development'
    - Text: 'Machine Learning, Neural Networks, TensorFlow'
    - Returns: True (this is distractor content)
    """
    if not jd_category or jd_category == 'General':
        return False
        
    text_lower = text.lower()
    
    # Domain Discrepancy Map
    DISTRACTOR_KEYWORDS = {
        'Backend Development': ['machine learning', 'neural network', 'bpmn', 'camunda', 'tensorflow', 'pytorch'],
        'Data Science': ['frontend', 'ui/ux', 'css', 'html', 'react', 'blockchain'],
    }
    
    distractors = DISTRACTOR_KEYWORDS.get(jd_category, [])
    match_count = sum(1 for d in distractors if d in text_lower)
    
    # If a short paragraph mentions more than 2 distractor keywords, it's a distractor
    if match_count >= 2:
        return True
        
    return False
