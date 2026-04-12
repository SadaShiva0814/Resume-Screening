import sys
import os
import re
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from pipeline.ontology import standardize_skill, identify_jd_anchor
from pipeline.semantic_hardening import clean_negated_content

def test_signal_density():
    print("\n--- Testing Signal Density Weighting ---")
    jd_skills = ['Java', 'Spring Boot', 'AWS', 'Microservices']
    
    # Focused Para (3/4 matches)
    focused_para = "We built Java microservices using Spring Boot and deployed on AWS."
    # Diluted Para (1/4 matches)
    diluted_para = "Experienced in Python and Django. Also has some Spring Boot exposure."
    
    # Logic from semantic_scorer.py
    def get_density_factor(para, jd_skills):
        p_lower = para.lower()
        p_matches = sum(1 for s in jd_skills if s.lower() in p_lower)
        return 0.8 + min(0.6, (p_matches / len(jd_skills)) * 2.0)

    f_factor = get_density_factor(focused_para, jd_skills)
    d_factor = get_density_factor(diluted_para, jd_skills)
    
    print(f"Focused Para Density Factor: {f_factor:.2f} (Expected > 1.0)")
    print(f"Diluted Para Density Factor: {d_factor:.2f} (Expected ~0.9)")
    
    if f_factor > d_factor:
        print("✅ SUCCESS: Focused paragraph gets higher density weight.")
    else:
        print("❌ FAILURE: Density weighting failed.")

def test_stack_mismatch():
    print("\n--- Testing Stack-Mismatch Guard ---")
    jd_text = "Experienced Java Developer"
    anchor = identify_jd_anchor(jd_text)
    print(f"Detected Anchor: {anchor}")
    
    # Candidate skills
    python_skills = ['Python', 'Django', 'FastAPI']
    java_skills = ['Java', 'Spring', 'AWS']
    
    has_java = any(anchor in s.lower() for s in java_skills)
    has_python = any(anchor in s.lower() for s in python_skills)
    
    print(f"Java Candidate has anchor: {has_java}")
    print(f"Python Candidate has anchor: {has_python}")
    
    if has_java and not has_python:
        print("✅ SUCCESS: Stack Guard correctly identifies anchor presence.")
    else:
        print("❌ FAILURE: Stack Guard logic error.")

def test_negation_cleaning():
    print("\n--- Testing Negation Cleaning ---")
    text = "I have experience with Python. No experience in Java."
    cleaned = clean_negated_content(text)
    print(f"Cleaned: {cleaned}")
    if "Java" not in cleaned:
        print("✅ SUCCESS: Negations removed.")
    else:
        print("❌ FAILURE: Negation cleaning failed.")

if __name__ == "__main__":
    test_signal_density()
    test_stack_mismatch()
    test_negation_cleaning()
