import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.section_parser import _estimate_years_experience

def test_experience():
    test_cases = [
        ("Professional Experience: June 2022 - Present", 3.8), # Approx depending on current date (April 2026)
        ("Work History: 2018-2020, 2021 to 2023", 4.0),
        ("Experience: 06/2022 - 08/2024", 2.2),
        ("I have 5 years of experience as a developer", 5.0),
        ("Software Development Engineer (Backend) Aug 2024 Present", 1.7), # Approx up to April 2026
        ("Backend Engineer (2+ YOE) specializing in...", 2.0),
        ("Internship: Jan 2024 - May 2024", 0.3),
        ("No dates here", 0)
    ]
    
    # Current date for testing is April 2026 as per base.html CURSOR context if relevant, 
    # but the parser uses datetime.now(). At this moment (real time 2026-04-10).
    
    print(f"{'Text Content':<50} | {'Expected':<10} | {'Result':<10}")
    print("-" * 75)
    
    for text, expected in test_cases:
        result = _estimate_years_experience('', text)
        print(f"{text[:48]:<50} | {expected:<10} | {result:<10}")

if __name__ == "__main__":
    test_experience()
