import sys
import os
sys.path.append(os.getcwd())

from pipeline.ranker import screen_resumes
from config import Config

# Sample JD based on user's screenshot
JD = """
We are looking for a highly skilled Software Development Engineer to join our backend engineering team. 
You will be responsible for designing, building, and optimizing scalable microservices and robust APIs that power our core platforms. 
The ideal candidate has a profound understanding of backend architecture, Node.js, TypeScript, Express, PostgreSQL, Redis, Docker, and microservices. 
Preferred qualification: top-tier competitive programming achievements.
"""

# Mock resume for Tafheem Ahemad (High Match, 0 years)
RESUME_TAFHEEM = """
TAFHEEM AHEMAD
Software Engineer | Backend Developer
Email: tafheem@example.com

SUMMARY:
Highly skilled backend engineer with a focus on scalable microservices. Pro-level competitive programmer.

TECHNICAL SKILLS:
Node.js, TypeScript, Express, PostgreSQL, Redis, Docker, Microservices, Kubernetes, Git, Java, Python.

PROJECTS:
- Scalable API Gateway: Built with Node.js and Express, handles 10k requests/sec.
- Distributed Caching Layer: Integrated Redis with Node.js microservices.
- competitive programming: Top 1% on Codeforces and LeetCode.

EDUCATION:
B.Tech in Computer Science, 2024.
"""

# Mock resume for a Low Match
RESUME_LOW = """
BAVIREDDI JAYAPRAKASH
Android Developer
Email: jay@example.com

SUMMARY:
Experienced Android App Developer with 40 years of experience in IoT and basic frontend tools.

SKILLS:
Android Studio, Kotlin, Java, IoT, HTML, CSS, XML.

EXPERIENCE:
40 years of experience in various hardware and mobile projects.
"""

def test_scoring():
    resumes = [
        {'text': RESUME_TAFHEEM, 'file_name': 'Tafheem_Ahemad.txt'},
        {'text': RESUME_LOW, 'file_name': 'Bavireddi_Jayaprakash.txt'}
    ]
    
    # Test 1: Explicit 100 total
    print("Test 1: Explicit 50/35/15 weights")
    weights1 = {'skills': 50, 'experience': 35, 'projects': 15}
    results1 = screen_resumes(resumes, JD, weights=weights1)
    print(f"Tafheem Score: {results1[0]['overall_score'] * 100:.1f}%")
    
    # Test 2: Automatic Remainder (70/20 -> Projects becomes 10)
    print("\nTest 2: Automatic Remainder (70 Skills, 20 Experience)")
    weights2 = {'skills': 70, 'experience': 20}
    results2 = screen_resumes(resumes, JD, weights=weights2)
    print(f"Tafheem Score: {results2[0]['overall_score'] * 100:.1f}%")
    
    # Test 3: Raw Normalization (80/50/5 -> sums to 135)
    print("\nTest 3: Raw Normalization (80/50/5)")
    weights3 = {'skills': 80, 'experience': 50, 'projects': 5}
    results3 = screen_resumes(resumes, JD, weights=weights3)
    print(f"Tafheem Score: {results3[0]['overall_score'] * 100:.1f}%")

if __name__ == "__main__":
    test_scoring()
