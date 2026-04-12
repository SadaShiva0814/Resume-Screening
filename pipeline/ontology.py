"""
Skill Ontology — Semantic Knowledge Graph

Maps technical aliases, acronyms, and related technologies
to standardized entities. This ensures "GCP" and "Google Cloud Platform"
are treated as a 100% semantic match.
"""

import re
import logging

# Alias Mapping: {Alias: Standardized Name}
SKILL_ONTOLOGY = {
    'gcp': 'Google Cloud Platform',
    'google cloud': 'Google Cloud Platform',
    'aws': 'Amazon Web Services',
    'amazon web services': 'Amazon Web Services',
    's3': 'Amazon S3',
    'ec2': 'Amazon EC2',
    'lambda': 'AWS Lambda',
    'k8s': 'Kubernetes',
    'spring': 'Spring Boot',
    'springboot': 'Spring Boot',
    'hibernate': 'Hibernate ORM',
    'postgres': 'PostgreSQL',
    'node': 'Node.js',
    'nodejs': 'Node.js',
    'react': 'React.js',
    'reactjs': 'React.js',
    'vue': 'Vue.js',
    'vuejs': 'Vue.js',
    'ml': 'Machine Learning',
    'nlp': 'Natural Language Processing',
}

# Domain Stack Groups
DOMAIN_KNOWLEDGE = {
    'Backend Development': [
        'Java', 'Spring Boot', 'Node.js', 'Python', 'Go', 'Microservices', 
        'REST API', 'SQL', 'NoSQL', 'PostgreSQL', 'Redis', 'Docker'
    ],
    'Cloud Engineering': [
        'Amazon Web Services', 'Google Cloud Platform', 'Azure', 
        'Kubernetes', 'Terraform', 'Docker', 'CI/CD'
    ],
    'Data Science': [
        'Python', 'Machine Learning', 'TensorFlow', 'PyTorch', 
        'Pandas', 'Spark', 'Big Data', 'Statistics'
    ]
}

# Stack Anchor Languages (Highest Priority)
STACK_ANCHORS = ['java', 'python', 'javascript', 'typescript', 'c++', 'c#', 'go', 'php', 'ruby', 'kotlin', 'scala']

def standardize_skill(skill):
    """Normalize a skill name using the ontology."""
    if not skill: return ""
    s = skill.lower().strip()
    return SKILL_ONTOLOGY.get(s, skill)

def identify_jd_anchor(jd_text):
    """
    Identify the main core language of the JD based on frequency and context.
    Returns the lowercase anchor name.
    """
    if not jd_text: return 'general'
    
    jd_lower = jd_text.lower()
    counts = {}
    for lang in STACK_ANCHORS:
        # Check for word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(lang) + r'\b'
        matches = len(re.findall(pattern, jd_lower))
        if matches > 0:
            counts[lang] = matches
            
    if not counts:
        return 'general'
        
    # Pick the most frequently mentioned anchor
    return max(counts, key=counts.get)

def get_related_skills(standardized_skill):
    """Return a list of skills often associated with this one."""
    # This can be expanded into a real graph later
    s = standardized_skill.lower()
    if 'spring' in s or 'java' in s:
        return ['Java', 'Spring Boot', 'Hibernate', 'Microservices', 'Maven']
    if 'python' in s:
        return ['Python', 'Django', 'Flask', 'FastAPI', 'Pandas', 'Numpy']
    if 'aws' in s or 'amazon web services' in s:
        return ['EC2', 'S3', 'Lambda', 'AWS', 'Cloud Engineering']
    return []
