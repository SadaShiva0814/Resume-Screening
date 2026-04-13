"""
Resume Section Parser — NLP-Powered Section Extraction

This is the SEMANTIC core of the resume understanding engine.
Instead of treating a resume as a flat blob of text, we intelligently
segment it into meaningful sections using NLP.

Approach:
1. spaCy for sentence tokenization and Named Entity Recognition
2. Pattern-based section header detection (regex + NLP heuristics)
3. Contextual section classification using semantic cues
4. Structured entity extraction (name, email, phone, skills, etc.)

This is what makes our system DIFFERENT from keyword-based ATS:
- We UNDERSTAND which part of the resume is talking about what
- We can weight sections differently based on what matters for the role
"""

import re
import hashlib
import logging
import spacy
from datetime import datetime
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Load spaCy model (singleton)
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            raise
    return _nlp


# =====================================================
# Section Header Patterns — comprehensive detection
# =====================================================

SECTION_PATTERNS = {
    'summary': [
        r'(?:professional\s+)?summary',
        r'(?:career\s+)?objective',
        r'about\s+me',
        r'profile\s+(?:summary|overview)',
        r'executive\s+summary',
        r'personal\s+statement',
        r'career\s+overview',
        r'professional\s+profile',
    ],
    'experience': [
        r'(?:work|professional|employment)\s+(?:experience|history)',
        r'experience',
        r'work\s+history',
        r'career\s+history',
        r'professional\s+background',
        r'employment',
        r'positions?\s+held',
        r'relevant\s+experience',
    ],
    'education': [
        r'education(?:al)?\s*(?:background|qualification|history)?',
        r'academic\s+(?:background|qualification|record)',
        r'qualifications?',
        r'degrees?',
        r'certifications?\s*(?:and|&)?\s*education',
    ],
    'skills': [
        r'(?:technical\s+)?skills?',
        r'(?:core\s+)?competenc(?:ies|e)',
        r'technical\s+(?:proficiency|expertise|skills)',
        r'(?:areas?\s+of\s+)?expertise',
        r'technologies',
        r'tools?\s*(?:and|&)?\s*technologies',
        r'programming\s+languages?',
        r'(?:software|technical)\s+knowledge',
        r'(?:key\s+)?strengths',
    ],
    'projects': [
        r'projects?',
        r'(?:key|notable|significant|academic|personal)\s+projects?',
        r'project\s+(?:experience|work|details)',
        r'portfolio',
    ],
    'certifications': [
        r'certifications?',
        r'licenses?\s*(?:and|&)?\s*certifications?',
        r'professional\s+certifications?',
        r'credentials?',
    ],
    'awards': [
        r'awards?\s*(?:and|&)?\s*(?:honors?|achievements?)?',
        r'honors?\s*(?:and|&)?\s*awards?',
        r'achievements?',
        r'accomplishments?',
        r'recognition',
    ],
    'publications': [
        r'publications?',
        r'research\s+(?:papers?|publications?|work)',
        r'papers?\s+published',
    ],
    'languages': [
        r'languages?\s*(?:known|spoken|proficiency)?',
    ],
    'interests': [
        r'(?:hobbies?\s*(?:and|&)?\s*)?interests?',
        r'hobbies?',
        r'extracurricular\s+activities?',
        r'activities?',
    ],
    'references': [
        r'references?',
    ],
}

# Compile patterns for performance
_COMPILED_PATTERNS = {}
for section, patterns in SECTION_PATTERNS.items():
    combined = '|'.join(f'(?:{p})' for p in patterns)
    _COMPILED_PATTERNS[section] = re.compile(
        r'^\s*(?:[-•*=_|#]+\s*)?(' + combined + r')[\s:.\-_|]*$',
        re.IGNORECASE | re.MULTILINE
    )


# =====================================================
# Contact Information Extraction
# =====================================================

EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PHONE_PATTERN = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}')
LINKEDIN_PATTERN = re.compile(r'(?:linkedin\.com/in/|linkedin:\s*)([a-zA-Z0-9_-]+)', re.IGNORECASE)
GITHUB_PATTERN = re.compile(r'(?:github\.com/|github:\s*)([a-zA-Z0-9_-]+)', re.IGNORECASE)


def parse_resume(text, html_header=None):
    """
    Parse a resume text into structured sections.
    """
    # Safety check for non-string input
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    if not text or len(text.strip()) < 20:
        return _empty_result()
    
    nlp = _get_nlp()
    
    # Extract contact info first (usually at the top)
    contact = _extract_contact_info(text, nlp, html_header=html_header)
    
    # Segment into sections
    sections = _segment_sections(text)
    
    # If section detection failed, try contextual classification
    if _is_poorly_segmented(sections):
        sections = _contextual_segment(text, nlp)
    
    # Extract individual skills from the skills section
    extracted_skills = _extract_skills(sections.get('skills', ''), text, nlp)
    
    # Extract years of experience (passed sections for masking)
    years_exp = _estimate_years_experience(sections.get('experience', ''), text, sections)
    
    # Detect education level
    edu_level = _detect_education_level(sections.get('education', ''), text)
    
    # Create formatted display versions of sections (for UI display)
    display_sections = {}
    for section_name, section_text in sections.items():
        if section_text and section_text.strip():
            display_sections[section_name] = _format_section_text(section_text, section_name)
        else:
            display_sections[section_name] = ''
    
    return {
        'contact': contact,
        'sections': sections,              # Raw text for ML scoring
        'display_sections': display_sections,  # Formatted text for UI
        'extracted_skills': extracted_skills,
        'years_of_experience': years_exp,
        'education_level': edu_level,
    }


def _format_section_text(text, section_type='general'):
    """
    Format raw extracted section text into clean, readable paragraphs.
    
    Transforms walls-of-text into structured content by:
    - Breaking at date patterns (experience entries)
    - Breaking at company/role boundaries
    - Preserving bullet points
    - Normalizing whitespace
    """
    if not text or not text.strip():
        return ''
    
    # Step 1: Normalize whitespace (collapse multiple spaces but keep newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Step 2: Insert line breaks at structural boundaries
    
    # Break before date ranges (e.g., "Jan 2020 to Dec 2022", "2018 - 2020")
    text = re.sub(
        r'(?<=[a-z.])\s+(?=(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})',
        '\n\n', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'(?<=[a-z.])\s+(?=\d{4}\s*[-–to]+\s*(?:\d{4}|present|current))',
        '\n\n', text, flags=re.IGNORECASE
    )
    
    # Break before "Company Name" patterns
    text = re.sub(r'\s+(?=Company Name)', '\n', text)
    
    # Break before common section sub-headers
    text = re.sub(
        r'\s+(?=(?:Responsibilities|Duties|Achievements|Key Skills|Education|Certifications?)\s*:)',
        '\n\n', text, flags=re.IGNORECASE
    )
    
    # Break before bullet-like patterns at the start of items
    text = re.sub(r'\s+(?=[•·▪■-]\s)', '\n', text)
    
    # For skills sections: format comma-separated lists nicely
    if section_type == 'skills':
        # If it looks like a long comma-separated list, keep it formatted
        if text.count(',') > 5:
            items = [item.strip() for item in text.split(',') if item.strip()]
            # Group into rows of ~5 items
            rows = []
            for i in range(0, len(items), 5):
                rows.append(', '.join(items[i:i+5]))
            text = '\n'.join(rows)
    
    # Step 3: Clean up multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Step 4: Clean individual lines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != '':
            cleaned_lines.append('')  # Keep one blank line for paragraph breaks
    
    # Step 5: Truncate very long sections (keep first 1500 chars)
    result = '\n'.join(cleaned_lines).strip()
    if len(result) > 1500:
        result = result[:1500].rsplit('\n', 1)[0] + '\n...'
    
    return result


def _empty_result():
    """Return empty result structure."""
    return {
        'contact': {'name': '', 'email': '', 'phone': '', 'linkedin': '', 'github': ''},
        'sections': {
            'summary': '', 'skills': '', 'experience': '',
            'education': '', 'projects': ''
        },
        'extracted_skills': [],
        'years_of_experience': 0,
        'education_level': 'unknown',
    }


# =====================================================
# Contact Information Extraction
# =====================================================

def _extract_contact_info(text, nlp, html_header=None):
    """Extract contact details from resume text.
    
    Args:
        text: Full resume text
        nlp: spaCy language model
        html_header: Optional header text extracted from HTML (Kaggle dataset)
    """
    contact = {
        'name': '',
        'email': '',
        'phone': '',
        'linkedin': '',
        'github': ''
    }
    
    # Email
    email_match = EMAIL_PATTERN.search(text)
    if email_match:
        contact['email'] = email_match.group()
    
    # Phone
    phone_match = PHONE_PATTERN.search(text)
    if phone_match:
        contact['phone'] = phone_match.group()
    
    # LinkedIn
    linkedin_match = LINKEDIN_PATTERN.search(text)
    if linkedin_match:
        contact['linkedin'] = linkedin_match.group(1)
    
    # GitHub
    github_match = GITHUB_PATTERN.search(text)
    if github_match:
        contact['github'] = github_match.group(1)
    
    # =========== Name Extraction (Multi-Strategy) ===========
    contact['role'] = ''
    
    # If we have an HTML header (Kaggle dataset), use it directly.
    # The Kaggle dataset is anonymized — Resume_str doesn't contain real names.
    # The SECTION_NAME holds the job title. We set it as the role, and generate a synthetic name so it looks realistic.
    if html_header:
        contact['role'] = _format_job_title_label(html_header)
        contact['name'] = _generate_synthetic_name(html_header)
    else:
        # For real PDF/DOCX uploads: try to find actual person names
        
        # Strategy 1: spaCy NER on the first few lines
        first_lines = '\n'.join(text.split('\n')[:5])
        doc = nlp(first_lines)
        
        def _is_valid_name(n):
            n_lower = n.lower()
            n_words = set(re.findall(r'\w+', n_lower))
            
            # 1. Block common section headers and meta-terms
            invalid_words = {'summary', 'profile', 'prole', 'experience', 'education', 'skills', 'objective', 'resume', 'cv', 'contact', 'personal', 'projects', 'hobby', 'hobbies', 'achievement', 'achievements', 'environment', 'process', 'improvement', 'international', 'strategic', 'planning', 'driven', 'binding', 'view', 'compose', 'jetpack'}
            if any(w in n_words for w in invalid_words):
                return False
                
            # 2. Block address/location patterns
            if ',' in n and len(n.split()) <= 3:
                return False
                
            # 3. Block common location keywords
            locations = {'india', 'usa', 'uk', 'bengaluru', 'bangalore', 'hyderabad', 'pune', 'mumbai', 'delhi', 'chennai', 'agartala', 'telangana', 'karnataka', 'london', 'york', 'california', 'texas', 'belarus', 'minsk', 'lithuania', 'vilnius'}
            if any(loc in n_words for loc in locations):
                return False
                
            # 4. Block digits and symbols (Stricter: block colons, commas, dots, etc.)
            if bool(re.search(r'[\d@#$^*]|=|[():;,.!]', n)):
                return False
                
            # 5. Length check — names are usually 2-3 words
            words = n.split()
            if len(words) < 2 or len(words) > 4:
                return False

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                name_candidate = ent.text.strip()
                # Validate: 2-4 words, no digits, no resume keywords, not a job title
                if (2 <= len(name_candidate.split()) <= 4 and _is_valid_name(name_candidate)):
                    contact['name'] = _format_name(name_candidate)
                    break
        
        # Strategy 2: Priority First Line check (Professional Standard)
        if not contact['name']:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if lines:
                first_line = lines[0]
                # If first line is short and looks like a name, lock it in
                if (2 <= len(first_line.split()) <= 4 and 
                    not EMAIL_PATTERN.search(first_line) and 
                    not PHONE_PATTERN.search(first_line) and
                    _is_valid_name(first_line)):
                    contact['name'] = _format_name(first_line)
        
        # Strategy 3: Priority scan of first 5 lines (Stricter)
        if not contact['name']:
            for line in text.split('\n')[:5]:
                line = line.strip()
                if (line and 
                    2 <= len(line.split()) <= 3 and  # Names are rarely 4 words
                    not EMAIL_PATTERN.search(line) and 
                    not PHONE_PATTERN.search(line) and
                    _is_valid_name(line)):
                    contact['name'] = _format_name(line)
                    break
        
    # Final fallback for Anonymized Resumes (Kaggle/Dataset)
    if not contact['name']:
        # If we really can't find a name, generate one based on the first few words
        # This prevents "Build Tools" etc. from appearing as names
        contact['name'] = _generate_synthetic_name(text[:100])
    
    return contact


# Common job-title keywords — if a line contains these, it's NOT a person's name
_JOB_TITLE_KEYWORDS = {
    'administrator', 'manager', 'director', 'specialist', 'coordinator',
    'analyst', 'engineer', 'developer', 'designer', 'consultant',
    'associate', 'assistant', 'supervisor', 'executive', 'officer',
    'lead', 'senior', 'junior', 'intern', 'trainee', 'architect',
    'scientist', 'researcher', 'technician', 'representative',
    'accountant', 'auditor', 'recruiter', 'advisor', 'planner',
    'chef', 'teacher', 'professor', 'nurse', 'therapist', 'attorney',
    'clerk', 'receptionist', 'secretary', 'operator', 'mechanic',
    'hr', 'it', 'qa', 'waitress', 'bartender', 'driver', 'pilot', 'security', 'guard',
    'student', 'graduate', 'undergraduate', 'postgraduate', 'candidate', 'professional',
    'freelancer', 'contractor', 'consultant', 'full stack', 'backend', 'frontend',
    'jetpack', 'compose', 'android', 'ios', 'mobile', 'flutter', 'react', 'native'
}

_FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"]
_LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"]

def _generate_synthetic_name(seed_text):
    """Generate a consistent synthetic name based on hashing the input text."""
    # Use MD5 hash to get a consistent integer from the text
    hash_obj = hashlib.md5(seed_text.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)
    
    first_idx = hash_int % len(_FIRST_NAMES)
    last_idx = (hash_int // len(_FIRST_NAMES)) % len(_LAST_NAMES)
    
    return f"{_FIRST_NAMES[first_idx]} {_LAST_NAMES[last_idx]}"

def _is_job_title(text):
    """Check if a text string looks like a job title rather than a person's name."""
    words = text.lower().split()
    # Remove common connecting words
    content_words = [w.strip('/-.,()') for w in words if w.strip('/-.,()')]
    for word in content_words:
        if word in _JOB_TITLE_KEYWORDS:
            return True
    # Also check for slashes (e.g., "HR ADMINISTRATOR/MARKETING ASSOCIATE")
    if '/' in text:
        return True
    return False


def _format_name(name):
    """Format a person's name to proper title case."""
    if not name or not isinstance(name, str):
        return str(name) if name is not None else ""
        
    # Handle ALL CAPS
    if name.isupper():
        return name.title()
    return name.strip()


def _format_job_title_label(raw_label):
    """Format a job title from SCREAMING CAPS to clean title case.
    
    E.g. 'HR ADMINISTRATOR/MARKETING ASSOCIATE' → 'HR Administrator'
    """
    if not raw_label or not isinstance(raw_label, str):
        return 'Unnamed Candidate'
    
    # Take the primary title (before slash if present)
    label = raw_label.split('/')[0].strip()
    
    # Clean up extra whitespace
    label = re.sub(r'\s+', ' ', label).strip()
    
    # Title case, but preserve known acronyms
    acronyms = {'HR', 'IT', 'QA', 'UI', 'UX', 'VP', 'CEO', 'CFO', 'CTO', 'COO',
                'AI', 'ML', 'NLP', 'SQL', 'AWS', 'API', 'R&D', 'PR', 'CRM', 'ERP'}
    
    words = label.split()
    formatted = []
    for word in words:
        if word.upper() in acronyms:
            formatted.append(word.upper())
        elif word.isupper() and len(word) > 2:
            formatted.append(word.title())
        else:
            formatted.append(word)
    
    result = ' '.join(formatted)
    
    # Truncate if too long
    if len(result) > 40:
        result = result[:37] + '...'
    
    return result


# =====================================================
# Section Segmentation
# =====================================================

def _segment_sections(text):
    """
    Segment resume text into sections using header detection.
    
    Strategy:
    1. Scan each line for section headers using precompiled regex patterns
    2. Assign content between headers to the detected section
    3. Content before the first header goes to 'summary' or 'header_area'
    """
    lines = text.split('\n')
    sections = OrderedDict()
    current_section = 'summary'  # Default section for text before first header
    current_content = []
    
    for line in lines:
        detected_section = _detect_section_header(line)
        
        if detected_section:
            # Save previous section
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    if current_section in sections:
                        sections[current_section] += '\n' + content
                    else:
                        sections[current_section] = content
            
            current_section = detected_section
            current_content = []
        else:
            current_content.append(line)
    
    # Save the last section
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            if current_section in sections:
                sections[current_section] += '\n' + content
            else:
                sections[current_section] = content
    
    # Ensure all core sections exist (even if empty)
    for core_section in ['summary', 'skills', 'experience', 'education', 'projects']:
        if core_section not in sections:
            sections[core_section] = ''
    
    # Merge related sections
    sections = _merge_related_sections(sections)
    
    return dict(sections)


def _detect_section_header(line):
    """Detect if a line is a section header."""
    line_stripped = line.strip()
    
    # Skip empty lines or very long lines (not headers)
    if not line_stripped or len(line_stripped) > 60:
        return None
    
    # Skip lines that are obviously not headers
    if line_stripped.count(' ') > 8:  # Headers are usually short
        return None
    
    for section, pattern in _COMPILED_PATTERNS.items():
        if pattern.match(line_stripped):
            return section
    
    # Also check without special characters (some resumes use decorators)
    clean_line = re.sub(r'[^a-zA-Z\s]', '', line_stripped).strip()
    if clean_line and len(clean_line) < 40:
        for section, pattern in _COMPILED_PATTERNS.items():
            if pattern.match(clean_line):
                return section
    
    return None


def _merge_related_sections(sections):
    """Merge related subsections into main sections."""
    # Certifications → Education
    if 'certifications' in sections and sections['certifications']:
        edu_content = sections.get('education', '')
        sections['education'] = (edu_content + '\n\nCertifications:\n' + 
                                  sections['certifications']).strip()
        del sections['certifications']
    
    # Awards → Experience (achievements in work context)
    if 'awards' in sections and sections['awards']:
        exp_content = sections.get('experience', '')
        sections['experience'] = (exp_content + '\n\nAwards & Achievements:\n' + 
                                   sections['awards']).strip()
        del sections['awards']
    
    # Publications → Projects
    if 'publications' in sections and sections['publications']:
        proj_content = sections.get('projects', '')
        sections['projects'] = (proj_content + '\n\nPublications:\n' + 
                                 sections['publications']).strip()
        del sections['publications']
    
    return sections


def _is_poorly_segmented(sections):
    """Check if section detection was poor (most sections empty)."""
    core_sections = ['skills', 'experience', 'education', 'projects']
    non_empty = sum(1 for s in core_sections if sections.get(s, '').strip())
    return non_empty <= 1


# =====================================================
# Contextual Section Classification (Fallback)
# =====================================================

def _contextual_segment(text, nlp):
    """
    When header detection fails, use NLP to classify paragraphs
    into sections based on their content.
    
    This is the SEMANTIC fallback — it understands what the text
    is ABOUT, not just what headers say.
    """
    paragraphs = _split_paragraphs(text)
    sections = {
        'summary': '', 'skills': '', 'experience': '',
        'education': '', 'projects': ''
    }
    
    # Keywords and patterns for contextual classification
    experience_indicators = [
        r'\b\d{4}\s*[-–]\s*(?:\d{4}|present|current)\b',  # Date ranges
        r'\b(?:worked|working|responsible|managed|developed|led|created|designed)\b',
        r'\b(?:company|corporation|inc\.|ltd\.|llc|organization)\b',
        r'\b(?:senior|junior|lead|manager|engineer|developer|analyst|consultant)\b',
    ]
    
    education_indicators = [
        r'\b(?:university|college|institute|school|academy)\b',
        r'\b(?:bachelor|master|phd|doctorate|diploma|degree|b\.?tech|m\.?tech|b\.?e\b|m\.?e\b|b\.?sc|m\.?sc|b\.?a\b|m\.?a\b|mba|bba)\b',
        r'\b(?:cgpa|gpa|percentage|grade|marks)\b',
        r'\b(?:graduated|graduation|coursework)\b',
    ]
    
    skills_indicators = [
        r'\b(?:python|java|javascript|c\+\+|react|angular|node|sql|aws|docker|kubernetes|git|linux)\b',
        r'\b(?:proficient|experienced|expertise|familiar|knowledge)\b',
        r'(?:,\s*\w+){3,}',  # Comma-separated list (typical skills format)
    ]
    
    project_indicators = [
        r'\b(?:project|built|implemented|developed|created|designed)\b.*(?:using|with|in)\b',
        r'\b(?:github|repository|demo|deployed|hosted)\b',
        r'\bproject\s*(?:title|name|description)?\s*:',
    ]
    
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
        
        para_lower = para.lower()
        
        # Score each section
        scores = {
            'experience': sum(1 for p in experience_indicators if re.search(p, para_lower)),
            'education': sum(1 for p in education_indicators if re.search(p, para_lower)),
            'skills': sum(1 for p in skills_indicators if re.search(p, para_lower)),
            'projects': sum(1 for p in project_indicators if re.search(p, para_lower)),
        }
        
        # First paragraph is likely summary
        if i == 0 and max(scores.values()) < 2:
            sections['summary'] += para + '\n'
            continue
        
        # Assign to highest-scoring section
        best_section = max(scores, key=scores.get)
        if scores[best_section] > 0:
            sections[best_section] += para + '\n'
        else:
            # Can't classify — add to summary
            sections['summary'] += para + '\n'
    
    return {k: v.strip() for k, v in sections.items()}


def _split_paragraphs(text):
    """Split text into meaningful paragraphs."""
    # Split on double newlines or lines that are clearly separators
    paragraphs = re.split(r'\n{2,}|(?:\n\s*[-=_]{3,}\s*\n)', text)
    return [p.strip() for p in paragraphs if p.strip()]


# =====================================================
# Skills Extraction
# =====================================================

# Comprehensive tech skills list
KNOWN_SKILLS = {
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'ruby', 'go', 'golang',
    'rust', 'swift', 'kotlin', 'scala', 'perl', 'php', 'r', 'matlab', 'dart', 'lua',
    'objective-c', 'shell', 'bash', 'powershell', 'sql', 'nosql', 'html', 'css',
    
    # Frameworks & Libraries
    'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vuejs', 'vue.js',
    'django', 'flask', 'fastapi', 'spring', 'spring boot', 'springboot', 'express',
    'expressjs', 'node', 'nodejs', 'node.js', 'next.js', 'nextjs', 'nuxt.js',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
    'matplotlib', 'seaborn', 'opencv', 'nltk', 'spacy', 'transformers', 'huggingface',
    'jquery', 'bootstrap', 'tailwind', 'material-ui', 'redux', 'graphql',
    '.net', 'asp.net', 'entity framework', 'hibernate', 'laravel', 'symfony',
    'flutter', 'react native', 'xamarin', 'electron',
    
    # Databases
    'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch',
    'cassandra', 'dynamodb', 'sqlite', 'oracle', 'sql server', 'mariadb',
    'neo4j', 'couchdb', 'firebase', 'firestore', 'supabase',
    
    # Cloud & DevOps
    'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'heroku',
    'docker', 'kubernetes', 'k8s', 'jenkins', 'terraform', 'ansible',
    'ci/cd', 'github actions', 'gitlab ci', 'circleci', 'travis ci',
    'nginx', 'apache', 'linux', 'unix', 'windows server',
    
    # Data Science & ML
    'machine learning', 'deep learning', 'natural language processing', 'nlp',
    'computer vision', 'data science', 'data analysis', 'data engineering',
    'big data', 'hadoop', 'spark', 'apache spark', 'kafka', 'airflow',
    'data visualization', 'tableau', 'power bi', 'looker',
    'statistics', 'statistical analysis', 'a/b testing',
    'neural networks', 'cnn', 'rnn', 'lstm', 'bert', 'gpt', 'llm',
    'reinforcement learning', 'random forest', 'svm', 'xgboost',
    'feature engineering', 'model deployment', 'mlops',
    
    # Tools & Practices
    'git', 'github', 'gitlab', 'bitbucket', 'svn',
    'jira', 'confluence', 'trello', 'asana', 'slack',
    'agile', 'scrum', 'kanban', 'devops', 'microservices',
    'rest api', 'restful', 'api', 'graphql', 'grpc', 'websocket',
    'unit testing', 'integration testing', 'tdd', 'bdd',
    'jest', 'pytest', 'junit', 'selenium', 'cypress',
    
    # Android/iOS Specifics (Fix for name parsing)
    'view binding', 'data binding', 'jetpack compose', 'swiftui', 'uikit', 'pwa',
    
    # Soft Skills / Domains
    'project management', 'team leadership', 'communication',
    'problem solving', 'critical thinking', 'collaboration',
}


def _extract_skills(skills_section, full_text, nlp):
    """
    Extract individual skills from the resume.
    
    Uses multiple strategies:
    1. Match against known skills database
    2. Extract from comma/pipe-separated lists
    3. NLP-based entity extraction
    """
    detected_skills = set()
    search_text = (skills_section + ' ' + full_text).lower()
    
    # Strategy 1: Match known skills
    for skill in KNOWN_SKILLS:
        # Word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, search_text):
            detected_skills.add(skill.title() if len(skill) > 3 else skill.upper())
    
    # Strategy 2: Extract comma-separated items from skills section
    if skills_section:
        # Find comma or pipe separated lists
        lists = re.findall(r'(?:[\w+#./\s-]+(?:,|\|)[\s]*){2,}[\w+#./\s-]+', skills_section)
        for lst in lists:
            items = re.split(r'[,|]', lst)
            for item in items:
                item = item.strip()
                if 2 <= len(item) <= 40 and not item.isdigit():
                    detected_skills.add(item)
    
    return sorted(list(detected_skills))


# =====================================================
# Experience & Education Estimation
# =====================================================

def _estimate_years_experience(experience_section, full_text, all_sections=None):
    """
    Estimate total years of professional experience with robust date parsing.
    
    Fixed '40 Year' Bug:
    1. Restricts search to experience section only (avoiding header/contact noise)
    2. Uses Interval Merging to prevent double-counting overlapping roles
    3. Filters out implausible dates (pre-1980)
    """
    # Use only experience_section to avoid picking up dates from header/education
    text = experience_section if experience_section and len(experience_section.strip()) > 30 else ""
    if not text:
        return 0
    
    # 1. Look for explicit mentions (e.g., "3+ years of experience", "2+ YOE")
    explicit = re.search(
        r'(\b[1-3]?\d\b(?:\.\d+)?)\+?\s*(?:years?|yrs?|yoe)\s*(?:of\s+)?(?:experience|exp|working)?',
        text, re.IGNORECASE
    )
    if explicit:
        return float(explicit.group(1))
    
    # 2. Date Range Parsing
    # Handles: "Jan 2020 - Dec 2022", "2018-2020", "06/2022 - Present", "Aug 2024 Present", etc.
    month_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2}/)'
    year_pattern = r'(?:\d{4}|\d{2})'
    date_part = f'(?:{month_pattern}\\s*)?{year_pattern}'
    
    # Full range pattern: [Date] [Separator] [Date|Present]
    # Separator can be a dash, 'to', or just whitespace if followed by 'Present'
    range_pattern = re.compile(
        rf'({date_part})\s*(?:[-–—to]+|\s+)\s*({date_part}|present|current|now|till\s+date)',
        re.IGNORECASE
    )
    
    ranges = range_pattern.findall(text)
    
    if not ranges:
        return 0
        
    current_val = datetime.now().year * 12 + datetime.now().month
    intervals = []
    
    def parse_to_val(d_str):
        d_str = d_str.lower().strip()
        if not d_str or any(w in d_str for w in ['present', 'current', 'now', 'till']):
            return current_val
        y_match = re.search(r'(\d{4})', d_str)
        if not y_match:
            y_match = re.search(r'\b(\d{2})\b', d_str)
            if y_match:
                y = 2000 + int(y_match.group(1))
                if y > datetime.now().year + 1: y -= 100
                year = y
            else: return None
        else: year = int(y_match.group(1))
        
        if year < 1980 or year > datetime.now().year + 1: return None
        
        m = 1
        m_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        for name, val in m_map.items():
            if name in d_str: m = val; break
        else:
            n_match = re.search(r'(\d{1,2})/', d_str)
            if n_match: m = int(n_match.group(1))
        return year * 12 + m

    for s_str, e_str in ranges:
        s_v = parse_to_val(s_str)
        e_v = parse_to_val(e_str)
        if s_v and e_v and e_v >= s_v:
            intervals.append([s_v, e_v])
    
    if not intervals: return 0
    
    # Merge overlapping intervals
    intervals.sort()
    merged = [intervals[0]]
    for curr in intervals[1:]:
        if curr[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], curr[1])
        else:
            merged.append(curr)
            
    total_months = sum(e - s for s, e in merged)
                
    years = round(total_months / 12, 1)
    return min(years, 45)


def _detect_education_level(education_section, full_text):
    """Detect the highest education level."""
    text = (education_section + ' ' + full_text).lower()
    
    # Check in order of highest to lowest
    levels = [
        ('phd', [r'\bph\.?d\b', r'\bdoctorate\b', r'\bdoctoral\b']),
        ('masters', [r'\bmaster', r'\bm\.?tech\b', r'\bm\.?s\b', r'\bm\.?sc\b', 
                      r'\bm\.?e\b', r'\bm\.?a\b', r'\bmba\b', r'\bm\.?c\.?a\b',
                      r'\bpost\s*graduat']),
        ('bachelors', [r'\bbachelor', r'\bb\.?tech\b', r'\bb\.?s\b', r'\bb\.?sc\b',
                        r'\bb\.?e\b(?!\w)', r'\bb\.?a\b', r'\bbba\b', r'\bb\.?c\.?a\b',
                        r'\bundergraduat', r'\bb\.?com\b']),
        ('diploma', [r'\bdiploma\b', r'\bassociate\s*degree']),
        ('high_school', [r'\bhigh\s*school\b', r'\b(?:10|12)th\b', r'\bssc\b', r'\bhsc\b',
                          r'\bintermediate\b']),
    ]
    
    for level, patterns in levels:
        for pattern in patterns:
            if re.search(pattern, text):
                return level
    
    return 'unknown'
