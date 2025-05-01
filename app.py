import PyPDF2
import streamlit as st
import os                  
import re                 
import json               
from collections import Counter  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import plotly.express as px

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[\s\n]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()

def tokenize_text(text):
    text = clean_text(text)
    words = re.findall(r'\b[a-z0-9]+\b', text)
    return [w for w in words if w and len(w) > 1]

def split_into_sentences(text):
    text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1\n\2', text)
    sentences = [s.strip() for s in text.split('\n')]
    return [s for s in sentences if s]

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
    "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
    'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then'
}

DOMAIN_CATEGORIES = {
    'Programming Languages': {
        'weight': 2.0,
        'skills': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'swift',
            'kotlin', 'scala', 'php', 'ruby', 'perl', 'r', 'matlab', 'assembly', 'shell',
            'bash', 'powershell', 'sql', 'dart', 'groovy', 'lua', 'objective-c', 'haskell',
            'erlang', 'clojure', 'f#', 'cobol', 'fortran', 'pascal', 'prolog'
        ],
        'context_boosters': [
            'developed', 'implemented', 'programmed', 'coded', 'built', 'created', 'authored',
            'maintained', 'debugged', 'optimized', 'refactored', 'enhanced', 'architected'
        ]
    },
    'Software Development': {
        'weight': 1.9,
        'skills': [
            'object-oriented', 'functional programming', 'design patterns', 'algorithms', 
            'data structures', 'code review', 'unit testing', 'integration testing',
            'test-driven development', 'pair programming', 'agile', 'scrum', 'kanban',
            'version control', 'git', 'svn', 'mercurial', 'code quality', 'clean code',
            'refactoring', 'debugging', 'performance optimization', 'documentation',
            'technical writing', 'api design', 'rest', 'graphql', 'grpc', 'microservices',
            'monolithic architecture', 'serverless', 'containerization', 'docker',
            'kubernetes', 'ci/cd', 'jenkins', 'github actions', 'gitlab ci', 'circleci'
        ],
        'context_boosters': [
            'developed', 'designed', 'implemented', 'architected', 'engineered', 'optimized',
            'refactored', 'debugged', 'tested', 'documented', 'reviewed'
        ]
    },
    'Data Science & AI': {
        'weight': 1.8,
        'skills': [
            'machine learning', 'deep learning', 'neural networks', 'natural language processing',
            'computer vision', 'reinforcement learning', 'data mining', 'data analysis',
            'data visualization', 'statistics', 'probability', 'linear algebra', 'calculus',
            'pandas', 'numpy', 'scipy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            'opencv', 'nltk', 'spacy', 'matplotlib', 'seaborn', 'plotly', 'tableau',
            'power bi', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'elasticsearch', 'hadoop', 'spark', 'kafka', 'airflow', 'etl', 'data pipeline',
            'data warehouse', 'data lake', 'big data', 'feature engineering', 'model evaluation'
        ],
        'context_boosters': [
            'analyzed', 'modeled', 'trained', 'evaluated', 'optimized', 'implemented',
            'deployed', 'researched', 'published', 'presented', 'visualized'
        ]
    },
    'Systems & Security': {
        'weight': 1.7,
        'skills': [
            'operating systems', 'linux', 'windows', 'macos', 'networking', 'tcp/ip',
            'http', 'https', 'dns', 'load balancing', 'distributed systems', 'parallel computing',
            'concurrency', 'multithreading', 'memory management', 'compilers', 'interpreters',
            'virtual machines', 'embedded systems', 'iot', 'cybersecurity', 'encryption',
            'authentication', 'authorization', 'oauth', 'jwt', 'penetration testing',
            'vulnerability assessment', 'firewalls', 'ids/ips', 'siem', 'secure coding',
            'owasp', 'pki', 'ssl/tls', 'vpn', 'wireshark', 'nmap', 'metasploit'
        ],
        'context_boosters': [
            'secured', 'protected', 'hardened', 'implemented', 'configured', 'monitored',
            'analyzed', 'tested', 'audited', 'remediated', 'enforced'
        ]
    },
    'Web & Mobile': {
        'weight': 1.6,
        'skills': [
            'html', 'css', 'javascript', 'typescript', 'react', 'angular', 'vue', 'svelte',
            'next.js', 'nuxt.js', 'gatsby', 'webpack', 'babel', 'vite', 'redux', 'mobx',
            'graphql', 'rest', 'websocket', 'pwa', 'spa', 'ssr', 'node.js', 'express',
            'nest.js', 'django', 'flask', 'fastapi', 'spring', 'laravel', 'rails',
            'asp.net', 'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin',
            'objective-c', 'xamarin', 'cordova', 'ionic', 'responsive design', 'ux/ui',
            'accessibility', 'seo', 'performance optimization', 'caching', 'cdn'
        ],
        'context_boosters': [
            'developed', 'built', 'designed', 'implemented', 'created', 'maintained',
            'optimized', 'deployed', 'scaled', 'architected', 'engineered'
        ]
    }
}

def extract_skills_and_keywords(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    skills_data = {category: {'skills': {}, 'context_count': 0} 
                  for category in DOMAIN_CATEGORIES}
    
    for category, data in DOMAIN_CATEGORIES.items():
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, data['skills'])) + r')\b')
        
        for match in pattern.finditer(text):
            skill = match.group()
            context = text[max(0, match.start()-100):min(len(text), match.end()+100)]
            
            base_score = 0.7
            if any(boost in context for boost in data['context_boosters']):
                base_score = min(1.0, base_score + 0.3)
            
            if skill in skills_data[category]['skills']:
                base_score = min(1.0, skills_data[category]['skills'][skill] + 0.15)
            
            skills_data[category]['skills'][skill] = base_score
            skills_data[category]['context_count'] += 1
    
    keywords = [word for word in re.findall(r'\b\w{4,}\b', text) 
               if word not in STOP_WORDS and not word.isdigit()]
    
    return keywords, {cat: data['skills'] for cat, data in skills_data.items()}

def calculate_match_percentage(resume_text, jd_text):
    try:
        resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
        
        category_scores = {}
        
        for category in DOMAIN_CATEGORIES:
            jd_skills = jd_categories.get(category, {})
            resume_skills = resume_categories.get(category, {})
            
            if not jd_skills:
                continue
                
            matched_score = sum(
                min(jd_skills[skill], resume_skills.get(skill, 0)) 
                for skill in jd_skills
            )
            
            total_score = sum(jd_skills.values())
            
            if total_score > 0:
                match_percent = (matched_score / total_score) * 100
                category_scores[category] = max(1, round(match_percent))
            
        if not category_scores:
            return {'score': 0, 'category_scores': {}}
            
        overall_score = sum(
            score * DOMAIN_CATEGORIES[category]['weight'] 
            for category, score in category_scores.items()
        ) / sum(
            DOMAIN_CATEGORIES[category]['weight'] 
            for category in category_scores
        )
        
        return {
            'score': round(overall_score, 1),
            'category_scores': category_scores
        }
        
    except Exception as e:
        print(f"Error in calculate_match_percentage: {str(e)}")
        return {'score': 0, 'category_scores': {}}

def extract_text_from_pdf(uploaded_file):
    try:
        # Try different PDF extraction methods for better results
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        
        # First pass: extract text with standard method
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Normalize whitespace but preserve paragraph breaks
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'\. ', '.\n', page_text)  # Add line breaks after periods
                text += page_text + '\n\n'
        
        # Enhance section headers for better detection
        text = re.sub(r'(?i)\b(education|academic|qualification)s?\b', '\nEDUCATION\n', text)
        text = re.sub(r'(?i)\b(experience|work history|employment|professional)\b', '\nEXPERIENCE\n', text)
        text = re.sub(r'(?i)\b(projects?|technical projects?)\b', '\nPROJECTS\n', text)
        text = re.sub(r'(?i)\b(achievements?|accomplishments?|awards?)\b', '\nACHIEVEMENTS\n', text)
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_profile_summary(text):
    """Extract a comprehensive profile summary from resume text"""
    text_lower = text.lower()
    sentences = text.split('.')
    
    # Look for explicit summary section
    summary_keywords = ['summary', 'profile', 'objective', 'about me', 'professional background']
    summary_section = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in summary_keywords):
            # Get the next few sentences for context
            start_idx = sentences.index(sentence)
            summary_section = sentences[start_idx:start_idx + 3]
            break
    
    # If no explicit summary, build one from key information
    if not summary_section:
        # Extract years of experience
        exp_match = re.search(r'(\d+)(?:\+)?\s*years?(?:\s+of)?\s+experience', text_lower)
        years_exp = exp_match.group(1) if exp_match else None
        
        # Extract current/most recent role
        role_keywords = ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'architect']
        role = None
        for keyword in role_keywords:
            role_match = re.search(rf'(?:senior\s+)?{keyword}[\w\s]*', text_lower)
            if role_match:
                role = role_match.group(0).strip().title()
                break
        
        # Extract key skills (top 3)
        skills = []
        skill_keywords = ['python', 'java', 'javascript', 'react', 'node', 'aws', 'cloud', 'full stack', 'backend', 'frontend']
        for skill in skill_keywords:
            if skill in text_lower:
                skills.append(skill.title())
                if len(skills) == 3:
                    break
        
        # Build summary
        summary_parts = []
        if years_exp:
            summary_parts.append(f"{years_exp}+ years of experience")
        if role:
            summary_parts.append(f"as {role}")
        if skills:
            summary_parts.append(f"specializing in {', '.join(skills)}")
        
        summary_section = [' '.join(summary_parts)]
    
    return ' '.join(summary_section).strip()

def analyze_education(text):
    """Enhanced education analysis with better structure and detail capture"""
    # First look for EDUCATION section header (normalized by extract_text_from_pdf)
    edu_pattern = re.compile(r'EDUCATION\s*\n(.*?)(?:\n\n|\n[A-Z]{3,}|$)', re.DOTALL | re.IGNORECASE)
    edu_match = edu_pattern.search(text)
    
    if edu_match:
        # Found explicit education section
        edu_text = edu_match.group(1).strip()
        if edu_text:
            # Clean up the extracted text
            edu_text = re.sub(r'\n{3,}', '\n\n', edu_text)  # Normalize multiple newlines
            return edu_text
    
    # Fallback: search by degree keywords
    degree_keywords = ['bachelor', 'master', 'phd', 'b.tech', 'b.e', 'm.tech', 'bsc', 'msc', 
                      'b.s', 'm.s', 'b.a', 'm.a', 'degree', 'diploma']
    
    lines = text.split('\n')
    education_entries = []
    
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        # Look for degree indicators
        if any(keyword in line_lower for keyword in degree_keywords) or re.search(r'\b(20\d\d|\d{4})\b.*degree', line_lower):
            # Include context (university name, dates, etc.)
            context_start = max(0, i-1)
            context_end = min(len(lines), i+5)  # Get a few lines after for context
            degree_entry = '\n'.join([l.strip() for l in lines[context_start:context_end] if l.strip()])
            education_entries.append(degree_entry)
    
    # If we found entries, join them
    if education_entries:
        return '\n\n'.join(education_entries)
    
    # Last resort: look for university names
    university_keywords = ['university', 'college', 'institute', 'school']
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        if any(keyword in line_lower for keyword in university_keywords):
            context_start = max(0, i-1)
            context_end = min(len(lines), i+3)
            edu_entry = '\n'.join([l.strip() for l in lines[context_start:context_end] if l.strip()])
            education_entries.append(edu_entry)
    
    if education_entries:
        return '\n\n'.join(education_entries)
    
    return "No education details found"

def analyze_experience(text):
    """Enhanced experience analysis with better structure and detail capture"""
    # First look for EXPERIENCE section header (normalized by extract_text_from_pdf)
    exp_pattern = re.compile(r'EXPERIENCE\s*\n(.*?)(?:\n\n|\n[A-Z]{3,}|$)', re.DOTALL | re.IGNORECASE)
    exp_match = exp_pattern.search(text)
    
    if exp_match:
        # Found explicit experience section
        exp_text = exp_match.group(1).strip()
        if exp_text:
            # Clean up the extracted text
            exp_text = re.sub(r'\n{3,}', '\n\n', exp_text)  # Normalize multiple newlines
            # Split into entries if there are clear separations
            entries = re.split(r'(?:\n\n+|\n(?=\d{4}\s*-|\d{4}\s*to|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))', exp_text)
            return '\n\n'.join([entry.strip() for entry in entries if entry.strip()])
    
    # Fallback: search by role keywords and date patterns
    role_keywords = ['engineer', 'developer', 'programmer', 'manager', 'analyst', 'consultant', 
                    'architect', 'intern', 'scientist', 'designer', 'administrator', 'specialist']
    
    lines = text.split('\n')
    experience_entries = []
    
    # First pass: look for job titles and company names
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        
        # Look for job titles
        if any(keyword in line_lower for keyword in role_keywords):
            # Get context around this line
            context_start = max(0, i-1)
            context_end = min(len(lines), i+8)  # Get more lines for experience details
            
            # Look for date patterns in nearby lines
            has_date = False
            for j in range(context_start, context_end):
                if j < len(lines) and re.search(r'(20\d\d|19\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present)', lines[j].lower()):
                    has_date = True
                    break
            
            if has_date:
                exp_entry = '\n'.join([l.strip() for l in lines[context_start:context_end] if l.strip()])
                experience_entries.append(exp_entry)
    
    # Second pass: look for date patterns if we didn't find enough entries
    if len(experience_entries) < 2:
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            # Look for date patterns (2020-2022, 2019 to Present, etc.)
            if re.search(r'(20\d\d|19\d\d)\s*[-–—to]\s*(20\d\d|19\d\d|present|current|now)', line_lower) and not any(entry.lower().find(line_lower) >= 0 for entry in experience_entries):
                context_start = max(0, i-2)  # Include potential title line
                context_end = min(len(lines), i+6)  # Include responsibilities
                exp_entry = '\n'.join([l.strip() for l in lines[context_start:context_end] if l.strip()])
                experience_entries.append(exp_entry)
    
    if experience_entries:
        return '\n\n'.join(experience_entries)
    
    return "No experience details found"

def analyze_projects(text):
    projects = []
    
    # Split text into lines and look for project-related content
    lines = text.split('\n')
    is_project_section = False
    current_project = []
    
    # Keywords that indicate project information
    project_keywords = ['project', 'developed', 'implemented', 'created', 'built']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a project section
        if any(keyword.lower() in line.lower() for keyword in ['projects', 'technical projects']):
            is_project_section = True
            continue
            
        # If we're in project section, collect the information
        if is_project_section:
            # Check if we've reached the end of project section
            if any(keyword.lower() in line.lower() for keyword in ['education', 'experience', 'skills', 'achievements']):
                is_project_section = False
                if current_project:
                    projects.append(' '.join(current_project))
                break
                
            # Start a new project
            if any(keyword.lower() in line.lower() for keyword in project_keywords):
                if current_project:
                    projects.append(' '.join(current_project))
                current_project = [line]
            elif current_project:
                current_project.append(line)
    
    # Add the last project if any
    if current_project:
        projects.append(' '.join(current_project))
    
    # If no structured project section found, try to extract from whole text
    if not projects:
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in project_keywords):
                projects.append(line)
    
    return projects if projects else []

def analyze_achievements(text):
    achievements = []
    
    # Split text into lines and look for achievement-related content
    lines = text.split('\n')
    is_achievement_section = False
    current_achievement = []
    
    # Keywords that indicate achievements
    achievement_keywords = [
        'achieved', 'awarded', 'won', 'recognized', 'selected', 'ranked',
        'improved', 'increased', 'decreased', 'reduced', 'saved', 'delivered',
        'led', 'managed', 'honor', 'certificate', 'certification'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts an achievements section
        if any(keyword.lower() in line.lower() for keyword in ['achievements', 'accomplishments', 'honors']):
            is_achievement_section = True
            continue
            
        # If we're in achievements section, collect the information
        if is_achievement_section:
            # Check if we've reached the end of achievements section
            if any(keyword.lower() in line.lower() for keyword in ['education', 'experience', 'skills', 'projects']):
                is_achievement_section = False
                if current_achievement:
                    achievements.append(' '.join(current_achievement))
                break
                
            # Start a new achievement
            if any(keyword.lower() in line.lower() for keyword in achievement_keywords):
                if current_achievement:
                    achievements.append(' '.join(current_achievement))
                current_achievement = [line]
            elif current_achievement:
                current_achievement.append(line)
    
    # Add the last achievement if any
    if current_achievement:
        achievements.append(' '.join(current_achievement))
    
    # If no structured achievements section found, try to extract from whole text
    if not achievements:
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in achievement_keywords):
                achievements.append(line)
    
    return achievements if achievements else []

def extract_resume_sections(text):
    """Extract resume sections with improved structure and detail"""
    if not text:
        return {
            'Profile Summary': '',
            'Education': '',
            'Experience': '',
            'Projects': [],
            'Achievements': []
        }
    
    # Extract profile summary first
    profile_summary = extract_profile_summary(text)
    
    # Extract other sections
    education = analyze_education(text)
    experience = analyze_experience(text)
    projects = analyze_projects(text)
    achievements = analyze_achievements(text)
    
    return {
        'Profile Summary': profile_summary,
        'Education': education,
        'Experience': experience,
        'Projects': projects,
        'Achievements': achievements
    }

@st.cache_data 
def get_ats_feedback(resume_text, jd_text):
    try:
        # Extract all sections from resume
        sections = extract_resume_sections(resume_text)
        
        # Calculate match percentage and get match data
        match_data = calculate_match_percentage(resume_text, jd_text)
        
        # Extract keywords and skills - ensure proper structure
        resume_keywords, raw_resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, raw_jd_categories = extract_skills_and_keywords(jd_text)
        
        # Convert categories to proper DataFrame-compatible format
        jd_categories = {}
        resume_categories = {}
        for category in DOMAIN_CATEGORIES:
            jd_categories[category] = {
                skill: score 
                for skill, score in raw_jd_categories.get(category, {}).items()
            }
            resume_categories[category] = {
                skill: score 
                for skill, score in raw_resume_categories.get(category, {}).items()
            }
        
        # Calculate matched and missing keywords
        matched_keywords = list(set(resume_keywords) & set(jd_keywords))
        missing_keywords = list(set(jd_keywords) - set(resume_keywords))
        
        return {
            "JD Match": f"{match_data['score']}%",
            "Profile Summary": sections['Profile Summary'],
            "Education": sections['Education'],
            "Experience": sections['Experience'],
            "Projects": sections['Projects'],
            "Achievements": sections['Achievements'],
            "JD_Categories": jd_categories,
            "Resume_Categories": resume_categories,
            "Category_Matches": match_data['category_scores'],
            "Missing_Keywords": missing_keywords[:10],
            "Matched_Keywords": matched_keywords[:10]
        }
        
    except Exception as e:
        st.error(f"Error in text processing: {str(e)}")
        return None

# Streamlit App Interface
st.set_page_config(page_title="Verq ATS Resume Evaluator", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-color: white;
        }
    </style>
""")
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #1f497d;">Verq ATS Resume Evaluator</h1>
        <p>Upload your resume and job description to get instant feedback</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("##  ATS Resume Evaluator")
st.markdown("Upload your resume and job description to receive a tailored match percentage, keyword analysis, and improvement suggestions.")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        jd_input = st.text_area(" Job Description", height=300, placeholder="Paste the JD here...")

    with col2:
        uploaded_resume = st.file_uploader(" Upload Resume (PDF)", type=["pdf"])

if st.button(" Evaluate"):
    if uploaded_resume and jd_input.strip():
        with st.spinner("Analyzing Resume..."):
            # Extract text and analyze
            resume_text = extract_text_from_pdf(uploaded_resume)
            if not resume_text:
                st.error("Failed to extract text from PDF")
                st.stop()
            
            # Get ATS feedback
            ats_response = get_ats_feedback(resume_text, jd_input)
            if not ats_response:
                st.error("Failed to generate ATS feedback")
                st.stop()
            
            # Display results
            match_pct = float(ats_response['JD Match'].strip('%'))
            color = 'green' if match_pct >= 80 else 'orange' if match_pct >= 60 else 'red'
            
            st.markdown(
                f"<h2 style='color: {color}; text-align: center;'>"
                f"Overall Match: {match_pct:.1f}%</h2>", 
                unsafe_allow_html=True
            )
            
            # Create tabs for organized display
            tab1, tab2, tab3 = st.tabs(["Overview", "Skills Analysis", "Recommendations"])
            
            # Tab 1: Overview with improved structure
            with tab1:
                st.markdown("### Overview")
                
                # Match Score Card
                col1, col2 = st.columns([1,3])
                with col1:
                    st.metric("Match Score", ats_response['JD Match'])
                with col2:
                    score = float(ats_response['JD Match'].rstrip('%'))
                    if score >= 80:
                        st.success("Excellent match! Your resume strongly aligns with the job requirements.")
                    elif score >= 60:
                        st.warning("Good match. Consider adding a few more relevant skills.")
                    else:
                        st.error("Needs improvement. Review the recommendations tab for specific enhancements.")
                
                st.divider()
                
                # Profile Summary
                try:
                    if ats_response.get('Profile Summary') and isinstance(ats_response['Profile Summary'], str) \
                       and ats_response['Profile Summary'].strip() not in ["", "No profile summary found"]:
                        with st.expander("Profile Summary", expanded=True):
                            st.write(ats_response['Profile Summary'])
                    else:
                        st.warning("No profile summary found in resume")
                except Exception as e:
                    st.error(f"Error displaying profile summary: {str(e)}")
                
                # Education
                try:
                    if ats_response.get('Education') and isinstance(ats_response['Education'], str) \
                       and ats_response['Education'].strip() not in ["", "No education details found"]:
                        with st.expander("Education", expanded=True):
                            edu_entries = [entry.strip() for entry in ats_response['Education'].split('\n\n') if entry.strip()]
                            for entry in edu_entries:
                                lines = [line.strip() for line in entry.split('\n') if line.strip()]
                                if lines:
                                    # Highlight degree information
                                    if any(word in lines[0].lower() for word in ['bachelor', 'master', 'phd', 'b.tech', 'b.e', 'm.tech', 'bsc', 'msc', 'b.s', 'm.s']):
                                        st.markdown(f"**{lines[0]}**")
                                    else:
                                        st.markdown(f"**{lines[0]}**")
                                    
                                    # Process remaining lines with special formatting
                                    for detail in lines[1:]:
                                        if any(word in detail.lower() for word in ['gpa', 'grade', 'score']):
                                            st.markdown(f" *{detail}*")
                                        elif any(word in detail.lower() for word in ['university', 'college', 'institute', 'school']):
                                            st.markdown(f" {detail}")
                                        elif re.search(r'\b(20\d\d|\d{4})\b', detail):
                                            st.markdown(f" *{detail}*")
                                        else:
                                            st.markdown(f"- {detail}")
                    else:
                        st.warning("Education section required but not found in resume. Please ensure your resume has a clearly labeled Education section.")
                except Exception as e:
                    st.error(f"Error displaying education: {str(e)}")
                
                # Experience
                try:
                    if ats_response.get('Experience') and isinstance(ats_response['Experience'], str) \
                       and ats_response['Experience'].strip() not in ["", "No experience details found"]:
                        with st.expander("Experience", expanded=True):
                            exp_entries = [entry.strip() for entry in ats_response['Experience'].split('\n\n') if entry.strip()]
                            for entry in exp_entries:
                                lines = [line.strip() for line in entry.split('\n') if line.strip()]
                                if lines:
                                    # Highlight tech roles
                                    if any(word in lines[0].lower() for word in ['engineer', 'developer', 'programmer', 'analyst', 'scientist', 'architect']):
                                        st.markdown(f" **{lines[0]}**")
                                    else:
                                        st.markdown(f"**{lines[0]}**")
                                    
                                    # Format dates/company info
                                    if len(lines) > 1:
                                        if re.search(r'\b(20\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', lines[1].lower()):
                                            st.markdown(f" *{lines[1]}*")
                                        else:
                                            st.markdown(f"*{lines[1]}*")
                                    
                                    # Format bullet points with tech highlighting
                                    for bullet in lines[2:]:
                                        clean_bullet = re.sub(r'^[•\-\*]\s*', '', bullet)
                                        if any(tech in bullet.lower() for tech in ['python', 'java', 'javascript', 'react', 'node', 'aws', 'cloud', 'api']):
                                            st.markdown(f"- {clean_bullet}")
                                        elif any(word in bullet.lower() for word in ['lead', 'team', 'manage', 'collaborate']):
                                            st.markdown(f"- 👥 {clean_bullet}")
                                        else:
                                            st.markdown(f"- {clean_bullet}")
                    else:
                        st.warning("Experience section required but not found in resume. Please ensure your resume has a clearly labeled Experience section.")
                except Exception as e:
                    st.error(f"Error displaying experience: {str(e)}")
                
                # Projects
                try:
                    if ats_response.get('Projects') and isinstance(ats_response['Projects'], str) \
                       and ats_response['Projects'].strip() not in ["", "No projects found"]:
                        with st.expander("Projects", expanded=False):
                            proj_entries = [entry.strip() for entry in ats_response['Projects'].split('\n\n') if entry.strip()]
                            for entry in proj_entries:
                                lines = [line.strip() for line in entry.split('\n') if line.strip()]
                                if lines:
                                    st.markdown(f"**{lines[0]}**")
                                    for detail in lines[1:]:
                                        st.markdown(f"- {detail}")
                except Exception as e:
                    st.error(f"Error displaying projects: {str(e)}")
                
                # Achievements
                try:
                    if ats_response.get('Achievements') and isinstance(ats_response['Achievements'], str) \
                       and ats_response['Achievements'].strip() not in ["", "No achievements found"]:
                        with st.expander("Achievements", expanded=False):
                            ach_entries = [entry.strip() for entry in ats_response['Achievements'].split('\n\n') if entry.strip()]
                            for entry in ach_entries:
                                lines = [line.strip() for line in entry.split('\n') if line.strip()]
                                if lines:
                                    st.markdown(f"- **{lines[0]}**")
                                    for detail in lines[1:]:
                                        st.markdown(f"  - {detail}")
                except Exception as e:
                    st.error(f"Error displaying achievements: {str(e)}")
            
            # Tab 2: Skills Analysis
            with tab2:
                st.markdown("### Skills Analysis")
                
                # Detailed skills breakdown
                st.markdown("#### Skills Breakdown")
                
                for category in DOMAIN_CATEGORIES:
                    if category in ats_response['Category_Matches']:
                        match_percent = ats_response['Category_Matches'][category]
                        color = "green" if match_percent >= 80 else "orange" if match_percent >= 60 else "red"
                        
                        with st.expander(f"{category} ({match_percent}%)", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            # Job Description Skills
                            with col1:
                                st.markdown("**Job Description Skills**")
                                jd_skills = ats_response['JD_Categories'].get(category, {})
                                if jd_skills:
                                    df = pd.DataFrame({
                                        'Skill': list(jd_skills.keys()),
                                        'Score': list(jd_skills.values())
                                    }).sort_values('Score', ascending=False)
                                    st.dataframe(df)
                                else:
                                    st.info("No relevant skills found in job description")
                            
                            # Resume Skills
                            with col2:
                                st.markdown("**Your Resume Skills**")
                                resume_skills = ats_response['Resume_Categories'].get(category, {})
                                if resume_skills:
                                    df = pd.DataFrame({
                                        'Skill': list(resume_skills.keys()),
                                        'Score': list(resume_skills.values())
                                    }).sort_values('Score', ascending=False)
                                    st.dataframe(df)
                                else:
                                    st.warning("No matching skills found in resume")
                            
                            # Skill gap analysis
                            if jd_skills and resume_skills:
                                missing_skills = set(jd_skills.keys()) - set(resume_skills.keys())
                                if missing_skills:
                                    st.markdown("**Recommended Skills to Add**")
                                    st.write(", ".join(sorted(missing_skills)))
            
            # Tab 3: Recommendations
            with tab3:
                st.markdown("### Personalized Recommendations")
                
                # Resume Structure Analysis
                with st.expander("Resume Structure Evaluation", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if ats_response.get('Projects') and ats_response['Projects'] and ats_response['Projects'] not in ["", "No projects found"]:
                            st.success("✔ Strong projects section")
                            st.markdown(f"• {len(ats_response['Projects'])} relevant projects listed")
                        else:
                            st.error("✘ Missing projects section")
                    
                    with col2:
                        if ats_response.get('Achievements') and ats_response['Achievements'] and ats_response['Achievements'] not in ["", "No achievements found"]:
                            st.success("✔ Achievements well highlighted")
                            st.markdown(f"• {len(ats_response['Achievements'])} quantifiable achievements")
                        else:
                            st.error("✘ Missing achievements section")
                    
                    st.info(" Structure Improvement Tips:")
                    st.markdown("""
                    - **Bullet points**: Use for readability (3-5 per section)
                    - **Length**: Keep to 1-2 pages maximum
                    - **Action verbs**: Use strong verbs (developed, optimized, led)
                    - **Metrics**: Quantify achievements (e.g., "Improved performance by 30%")
                    - **White space**: Ensure proper spacing between sections
                    """)
                
                # Skill Development Recommendations
                with st.expander("Skill Enhancement", expanded=True):
                    if ats_response.get('Missing_Keywords') and ats_response['Missing_Keywords']:
                        st.error(" Key Skills to Develop:")
                        for keyword in ats_response['Missing_Keywords'][:5]:
                            st.markdown(f"- {keyword}")
                        
                        st.info(" Recommended Learning Resources:")
                        st.markdown("""
                        - [FreeCodeCamp](https://www.freecodecamp.org/) - Free coding tutorials
                        - [Coursera](https://www.coursera.org/) - Professional certificates  
                        - [Udemy](https://www.udemy.com/) - Affordable courses
                        - [LinkedIn Learning](https://www.linkedin.com/learning/) - Career-focused skills
                        """)
                    else:
                        st.success(" Excellent skill match with job requirements!")
                
                # ATS Optimization Tips
                with st.expander("ATS Optimization Tips", expanded=True):
                    st.markdown("""
                    **To improve your ATS score:**
                    - Include missing keywords naturally in your resume
                    - Match job title/headline with the position
                    - Use standard section headings (Experience, Education)
                    - Avoid graphics/tables that scanners can't read
                    - Save as .docx or .pdf (avoid images/scanned PDFs)
                    """)
                
                # Action Items
                if ats_response.get('Missing_Keywords') and ats_response['Missing_Keywords']:
                    st.markdown("### Action Items")
                    for i, rec in enumerate(ats_response['Missing_Keywords'], 1):
                        st.markdown(f"{i}. {rec}")
    else:
        st.warning("Please upload a resume and enter a job description.")