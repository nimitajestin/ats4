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

def extract_text_from_pdf(pdf_file):
    try:
        # Improved PDF text extraction with section preservation
        text = ""
        
        # Try PyPDF2 first
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Clean and preserve section structure
                page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                page_text = re.sub(r'(?<=\n)\s+', '\n', page_text)  # Clean line breaks
                text += page_text + '\n\n'
        
        # Fallback to pdfplumber if PyPDF2 fails
        if not text.strip():
            with pdfplumber.open(pdf_file) as pdf:
                text = '\n\n'.join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        # Enhanced section header detection
        text = re.sub(r'(?i)(education|academics|degree)', '\nEDUCATION\n', text)
        text = re.sub(r'(?i)(experience|work history|employment)', '\nEXPERIENCE\n', text)
        text = re.sub(r'(?i)(projects|technical projects)', '\nPROJECTS\n', text)
        
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

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
    education_info = []
    lines = text.split('\n')
    is_education_section = False
    current_degree = {}
    
    # Keywords for education detection
    edu_keywords = ['education', 'academic', 'qualification', 'university', 'college', 'institute', 'school']
    degree_keywords = ['bachelor', 'master', 'phd', 'b.tech', 'b.e', 'm.tech', 'bsc', 'msc']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for education section start
        if any(keyword.lower() in line.lower() for keyword in edu_keywords):
            is_education_section = True
            if current_degree:
                education_info.append(current_degree)
                current_degree = {}
            continue
        
        # Check for end of education section
        if is_education_section and any(keyword.lower() in line.lower() for keyword in ['experience', 'work', 'skills']):
            is_education_section = False
            if current_degree:
                education_info.append(current_degree)
            break
        
        if is_education_section:
            # Detect degree information
            if any(keyword.lower() in line.lower() for keyword in degree_keywords):
                if current_degree:
                    education_info.append(current_degree)
                current_degree = {'degree': line}
            # Detect university/institution
            elif any(keyword.lower() in line.lower() for keyword in ['university', 'college', 'institute']):
                if current_degree:
                    current_degree['institution'] = line
                else:
                    current_degree = {'institution': line}
            # Detect graduation year
            elif re.search(r'\b20\d{2}\b', line):
                if current_degree:
                    current_degree['year'] = re.search(r'\b20\d{2}\b', line).group()
            # Detect GPA/grades
            elif any(keyword.lower() in line.lower() for keyword in ['gpa', 'grade', 'cgpa']):
                if current_degree:
                    current_degree['grades'] = line
    
    # Add final degree if any
    if current_degree:
        education_info.append(current_degree)
    
    # Format education information
    formatted_education = []
    for edu in education_info:
        parts = []
        if 'degree' in edu:
            parts.append(edu['degree'])
        if 'institution' in edu:
            parts.append(edu['institution'])
        if 'year' in edu:
            parts.append(f"({edu['year']})")
        if 'grades' in edu:
            parts.append(f"- {edu['grades']}")
        formatted_education.append(' '.join(parts))
    
    return '\n'.join(formatted_education) if formatted_education else "No education details found"

def analyze_experience(text):
    """Enhanced experience analysis with better structure and detail capture"""
    experience_info = []
    lines = text.split('\n')
    is_experience_section = False
    current_role = {}
    
    # Keywords for experience detection
    exp_keywords = ['experience', 'employment', 'work history']
    role_keywords = ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'architect']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for experience section start
        if any(keyword.lower() in line.lower() for keyword in exp_keywords):
            is_experience_section = True
            if current_role:
                experience_info.append(current_role)
                current_role = {}
            continue
        
        # Check for end of experience section
        if is_experience_section and any(keyword.lower() in line.lower() for keyword in ['education', 'skills', 'projects']):
            is_experience_section = False
            if current_role:
                experience_info.append(current_role)
            break
        
        if is_experience_section:
            # Detect role/title
            if any(keyword.lower() in line.lower() for keyword in role_keywords):
                if current_role:
                    experience_info.append(current_role)
                current_role = {'title': line}
            # Detect company name (usually follows the title)
            elif current_role and 'company' not in current_role:
                current_role['company'] = line
            # Detect date range
            elif re.search(r'\b(19|20)\d{2}\b', line):
                if current_role:
                    current_role['duration'] = line
            # Detect responsibilities/achievements
            elif line.startswith(('•', '-', '*')) or re.search(r'^\d+\.', line):
                if 'responsibilities' not in current_role:
                    current_role['responsibilities'] = []
                current_role['responsibilities'].append(line)
    
    # Add final role if any
    if current_role:
        experience_info.append(current_role)
    
    # Format experience information
    formatted_experience = []
    for exp in experience_info:
        parts = []
        if 'title' in exp:
            parts.append(exp['title'])
        if 'company' in exp:
            parts.append(f"at {exp['company']}")
        if 'duration' in exp:
            parts.append(f"({exp['duration']})")
        formatted_experience.append(' '.join(parts))
        if 'responsibilities' in exp:
            formatted_experience.extend([f"  {resp}" for resp in exp['responsibilities']])
    
    return '\n'.join(formatted_experience) if formatted_experience else "No experience details found"

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
                            edu_text = ats_response['Education']
                            
                            # Handle common CS education patterns
                            if 'Degree' in edu_text or 'University' in edu_text or 'GPA' in edu_text:
                                entries = [e.strip() for e in re.split(r'\n\n|•|-', edu_text) if e.strip()]
                                for entry in entries:
                                    lines = [line.strip() for line in entry.split('\n') if line.strip()]
                                    if lines:
                                        st.markdown(f"**{lines[0]}**")
                                        for detail in lines[1:]:
                                            if any(word in detail.lower() for word in ['gpa', 'grade', 'score']):
                                                st.markdown(f"📊 *{detail}*")
                                            elif any(word in detail.lower() for word in ['university', 'college', 'institute']):
                                                st.markdown(f"🏛️ {detail}")
                                            else:
                                                st.markdown(f"- {detail}")
                    else:
                        st.warning("We couldn't detect your education details. For CS resumes, please ensure:")
                        st.markdown("""
                        - Your education section includes your degree (e.g., 'B.Tech Computer Science')
                        - University name is clearly listed
                        - Dates or expected graduation are included
                        """)
                except Exception as e:
                    st.error("Error processing education section. Please check your education formatting.")
                
                # Experience
                try:
                    if ats_response.get('Experience') and isinstance(ats_response['Experience'], str) \
                       and ats_response['Experience'].strip() not in ["", "No experience details found"]:
                        with st.expander("Experience", expanded=True):
                            exp_text = ats_response['Experience']
                            
                            # Enhanced parsing for technical experience
                            entries = []
                            if '\n\n' in exp_text:
                                entries = [e.strip() for e in exp_text.split('\n\n') if e.strip()]
                            elif '•' in exp_text:
                                entries = [e.strip() for e in exp_text.split('•') if e.strip()]
                            elif '-' in exp_text:
                                entries = [e.strip() for e in exp_text.split('-') if e.strip()]
                            else:
                                entries = [exp_text]
                            
                            for entry in entries:
                                lines = [line.strip() for line in entry.split('\n') if line.strip()]
                                if lines:
                                    # Position/Company with tech role detection
                                    role = lines[0]
                                    if any(word.lower() in role.lower() for word in ['engineer', 'developer', 'analyst', 'scientist', 'intern']):
                                        st.markdown(f"👨‍💻 **{role}**")
                                    else:
                                        st.markdown(f"**{role}**")
                                    
                                    # Dates/Location with tech term detection
                                    if len(lines) > 1:
                                        date_line = lines[1]
                                        if any(word.lower() in date_line.lower() for word in ['present', 'remote', 'hybrid']):
                                            st.markdown(f"📅 *{date_line}*")
                                        elif any(char.isdigit() for char in date_line):
                                            st.markdown(f"*{date_line}*")
                                        
                                    # Bullet points with tech keyword highlighting
                                    for line in lines[2:]:
                                        clean_line = re.sub(r'[•-]', '', line).strip()
                                        if clean_line:
                                            if any(word.lower() in clean_line.lower() for word in 
                                                  ['python', 'java', 'c++', 'algorithm', 'database', 'api', 'cloud']):
                                                st.markdown(f"- 💻 {clean_line}")
                                            elif any(word.lower() in clean_line.lower() for word in 
                                                    ['lead', 'manage', 'team']):
                                                st.markdown(f"- 👥 {clean_line}")
                                            else:
                                                st.markdown(f"- {clean_line}")
                    else:
                        st.warning("We couldn't detect your experience details. For CS resumes, please ensure:")
                        st.markdown("""
                        - Each position has a clear title (e.g., 'Software Engineer Intern')
                        - Technical skills and achievements are bulleted
                        - Dates are included for each role
                        """)
                except Exception as e:
                    st.error("Error processing experience section. Please check your experience formatting.")
            
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