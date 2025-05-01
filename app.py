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

# Enhanced CS Resume Analysis Functions
def extract_cs_resume_sections(resume_text):
    """
    Extract and categorize sections from CS resumes with high accuracy
    Returns: dict with keys: 'education', 'experience', 'skills', 'projects', 'publications'
    """
    sections = {
        'education': [],
        'experience': [],
        'skills': [],
        'projects': [],
        'publications': []
    }
    
    # Enhanced patterns for CS resumes
    edu_pattern = r'(?i)(education|academic background|degrees?)(.*?)(?=(experience|work history|projects|$))'
    exp_pattern = r'(?i)(experience|work history|employment)(.*?)(?=(projects|skills|education|$))'
    skills_pattern = r'(?i)(technical skills|skills|competencies)(.*?)(?=(projects|experience|education|$))'
    projects_pattern = r'(?i)(projects|research work)(.*?)(?=(skills|experience|education|$))'
    
    # Extract sections using improved patterns
    try:
        sections['education'] = re.findall(edu_pattern, resume_text, re.DOTALL)
        sections['experience'] = re.findall(exp_pattern, resume_text, re.DOTALL)
        sections['skills'] = re.findall(skills_pattern, resume_text, re.DOTALL)
        sections['projects'] = re.findall(projects_pattern, resume_text, re.DOTALL)
    except Exception as e:
        st.error(f"Section extraction error: {str(e)}")
    
    return sections

def analyze_cs_resume(resume_text, jd_text):
    """
    Comprehensive CS resume analysis
    Returns: {
        'match_score': float,
        'section_analysis': dict,
        'skill_analysis': dict,
        'recommendations': list
    }
    """
    analysis = {
        'match_score': 0,
        'section_analysis': {},
        'skill_analysis': {},
        'recommendations': []
    }
    
    try:
        # Section analysis
        sections = extract_cs_resume_sections(resume_text)
        analysis['section_analysis'] = sections
        
        # Skill matching
        skill_result = analyze_cs_skills(resume_text, jd_text)
        analysis['skill_analysis'] = skill_result
        analysis['match_score'] = skill_result['match_percentage']
        
        # Generate recommendations
        if skill_result['missing_skills']:
            analysis['recommendations'].append(
                f"Add these skills: {', '.join(skill_result['missing_skills'][:5])}"
            )
        
        if not sections['projects']:
            analysis['recommendations'].append("Add a projects section")
            
        if analysis['match_score'] < 70:
            analysis['recommendations'].append("Strengthen alignment with job requirements")
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
    
    return analysis

def analyze_cs_skills(resume_text, jd_text):
    """
    Specialized skill analysis for computer science resumes
    Returns: {
        'match_percentage': float,
        'skill_categories': dict,
        'missing_skills': list,
        'strong_skills': list
    }
    """
    # Enhanced CS-specific skill extraction
    resume_skills = extract_cs_resume_sections(resume_text)['skills']
    jd_skills = extract_skills_and_keywords(jd_text)
    
    # Calculate matches using TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    match_percentage = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    
    # Get skill gaps and strengths
    missing_skills = set(jd_skills) - set(resume_skills)
    strong_skills = set(resume_skills) & set(jd_skills)
    
    return {
        'match_percentage': round(match_percentage, 1),
        'skill_categories': DOMAIN_CATEGORIES,
        'missing_skills': sorted(missing_skills),
        'strong_skills': sorted(strong_skills)
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
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
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

# Updated Streamlit Interface for CS Resumes
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #1f497d;">CS Resume Analyzer</h1>
        <p style="color: #666;">Precision analysis for computer science resumes</p>
    </div>
""", unsafe_allow_html=True)

uploaded_resume = st.file_uploader("Upload Computer Science Resume (PDF)", type="pdf")
jd_input = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Resume"):
    if uploaded_resume and jd_input.strip():
        with st.spinner("Performing deep analysis..."):
            try:
                # Extract and analyze
                resume_text = extract_text_from_pdf(uploaded_resume)
                if not resume_text:
                    st.error("Failed to extract text from PDF")
                    st.stop()
                
                # Get comprehensive analysis
                analysis = analyze_cs_resume(resume_text, jd_input)
                sections = extract_resume_sections(resume_text)
                
                # Display results
                st.success("Analysis Complete!")
                
                # Match percentage
                st.metric("Overall Match Score", f"{analysis['match_score']}%")
                
                # Section analysis
                with st.expander("Resume Section Analysis", expanded=True):
                    for section, content in sections.items():
                        if content:
                            st.subheader(section.title())
                            st.write(content)
                
                # Skill matching
                with st.expander("Skill Matching", expanded=True):
                    st.subheader("Skill Match by Category")
                    for category, skills in analysis['skill_analysis']['skill_categories'].items():
                        matched = len(set(skills['skills']) & set(analysis['skill_analysis']['strong_skills']))
                        total = len(skills['skills'])
                        if total > 0:
                            percent = (matched / total) * 100
                            st.progress(int(percent), text=f"{category}: {matched}/{total} skills")
                
                # Missing skills
                if analysis['skill_analysis']['missing_skills']:
                    with st.expander("Recommended Skills to Add", expanded=True):
                        st.write(", ".join(analysis['skill_analysis']['missing_skills']))
                
                # Recommendations
                if analysis['recommendations']:
                    with st.expander("Recommendations", expanded=True):
                        st.write("\n".join(analysis['recommendations']))
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.stop()
    else:
        st.warning("Please upload a resume and enter a job description")