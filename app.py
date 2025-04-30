import PyPDF2
import streamlit as st
import os                  
import re                 
import json               
from collections import Counter  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity      

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
    'under', 'again', 'further', 'then', 'once'
}

DOMAIN_CATEGORIES = {
    'Core Technical': {
        'weight': 1.3,
        'skills': ['python', 'java', 'sql', 'aws', 'docker', 'react', 'django'],
        'context_boosters': ['built', 'developed', 'implemented', 'optimized']
    },
    'Industry Expertise': {
        'weight': 1.7,
        'skills': [
            'cloud architecture', 'devops', 'microservices', 'api design',
            'financial modeling', 'risk management', 'investment analysis',
            'health informatics', 'fda compliance', 'clinical systems',
            'agile development', 'user research', 'product strategy'
        ],
        'context_boosters': ['led', 'managed', 'spearheaded', 'transformed']
    },
    'Applied Skills': {
        'weight': 1.4,
        'skills': [
            'data analysis', 'machine learning', 'process automation',
            'system design', 'performance optimization', 'security compliance'
        ],
        'context_boosters': ['analyzed', 'improved', 'secured', 'scaled']
    },
    'Leadership': {
        'weight': 1.2,
        'skills': [
            'team leadership', 'stakeholder management',
            'strategic planning', 'cross-functional collaboration'
        ],
        'context_boosters': ['led', 'mentored', 'aligned', 'drove']
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
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def analyze_education(text):
    education_info = []
    
    lines = text.split('\n')
    is_education_section = False
    education_section = []
    
    edu_keywords = ['education', 'qualification', 'degree', 'university', 'college', 'institute', 'school']
    degree_keywords = ['bachelor', 'master', 'phd', 'b.tech', 'b.e', 'm.tech', 'bsc', 'msc']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if any(keyword.lower() in line.lower() for keyword in edu_keywords):
            is_education_section = True
            continue
            
        if is_education_section:
            if any(keyword.lower() in line.lower() for keyword in ['experience', 'work', 'skills', 'projects']):
                is_education_section = False
                break
                
            if any(keyword.lower() in line.lower() for keyword in degree_keywords) or \
               any(keyword.lower() in line.lower() for keyword in edu_keywords):
                education_section.append(line)
    
    if not education_section:
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in degree_keywords) or \
               any(keyword.lower() in line.lower() for keyword in edu_keywords):
                education_section.append(line)
    
    return '\n'.join(education_section) if education_section else "No education details found"

def analyze_experience(text):
    experience_info = []
    
    lines = text.split('\n')
    is_experience_section = False
    experience_section = []
    
    exp_keywords = ['experience', 'employment', 'work history', 'professional background']
    role_keywords = ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'specialist']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if any(keyword.lower() in line.lower() for keyword in exp_keywords):
            is_experience_section = True
            continue
            
        if is_experience_section:
            if any(keyword.lower() in line.lower() for keyword in ['education', 'skills', 'projects', 'achievements']):
                is_experience_section = False
                break
                
            if any(keyword.lower() in line.lower() for keyword in role_keywords) or \
               re.search(r'\b(19|20)\d{2}\b', line):
                experience_section.append(line)
    
    if not experience_section:
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in role_keywords) or \
               re.search(r'\b(19|20)\d{2}\b', line):
                experience_section.append(line)
    
    return '\n'.join(experience_section) if experience_section else "No experience details found"

def analyze_projects(text):
    projects = []
    
    lines = text.split('\n')
    is_project_section = False
    current_project = []
    
    project_keywords = ['project', 'developed', 'implemented', 'created', 'built']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if any(keyword.lower() in line.lower() for keyword in ['projects', 'technical projects']):
            is_project_section = True
            continue
            
        if is_project_section:
            if any(keyword.lower() in line.lower() for keyword in ['education', 'experience', 'skills', 'achievements']):
                is_project_section = False
                if current_project:
                    projects.append(' '.join(current_project))
                break
                
            if any(keyword.lower() in line.lower() for keyword in project_keywords):
                if current_project:
                    projects.append(' '.join(current_project))
                current_project = [line]
            elif current_project:
                current_project.append(line)
    
    if current_project:
        projects.append(' '.join(current_project))
    
    if not projects:
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in project_keywords):
                projects.append(line)
    
    return projects if projects else []

def analyze_achievements(text):
    achievements = []
    
    lines = text.split('\n')
    is_achievement_section = False
    current_achievement = []
    
    achievement_keywords = [
        'achieved', 'awarded', 'won', 'recognized', 'selected', 'ranked',
        'improved', 'increased', 'decreased', 'reduced', 'saved', 'delivered',
        'led', 'managed', 'honor', 'certificate', 'certification'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if any(keyword.lower() in line.lower() for keyword in ['achievements', 'accomplishments', 'honors']):
            is_achievement_section = True
            continue
            
        if is_achievement_section:
            if any(keyword.lower() in line.lower() for keyword in ['education', 'experience', 'skills', 'projects']):
                is_achievement_section = False
                if current_achievement:
                    achievements.append(' '.join(current_achievement))
                break
                
            if any(keyword.lower() in line.lower() for keyword in achievement_keywords):
                if current_achievement:
                    achievements.append(' '.join(current_achievement))
                current_achievement = [line]
            elif current_achievement:
                current_achievement.append(line)
    
    if current_achievement:
        achievements.append(' '.join(current_achievement))
    
    if not achievements:
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in achievement_keywords):
                achievements.append(line)
    
    return achievements if achievements else []

def extract_resume_sections(text):
    if not text:
        return {
            'Profile Summary': '',
            'Education': '',
            'Experience': '',
            'Projects': [],
            'Achievements': []
        }
    
    education = analyze_education(text)
    experience = analyze_experience(text)
    projects = analyze_projects(text)
    achievements = analyze_achievements(text)
    
    profile_summary = []
    if education and education != "No education details found":
        profile_summary.append(education.split('\n')[0])
    if experience and experience != "No experience details found":
        profile_summary.append(experience.split('\n')[0])
    
    return {
        'Profile Summary': ' | '.join(profile_summary) if profile_summary else "No profile summary available",
        'Education': education,
        'Experience': experience,
        'Projects': projects,
        'Achievements': achievements
    }

@st.cache_data 
def get_ats_feedback(resume_text, jd_text):
    try:
        sections = extract_resume_sections(resume_text)
        
        match_data = calculate_match_percentage(resume_text, jd_text)
        
        resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
        
        matched_keywords = list(set(resume_keywords) & set(jd_keywords))
        missing_keywords = list(set(jd_keywords) - set(resume_keywords))
        
        return {
            "JD Match": f"{match_data['score']}%",
            "Profile Summary": sections['Profile Summary'],
            "Education": sections['Education'],
            "Experience": sections['Experience'],
            "Projects": sections['Projects'],
            "Achievements": sections['Achievements'],
            "Key Strengths": matched_keywords[:5],
            "Missing Keywords": missing_keywords[:5],
            "Category Matches": match_data['category_scores'],
            "Recommendations": [
                "Add missing keywords to your resume",
                "Quantify your achievements with metrics",
                "Add more detail to your project descriptions",
                "Highlight relevant technical skills",
                "Include certifications if available"
            ]
        }
    except Exception as e:
        st.error(f"Error in text processing: {str(e)}")
        return None

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
            resume_text = extract_text_from_pdf(uploaded_resume)
            if not resume_text:
                st.error("Failed to extract text from PDF")
                st.stop()
            
            ats_response = get_ats_feedback(resume_text, jd_input)
            if not ats_response:
                st.error("Failed to generate ATS feedback")
                st.stop()
            
            match_pct = float(ats_response['JD Match'].strip('%'))
            color = 'green' if match_pct >= 80 else 'orange' if match_pct >= 60 else 'red'
            
            st.markdown(
                f"<h2 style='color: {color}; text-align: center;'>"
                f"Overall Match: {match_pct:.1f}%</h2>", 
                unsafe_allow_html=True
            )
            
            tab1, tab2, tab3 = st.tabs(["Overview", "Skills Analysis", "Recommendations"])
            
            with tab1:
                st.markdown("### Profile Summary")
                if ats_response['Profile Summary']:
                    st.info(ats_response['Profile Summary'])
                else:
                    st.warning("No profile summary found")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Education")
                    if ats_response['Education'] and ats_response['Education'] != "No education details found":
                        for line in ats_response['Education'].split('\n'):
                            if line.strip():
                                st.write(line)
                    else:
                        st.warning("No education details found")
                
                with col2:
                    st.markdown("### Experience")
                    if ats_response['Experience'] and ats_response['Experience'] != "No experience details found":
                        for line in ats_response['Experience'].split('\n'):
                            if line.strip():
                                st.write(line)
                    else:
                        st.warning("No experience details found")
                
                st.markdown("### Key Highlights")
                
                if ats_response['Projects']:
                    st.markdown("#### Notable Projects")
                    for project in ats_response['Projects']:
                        st.markdown(f"* {project}")
                else:
                    st.warning("No projects found")
                
                if ats_response['Achievements']:
                    st.markdown("#### Key Achievements")
                    for achievement in ats_response['Achievements']:
                        st.markdown(f"* {achievement}")
                else:
                    st.warning("No achievements found")
            
            with tab2:
                st.markdown("### Skills Analysis")
                
                st.markdown("#### Key Strengths")
                if ats_response['Key Strengths']:
                    for strength in ats_response['Key Strengths']:
                        st.success(f"✓ {strength}")
                else:
                    st.warning("No key strengths identified")
                
                st.markdown("#### Missing Keywords")
                if ats_response['Missing Keywords']:
                    for keyword in ats_response['Missing Keywords']:
                        st.error(f"✗ {keyword}")
                else:
                    st.success("No critical missing keywords")
                
                st.markdown("#### Category Matches")
                for category, score in ats_response['Category Matches'].items():
                    color = 'green' if score >= 80 else 'orange' if score >= 60 else 'red'
                    st.markdown(f"<span style='color:{color}'>{category}: {score}%</span>", unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### Recommendations")
                if ats_response['Recommendations']:
                    for i, rec in enumerate(ats_response['Recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
                else:
                    st.info("No specific recommendations at this time")
    else:
        st.warning("Please upload a resume and enter a job description.")
