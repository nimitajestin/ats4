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
        
        # Calculate match percentage
        match_data = calculate_match_percentage(resume_text, jd_text)
        
        # Extract keywords and skills
        resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
        
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
                # Profile Summary with better formatting
                st.markdown("### Professional Profile")
                if ats_response['Profile Summary']:
                    st.info(ats_response['Profile Summary'])
                else:
                    st.warning("No profile summary found")
                
                # Education and Experience in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Education")
                    if ats_response['Education'] and ats_response['Education'] != "No education details found":
                        for line in ats_response['Education'].split('\n'):
                            if line.strip():
                                st.markdown(f"- {line}")
                    else:
                        st.warning("No education details found")
                
                with col2:
                    st.markdown("### Experience")
                    if ats_response['Experience'] and ats_response['Experience'] != "No experience details found":
                        for line in ats_response['Experience'].split('\n'):
                            if line.strip():
                                if line.startswith('  '):  # It's a responsibility/achievement
                                    st.markdown(f"  • {line.strip()}")
                                else:  # It's a role title
                                    st.markdown(f"**{line}**")
                    else:
                        st.warning("No experience details found")
                
                # Projects and Achievements
                if ats_response['Projects'] or ats_response['Achievements']:
                    st.markdown("### Key Highlights")
                    
                    if ats_response['Projects']:
                        st.markdown("#### Notable Projects")
                        for project in ats_response['Projects']:
                            st.markdown(f"• {project}")
                    
                    if ats_response['Achievements']:
                        st.markdown("#### Key Achievements")
                        for achievement in ats_response['Achievements']:
                            st.markdown(f"• {achievement}")
            
            # Tab 2: Skills Analysis
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
            
            # Tab 3: Recommendations - Comprehensive feedback and suggestions
            with tab3:
                st.markdown("### Personalized Recommendations")
                
                # Resume Structure Evaluation
                with st.expander("📄 Resume Structure", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if ats_response.get('Projects'):
                            st.success("✅ Strong Projects Section")
                            st.metric("Projects Listed", len(ats_response['Projects']))
                        else:
                            st.error("❌ Missing Projects Section")
                    
                    with col2:
                        if ats_response.get('Achievements'):
                            st.success("✅ Achievements Highlighted")
                            st.metric("Key Achievements", len(ats_response['Achievements']))
                        else:
                            st.error("❌ Missing Achievements Section")

                    st.divider()
                    st.markdown("**Improvement Tips:**")
                    st.markdown("""
                    - Use 3-5 bullet points per section
                    - Keep resume length 1-2 pages
                    - Start with strong action verbs
                    - Include quantifiable metrics
                    - Use standard section headers
                    """)

                # Skill Development Section
                with st.expander("📚 Skill Development", expanded=True):
                    if ats_response['Missing Keywords']:
                        cols = st.columns(2)
                        with cols[0]:
                            st.error("🚨 Missing Key Skills")
                            for skill in ats_response['Missing Keywords'][:3]:
                                st.markdown(f"- {skill}")
                        
                        with cols[1]:
                            st.info("💡 Learning Resources")
                            st.markdown("""
                            - [Coursera](https://www.coursera.org)
                            - [Udemy](https://www.udemy.com)
                            - [FreeCodeCamp](https://freecodecamp.org)
                            """)
                    else:
                        st.success("🎯 Excellent Skill Alignment!")

                # ATS Optimization Tips
                with st.expander("🤖 ATS Optimization", expanded=True):
                    st.markdown("""
                    **Top ATS Tips:**
                    - Use exact job title match
                    - Include keywords from description
                    - Avoid graphics/tables
                    - Use .docx or .pdf format
                    - Keep formatting simple
                    """)

                # Action Items
                st.markdown("---")
                st.markdown("### Action Items")
                if ats_response['Recommendations']:
                    for i, rec in enumerate(ats_response['Recommendations'], 1):
                        st.markdown(f"{i}. **{rec}**")
                else:
                    st.info("No additional recommendations")