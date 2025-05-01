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
        'skills': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'asp.net',
            'html', 'css', 'sass', 'webpack', 'git'
        ],
        'context_boosters': ['built', 'developed', 'implemented', 'optimized', 'architected']
    },
    'Industry Expertise': {
        'weight': 1.7,
        'skills': [
            'cloud architecture', 'devops', 'ci/cd', 'microservices', 'api design',
            'system design', 'distributed systems', 'scalability',
            'financial modeling', 'risk management', 'investment analysis',
            'health informatics', 'fda compliance', 'clinical systems',
            'agile development', 'scrum', 'kanban', 'user research', 'product strategy'
        ],
        'context_boosters': ['led', 'managed', 'spearheaded', 'transformed', 'designed']
    },
    'Applied Skills': {
        'weight': 1.4,
        'skills': [
            'data analysis', 'machine learning', 'deep learning', 'nlp',
            'process automation', 'etl', 'data pipeline',
            'system design', 'performance optimization', 'security compliance',
            'testing', 'debugging', 'monitoring', 'logging'
        ],
        'context_boosters': ['analyzed', 'improved', 'secured', 'scaled', 'automated']
    },
    'Leadership': {
        'weight': 1.2,
        'skills': [
            'team leadership', 'project management', 'stakeholder management',
            'strategic planning', 'cross-functional collaboration',
            'mentoring', 'coaching', 'resource allocation',
            'budget management', 'vendor management'
        ],
        'context_boosters': ['led', 'mentored', 'aligned', 'drove', 'coordinated']
    }
}

def extract_skills_and_keywords(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    skills_data = {category: {'skills': {}, 'context_count': 0} 
                  for category in DOMAIN_CATEGORIES}
    
    # Extract skills with context
    for category, data in DOMAIN_CATEGORIES.items():
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, data['skills'])) + r')\b')
        
        for match in pattern.finditer(text):
            skill = match.group()
            context = text[max(0, match.start()-100):min(len(text), match.end()+100)]
            
            base_score = 0.7
            context_boost = sum(boost in context for boost in data['context_boosters']) * 0.1
            base_score = min(1.0, base_score + context_boost)
            
            if skill in skills_data[category]['skills']:
                base_score = min(1.0, skills_data[category]['skills'][skill] + 0.15)
            
            skills_data[category]['skills'][skill] = base_score
            skills_data[category]['context_count'] += 1
    
    # Extract additional keywords
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
        role_keywords = [
            'software engineer', 'senior engineer', 'lead engineer', 'technical lead',
            'developer', 'full stack developer', 'backend developer', 'frontend developer',
            'data scientist', 'machine learning engineer', 'devops engineer',
            'product manager', 'project manager', 'program manager',
            'architect', 'solutions architect', 'technical architect',
            'analyst', 'business analyst', 'data analyst',
            'consultant', 'technical consultant'
        ]
        
        role = None
        for keyword in role_keywords:
            role_match = re.search(rf'(?i){keyword}[\w\s]*', text)
            if role_match:
                role = role_match.group(0).strip().title()
                break
        
        # Extract key skills (top 3-5)
        skills = []
        skill_keywords = list(set([skill for category in DOMAIN_CATEGORIES.values() for skill in category['skills']]))
        for skill in skill_keywords:
            if skill in text_lower:
                skills.append(skill.title())
                if len(skills) == 5:
                    break
        
        # Build comprehensive summary
        summary_parts = []
        if role:
            summary_parts.append(f"Experienced {role}")
        if years_exp:
            summary_parts.append(f"with {years_exp}+ years of professional experience")
        if skills:
            summary_parts.append(f"specializing in {', '.join(skills[:-1])} and {skills[-1]}")
        
        # Add any certifications or notable achievements
        cert_match = re.search(r'(?i)(certified|certification|certificate)\s+(\w+(?:\s+\w+){0,3})', text)
        if cert_match:
            summary_parts.append(f"Holds {cert_match.group(0)}")
        
        summary_section = [' '.join(summary_parts)]
    
    return ' '.join(summary_section).strip()

def analyze_education(text):
    """Enhanced education analysis with comprehensive degree and field detection"""
    education_info = []
    lines = text.split('\n')
    is_education_section = False
    current_degree = {}
    
    # Enhanced keywords for education detection
    edu_keywords = ['education', 'academic', 'qualification', 'university', 'college', 'institute', 'school']
    degree_patterns = {
        'bachelor': [
            r'bachelor[s]?\s+(?:of|in)\s+(?:science|arts|engineering|technology|computer)',
            r'b\.?(?:sc|tech|e|ca|ba|s)\b',
            r'undergraduate degree'
        ],
        'master': [
            r'master[s]?\s+(?:of|in)\s+(?:science|arts|engineering|technology|computer)',
            r'm\.?(?:sc|tech|ca|ba|s)\b',
            r'graduate degree'
        ],
        'phd': [
            r'ph\.?d',
            r'doctor\s+of\s+philosophy',
            r'doctorate'
        ]
    }
    
    major_patterns = [
        r'computer science',
        r'information technology',
        r'software engineering',
        r'data science',
        r'artificial intelligence',
        r'electrical engineering',
        r'information systems',
        r'business administration',
        r'mathematics'
    ]
    
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
        
        if is_education_section:
            # Check for end of education section
            if any(keyword.lower() in line.lower() for keyword in ['experience', 'work', 'skills']):
                is_education_section = False
                if current_degree:
                    education_info.append(current_degree)
                break
            
            # Detect degree level and major
            for level, patterns in degree_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line.lower()):
                        current_degree['level'] = level.title()
                        # Look for major in the same line
                        for major in major_patterns:
                            if major.lower() in line.lower():
                                current_degree['major'] = major.title()
                        break
            
            # Detect university/institution
            if any(keyword.lower() in line.lower() for keyword in ['university', 'college', 'institute']):
                current_degree['institution'] = line.strip()
            
            # Detect graduation year
            year_match = re.search(r'\b(19|20)\d{2}\b', line)
            if year_match:
                current_degree['year'] = year_match.group()
            
            # Detect GPA/grades
            gpa_match = re.search(r'(?i)(?:gpa|cgpa|grade)[\s:]+([0-9.]+)', line)
            if gpa_match:
                current_degree['gpa'] = gpa_match.group(1)
            
            # If we have enough information, save this degree and start a new one
            if len(current_degree) >= 2 and 'level' in current_degree:
                education_info.append(current_degree)
                current_degree = {}
    
    # Add final degree if any
    if current_degree:
        education_info.append(current_degree)
    
    # Format education information
    formatted_education = []
    for edu in education_info:
        parts = []
        if 'level' in edu and 'major' in edu:
            parts.append(f"{edu['level']} in {edu['major']}")
        elif 'level' in edu:
            parts.append(edu['level'])
        
        if 'institution' in edu:
            parts.append(f"from {edu['institution']}")
        
        if 'year' in edu:
            parts.append(f"({edu['year']})")
        
        if 'gpa' in edu:
            parts.append(f"- GPA: {edu['gpa']}")
        
        if parts:
            formatted_education.append(' '.join(parts))
    
    return '\n'.join(formatted_education) if formatted_education else "No education details found"

def analyze_experience(text):
    """Enhanced experience analysis with better structure and detail capture"""
    experience_info = []
    lines = text.split('\n')
    is_experience_section = False
    current_role = {}
    
    # Enhanced keywords for experience detection
    exp_keywords = ['experience', 'employment', 'work history', 'professional background']
    role_keywords = [
        'engineer', 'developer', 'manager', 'analyst', 'consultant', 'architect',
        'lead', 'director', 'specialist', 'administrator', 'designer'
    ]
    
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
        
        if is_experience_section:
            # Check for end of experience section
            if any(keyword.lower() in line.lower() for keyword in ['education', 'skills', 'projects']):
                is_experience_section = False
                if current_role:
                    experience_info.append(current_role)
                break
            
            # Detect role/title with more context
            if any(keyword.lower() in line.lower() for keyword in role_keywords):
                if current_role:
                    experience_info.append(current_role)
                current_role = {'title': line.strip()}
            
            # Detect company name
            elif current_role and 'company' not in current_role and not line.startswith(('•', '-', '*')):
                current_role['company'] = line.strip()
            
            # Detect date range
            elif re.search(r'\b(19|20)\d{2}\b', line):
                if current_role:
                    current_role['duration'] = line.strip()
            
            # Detect responsibilities/achievements
            elif line.startswith(('•', '-', '*')) or re.search(r'^\d+\.', line):
                if 'responsibilities' not in current_role:
                    current_role['responsibilities'] = []
                # Clean up the bullet point and add it
                cleaned_line = re.sub(r'^[•\-*]\s*', '', line).strip()
                if cleaned_line:
                    current_role['responsibilities'].append(cleaned_line)
    
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
            for resp in exp['responsibilities']:
                formatted_experience.append(f"  • {resp}")
    
    return '\n'.join(formatted_experience) if formatted_experience else "No experience details found"

def analyze_projects(text):
    """Enhanced project analysis with better context and detail extraction"""
    projects = []
    lines = text.split('\n')
    is_project_section = False
    current_project = {'name': '', 'description': [], 'technologies': set()}
    
    # Enhanced keywords for project detection
    project_keywords = ['project', 'developed', 'implemented', 'created', 'built', 'designed']
    tech_keywords = set([skill.lower() for category in DOMAIN_CATEGORIES.values() for skill in category['skills']])
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for project section start
        if any(keyword.lower() in line.lower() for keyword in ['projects', 'technical projects']):
            is_project_section = True
            continue
        
        if is_project_section:
            # Check for end of project section
            if any(keyword.lower() in line.lower() for keyword in ['education', 'experience', 'skills', 'achievements']):
                is_project_section = False
                if current_project['name']:
                    projects.append(current_project)
                break
            
            # Start new project
            if any(keyword.lower() in line.lower() for keyword in project_keywords):
                if current_project['name']:
                    projects.append(current_project)
                current_project = {'name': line.strip(), 'description': [], 'technologies': set()}
            elif current_project['name']:
                # Add description
                current_project['description'].append(line)
                # Extract technologies
                words = line.lower().split()
                current_project['technologies'].update(tech for tech in tech_keywords if tech in words)
    
    # Add final project if any
    if current_project['name']:
        projects.append(current_project)
    
    # Format projects
    formatted_projects = []
    for project in projects:
        project_str = f"{project['name']}"
        if project['technologies']:
            project_str += f" (Technologies: {', '.join(project['technologies'])})"
        if project['description']:
            project_str += f"\n  • {' '.join(project['description'])}"
        formatted_projects.append(project_str)
    
    return formatted_projects if formatted_projects else []

def analyze_achievements(text):
    """Enhanced achievement analysis with better context and metric extraction"""
    achievements = []
    lines = text.split('\n')
    is_achievement_section = False
    current_achievement = []
    
    # Enhanced keywords for achievement detection
    achievement_keywords = [
        'achieved', 'awarded', 'won', 'recognized', 'selected', 'ranked',
        'improved', 'increased', 'decreased', 'reduced', 'saved', 'delivered',
        'led', 'managed', 'honor', 'certificate', 'certification',
        'optimized', 'accelerated', 'enhanced', 'streamlined'
    ]
    
    metric_patterns = [
        r'\d+%',
        r'\$\d+(?:,\d+)*(?:\.\d+)?[KMB]?',
        r'\d+(?:,\d+)*(?:\.\d+)?[KMB]?\s*(?:users|customers|clients|hours|days)',
        r'top\s+\d+%'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for achievements section start
        if any(keyword.lower() in line.lower() for keyword in ['achievements', 'accomplishments', 'honors']):
            is_achievement_section = True
            continue
        
        if is_achievement_section:
            # Check for end of achievements section
            if any(keyword.lower() in line.lower() for keyword in ['education', 'experience', 'skills', 'projects']):
                is_achievement_section = False
                if current_achievement:
                    achievements.append(' '.join(current_achievement))
                break
            
            # Process achievement
            has_keyword = any(keyword.lower() in line.lower() for keyword in achievement_keywords)
            has_metric = any(re.search(pattern, line) for pattern in metric_patterns)
            
            if has_keyword or has_metric:
                if current_achievement:
                    achievements.append(' '.join(current_achievement))
                current_achievement = [line]
            elif current_achievement:
                current_achievement.append(line)
    
    # Add final achievement if any
    if current_achievement:
        achievements.append(' '.join(current_achievement))
    
    # If no structured achievements section found, scan entire text
    if not achievements:
        for line in lines:
            has_keyword = any(keyword.lower() in line.lower() for keyword in achievement_keywords)
            has_metric = any(re.search(pattern, line) for pattern in metric_patterns)
            if has_keyword and has_metric:
                achievements.append(line.strip())
    
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
        
        # Extract keywords and skills
        resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
        
        # Calculate matched and missing keywords
        matched_keywords = list(set(resume_keywords) & set(jd_keywords))
        missing_keywords = list(set(jd_keywords) - set(resume_keywords))
        
        # Generate customized recommendations
        recommendations = []
        
        # Add keyword-based recommendations
        if missing_keywords:
            recommendations.append(f"Add missing keywords: {', '.join(missing_keywords[:3])}")
        
        # Add section-based recommendations
        if not sections['Projects']:
            recommendations.append("Add a projects section to showcase your practical experience")
        if not sections['Achievements']:
            recommendations.append("Include quantifiable achievements to demonstrate impact")
        
        # Add skill-based recommendations
        if match_data['score'] < 70:
            recommendations.append("Focus on developing skills that align with job requirements")
        
        # Add customized recommendations based on analysis
        if sections['Experience']:
            if not any('led' in exp.lower() or 'managed' in exp.lower() for exp in sections['Experience'].split('\n')):
                recommendations.append("Highlight leadership and management experience")
        
        if sections['Education']:
            if not any('certification' in edu.lower() for edu in sections['Education'].split('\n')):
                recommendations.append("Consider adding relevant certifications")
        
        # Add general recommendations
        recommendations.extend([
            "Ensure all experiences are quantified with metrics",
            "Keep resume format ATS-friendly"
        ])
        
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
            "Recommendations": recommendations
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
                
                # Display key strengths
                st.markdown("#### Key Strengths")
                if ats_response['Key Strengths']:
                    for strength in ats_response['Key Strengths']:
                        st.success(f"✓ {strength}")
                else:
                    st.warning("No key strengths identified")
                
                # Display missing keywords
                st.markdown("#### Missing Keywords")
                if ats_response['Missing Keywords']:
                    for keyword in ats_response['Missing Keywords']:
                        st.error(f"✗ {keyword}")
                else:
                    st.success("No critical missing keywords")
                
                # Display category matches
                st.markdown("#### Category Matches")
                for category, score in ats_response['Category Matches'].items():
                    color = 'green' if score >= 80 else 'orange' if score >= 60 else 'red'
                    st.markdown(f"<span style='color:{color}'>{category}: {score}%</span>", unsafe_allow_html=True)
            
            # Tab 3: Recommendations
            with tab3:
                st.markdown("### Personalized Recommendations")
                
                # Resume Structure Analysis
                with st.expander("Resume Structure Evaluation", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if ats_response['Projects']:
                            st.success("✔ Strong projects section")
                            st.markdown(f"• {len(ats_response['Projects'])} relevant projects listed")
                        else:
                            st.error("✘ Missing projects section")
                    
                    with col2:
                        if ats_response['Achievements']:
                            st.success("✔ Achievements well highlighted")
                            st.markdown(f"• {len(ats_response['Achievements'])} quantifiable achievements")
                        else:
                            st.error("✘ Missing achievements section")
                    
                    st.info("💡 Structure Improvement Tips:")
                    st.markdown("""
                    - **Bullet points**: Use for readability (3-5 per section)
                    - **Length**: Keep to 1-2 pages maximum
                    - **Action verbs**: Use strong verbs (developed, optimized, led)
                    - **Metrics**: Quantify achievements (e.g., "Improved performance by 30%")
                    - **White space**: Ensure proper spacing between sections
                    """)
                
                # Skill Development Recommendations
                with st.expander("Skill Enhancement", expanded=True):
                    if ats_response['Missing Keywords']:
                        st.error("🔍 Key Skills to Develop:")
                        for keyword in ats_response['Missing Keywords'][:5]:
                            st.markdown(f"- {keyword}")
                        
                        st.info("📚 Recommended Learning Resources:")
                        st.markdown("""
                        - [FreeCodeCamp](https://www.freecodecamp.org/) - Free coding tutorials
                        - [Coursera](https://www.coursera.org/) - Professional certificates  
                        - [Udemy](https://www.udemy.com/) - Affordable courses
                        - [LinkedIn Learning](https://www.linkedin.com/learning/) - Career-focused skills
                        """)
                    else:
                        st.success("🎯 Excellent skill match with job requirements!")
                
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
                if ats_response['Recommendations']:
                    st.markdown("### Action Items")
                    for i, rec in enumerate(ats_response['Recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
    else:
        st.warning("Please upload a resume and enter a job description.")