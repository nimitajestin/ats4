import PyPDF2
import streamlit as st
import os                  
import re                 
import json               
from collections import Counter  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity      

# Text processing utilities
def clean_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces and newlines with single space
    text = re.sub(r'[\s\n]+', ' ', text)
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()

def tokenize_text(text):
    """Split text into words"""
    text = clean_text(text)
    # Split into words (alphanumeric sequences)
    words = re.findall(r'\b[a-z0-9]+\b', text)
    return [w for w in words if w and len(w) > 1]  # Filter out single characters

def split_into_sentences(text):
    """Split text into sentences"""
    # First clean up obvious sentence boundaries
    text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1\n\2', text)
    # Split on newlines and filter empty strings
    sentences = [s.strip() for s in text.split('\n')]
    return [s for s in sentences if s]

# Common English stop words
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

# Enhanced domain-focused skill categories
DOMAIN_CATEGORIES = {
    'Core Technical': {
        'weight': 1.3,
        'skills': ['python', 'java', 'sql', 'aws', 'docker', 'react', 'django'],
        'context_boosters': ['built', 'developed', 'implemented', 'optimized']
    },
    'Industry Expertise': {
        'weight': 1.7,  # Highest weight for domain knowledge
        'skills': [
            # Tech
            'cloud architecture', 'devops', 'microservices', 'api design',
            # Finance
            'financial modeling', 'risk management', 'investment analysis',
            # Healthcare
            'health informatics', 'fda compliance', 'clinical systems',
            # Product
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
    """Advanced skill extraction with contextual scoring"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Better normalization
    
    skills_data = {category: {'skills': {}, 'context_count': 0} 
                  for category in DOMAIN_CATEGORIES}
    
    # Multi-pass analysis for better accuracy
    for category, data in DOMAIN_CATEGORIES.items():
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, data['skills'])) + r')\b')
        
        for match in pattern.finditer(text):
            skill = match.group()
            context = text[max(0, match.start()-100):min(len(text), match.end()+100)]
            
            # Contextual scoring
            base_score = 0.7
            if any(boost in context for boost in data['context_boosters']):
                base_score = min(1.0, base_score + 0.3)
            
            # Multiple mentions boost
            if skill in skills_data[category]['skills']:
                base_score = min(1.0, skills_data[category]['skills'][skill] + 0.15)
            
            skills_data[category]['skills'][skill] = base_score
            skills_data[category]['context_count'] += 1
    
    # Extract meaningful keywords (4+ chars, not stopwords)
    keywords = [word for word in re.findall(r'\b\w{4,}\b', text) 
               if word not in STOP_WORDS and not word.isdigit()]
    
    return keywords, {cat: data['skills'] for cat, data in skills_data.items()}

def calculate_match_percentage(resume_text, jd_text):
    try:
        resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
        
        # Calculate category matches
        category_scores = {}
        
        for category in DOMAIN_CATEGORIES:
            jd_skills = jd_categories.get(category, {})
            resume_skills = resume_categories.get(category, {})
            
            if not jd_skills:  # Skip if JD doesn't require this category
                continue
                
            # Calculate match percentage for this category
            matched_score = sum(
                min(jd_skills[skill], resume_skills.get(skill, 0)) 
                for skill in jd_skills
            )
            
            total_score = sum(jd_skills.values())
            
            if total_score > 0:
                match_percent = (matched_score / total_score) * 100
                category_scores[category] = max(1, round(match_percent))  # Ensure at least 1% if any match
            
        # Calculate overall score (weighted average)
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
    """Handle Streamlit UploadedFile objects"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return extract_resume_sections(text)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def preprocess_text(text):
    """Enhanced text preprocessing for better analysis"""
    # Preserve special characters that might indicate skills/qualifications
    text = re.sub(r'([a-z])\s*[/&]\s*([a-z])', r'\1/\2', text.lower())  # Preserve skill combinations
    
    # Handle bullet points and special formatting
    text = re.sub(r'•|\u2022', '*', text)  # Standardize bullet points
    
    # Remove unwanted characters but preserve meaningful symbols
    text = re.sub(r'[^a-z0-9\s*&+\-/,]', ' ', text)
    
    # Normalize whitespace while preserving list structures
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_education(text):
    text_lower = text.lower()
    
    sentences = text_lower.split('.')
    edu_sentences = [s.strip() for s in sentences if any(term in s for term in ['bachelor', 'master', 'phd', 'degree', 'b.tech', 'b.e', 'm.tech', 'bsc', 'msc', 'computer science', 'cs', 'information technology', 'it', 'engineering', 'technology', 'computer engineering', 'software engineering', 'information systems', 'artificial intelligence', 'data science', 'university', 'college', 'institute', 'gpa', 'cgpa', 'grade', 'honors', 'distinction'])]
    
    is_cs = any(term in text_lower for term in ['computer science', 'cs', 'information technology', 'it', 'software engineering'])
    
   
    if edu_sentences:
        main_edu = edu_sentences[0]  # Take the first education-related sentence
    
        if is_cs and 'computer science' not in main_edu and 'cs' not in main_edu:
            main_edu += ' (Computer Science/IT background)'
        return main_edu
    
    if is_cs:
        return "Computer Science/IT background detected"
    
    return ""

def analyze_experience(text):
    text_lower = text.lower()
    sentences = [s.strip() for s in text_lower.split('.')]

    experiences = {
        'work': [],
        'projects': [],
        'internships': []
    }
    
    for sentence in sentences:
        if any(word in sentence for word in ['experience', 'worked', 'working']):
            experiences['work'].append(sentence)
        elif any(word in sentence for word in ['project', 'developed', 'built', 'created', 'implemented']):
            experiences['projects'].append(sentence)
        elif 'intern' in sentence:
            experiences['internships'].append(sentence)
    
    years_exp = 0
    for sentence in experiences['work']:
        year_matches = re.findall(r'\d+\+?\s*(?:year|yr)', sentence)  # Match patterns like "5+ years" or "3 yr"
        if year_matches:
            try:
                years_exp = max(years_exp, int(re.findall(r'\d+', year_matches[0])[0]))
            except:
                pass
    
    experience_summary = []
    if years_exp > 0:
        experience_summary.append(f"{years_exp}+ years of experience")
    
    if experiences['work']:
        experience_summary.append(experiences['work'][0])
    if experiences['internships']:
        experience_summary.append(f"Has internship experience: {experiences['internships'][0]}")
    if experiences['projects']:
        experience_summary.append(f"Project experience: {experiences['projects'][0]}")
    
    return ' | '.join(experience_summary) if experience_summary else ""

def get_key_strengths(resume_keywords, jd_keywords):
    strengths = list(set(resume_keywords) & set(jd_keywords))
    return sorted(strengths, key=lambda x: len(x), reverse=True)[:5]

def analyze_projects(text):
    """Analyze project information from text"""
    sentences = split_into_sentences(text)
    project_words = ['project', 'developed', 'built', 'created', 'implemented']
    projects = [s for s in sentences if any(word in s.lower() for word in project_words)]
    return projects

def analyze_achievements(text):
    """Analyze achievement information from text"""
    sentences = split_into_sentences(text)
    achievement_words = [
        'achieved', 'awarded', 'won', 'recognized', 'selected', 'ranked', 'improved',
        'increased', 'decreased', 'reduced', 'saved', 'delivered', 'led', 'managed'
    ]
    achievements = [s for s in sentences if any(word in s.lower() for word in achievement_words)]
    return achievements

def generate_recommendations(resume_data, jd_data, matched_skills, skill_gaps):
    """Generate personalized recommendations based on resume analysis"""
    recommendations = []
    
    # 1. Skill development recommendations
    for category, gaps in skill_gaps.items():
        if gaps:
            if category in ['Core Technical', 'Applied Skills']:
                rec = f"Develop {category.lower()} skills: Consider learning {gaps[0]} through online courses " \
                      f"or projects. Resources: " \
                      f"{'Udemy' if 'python' in gaps[0].lower() else 'Codecademy' if 'javascript' in gaps[0].lower() else 'Coursera'}"
            elif category == 'Industry Expertise':
                rec = f"Gain {gaps[0]} experience: Try building a small project using {gaps[0]} " \
                      f"with a frontend framework you know."
            elif category == 'Leadership':
                rec = f"Learn {gaps[0]}: Set up a CI/CD pipeline or containerize an existing project " \
                      f"to gain hands-on experience."
            else:
                rec = f"Improve {category.lower()} skills: Focus on developing {gaps[0]}"
            recommendations.append(rec)
    
    # 2. Project suggestions based on existing skills
    strong_skills = [skill for skills in matched_skills.values() for skill in skills]
    if strong_skills:
        if 'python' in strong_skills and 'django' in strong_skills:
            recommendations.append("Build a Django web app to showcase your full-stack Python skills")
        if 'javascript' in strong_skills and 'react' in strong_skills:
            recommendations.append("Create a React portfolio project to demonstrate modern frontend skills")
    
    # 3. Resume improvement tips
    if not resume_data.get('Projects', []):
        recommendations.append("Add projects section: Include 2-3 relevant projects with technologies used")
    if not resume_data.get('Achievements', []):
        recommendations.append("Highlight achievements: Quantify your impact (e.g. 'Improved performance by X%')")
    
    return recommendations[:5]  # Return top 5 most relevant recommendations

def extract_resume_sections(text):
    """Extract resume sections with comprehensive error handling"""
    # Initialize default sections
    sections = {
        'Profile Summary': '',
        'Education': '', 
        'Experience': '',
        'Projects': [],
        'Achievements': []
    }
    
    # Validate input
    if not text or not isinstance(text, (str, bytes)):
        return sections
    
    try:
        # Ensure we have a string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
            
        # Normalize text safely
        text = str(text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Section patterns
        section_patterns = {
            'Profile Summary': [r'SUMMARY', r'PROFILE', r'ABOUT', r'OBJECTIVE'],
            'Education': [r'EDUCATION', r'ACADEMIC', r'QUALIFICATION'],
            'Experience': [r'EXPERIENCE', r'WORK HISTORY', r'EMPLOYMENT'],
            'Projects': [r'PROJECTS', r'PROJECT EXPERIENCE'],
            'Achievements': [r'ACHIEVEMENTS', r'ACCOMPLISHMENTS', r'HONORS']
        }
        
        current_section = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            for section, patterns in section_patterns.items():
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                    current_section = section
                    break
            
            # Add content to current section
            if current_section:
                if current_section in ['Projects', 'Achievements']:
                    if line and line not in sections[current_section]:
                        sections[current_section].append(line)
                else:
                    sections[current_section] += f"{line}\n"
        
        # Clean up sections
        for section in sections:
            if isinstance(sections[section], str):
                sections[section] = sections[section].strip()
        
    except Exception as e:
        st.warning(f"Section extraction warning: {str(e)}")
    
    return sections

@st.cache_data 
def get_ats_feedback(resume_text, jd_text):
    try:
        # Initialize match percentage with a default value
        match_percentage = {'score': 0, 'category_scores': {}}
        
        # Initialize other variables
        profile_summary = ""
        matched_keywords = []
        missing_keywords = []
        category_scores = {}
        skill_gaps = {}
        matched_skills = {}
        
        # Perform analysis only if inputs exist
        if resume_text and jd_text:
            resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
            jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
            
            # Calculate keyword matches
            matched_keywords = [(kw, jd_keywords.count(kw)) 
                              for kw in set(jd_keywords) if kw in resume_keywords]
            missing_keywords = [(kw, jd_keywords.count(kw)) 
                              for kw in set(jd_keywords) if kw not in resume_keywords]
            
            # Sort by frequency in JD (importance)
            matched_keywords.sort(key=lambda x: x[1], reverse=True)
            missing_keywords.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate match percentage
            match_percentage = calculate_match_percentage(resume_text, jd_text)
            
            education = analyze_education(resume_text)
            experience = analyze_experience(resume_text)
            
            if education:
                profile_summary = education.strip().capitalize()
            if experience:
                if profile_summary:
                    profile_summary += " | "
                profile_summary += experience
            
        return {
            "JD Match": f"{match_percentage['score']}%",
            "Profile Summary": profile_summary,
            "Key Strengths": [kw[0] for kw in matched_keywords[:5]],
            "Missing Keywords": [kw[0] for kw in missing_keywords[:5]],
            "Education": analyze_education(resume_text),
            "Experience": analyze_experience(resume_text),
            "Projects": analyze_projects(resume_text)[:3] if analyze_projects(resume_text) else [],
            "Achievements": analyze_achievements(resume_text)[:3] if analyze_achievements(resume_text) else [],
            "Category Matches": match_percentage['category_scores'],
            "Skill Gaps": skill_gaps,
            "Matched Skills": matched_skills,
            "Recommendations": generate_recommendations(
                {"Projects": analyze_projects(resume_text), "Achievements": analyze_achievements(resume_text)},
                jd_text,
                matched_skills,
                skill_gaps
            )
        }
    except Exception as e:
        st.error(f"Error in text processing: {str(e)}")
        return None

# Streamlit App Interface

# Display the application header with custom styling
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #1f497d;">Verq ATS Resume Evaluator</h1>
        <p>Upload your resume and job description to get instant feedback</p>
    </div>
    """, unsafe_allow_html=True)

# Add application description
st.markdown("##  ATS Resume Evaluator")
st.markdown("Upload your resume and job description to receive a tailored match percentage, keyword analysis, and improvement suggestions.")

# Create a two-column layout for inputs
with st.container():
    col1, col2 = st.columns(2)

    # Left column: Job Description input
    with col1:
        jd_input = st.text_area(" Job Description", height=300, placeholder="Paste the JD here...")

    # Right column: Resume upload
    with col2:
        uploaded_resume = st.file_uploader(" Upload Resume (PDF)", type=["pdf"])

# Evaluate button and results display
if st.button(" Evaluate"):
    if uploaded_resume and jd_input.strip():
        with st.spinner("Analyzing Resume..."):
            # Extract text from PDF
            resume_text = ""
            try:
                reader = PyPDF2.PdfReader(uploaded_resume)
                for page in reader.pages:
                    resume_text += page.extract_text() + "\n"
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                st.stop()

            if not resume_text.strip():
                st.error("Could not extract text from the PDF")
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

            # Create three tabs for organized results display
            tab1, tab2, tab3 = st.tabs(["Overview", "Skills Analysis", "Recommendations"])

            # Tab 1: Overview - Display basic profile information
            with tab1:
                # Show profile summary
                st.markdown("### Profile Summary")
                if ats_response['Profile Summary']:
                    st.info(ats_response['Profile Summary'])
                else:
                    st.warning("No profile summary found")

                # Display education and experience in two columns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Education")
                    if ats_response['Education']:
                        st.write(ats_response['Education'])
                    else:
                        st.warning("No education details found")

                with col2:
                    st.markdown("### Experience")
                    if ats_response['Experience']:
                        st.write(ats_response['Experience'])
                    else:
                        st.warning("No experience details found")

                # Display projects and achievements
                st.markdown("### Key Highlights")
                
                # Show projects
                if ats_response['Projects']:
                    st.markdown("#### Notable Projects")
                    for project in ats_response['Projects']:
                        st.markdown(f"* {project}")
                else:
                    st.warning("No projects found")

                # Show achievements
                if ats_response['Achievements']:
                    st.markdown("#### Key Achievements")
                    for achievement in ats_response['Achievements']:
                        st.markdown(f"* {achievement}")
                else:
                    st.warning("No achievements found")

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
                st.markdown("### Recommendations")
                if ats_response['Recommendations']:
                    for i, rec in enumerate(ats_response['Recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
                else:
                    st.info("No specific recommendations at this time")

    else:
        st.warning("Please upload a resume and enter a job description.")
