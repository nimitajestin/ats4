import streamlit as st
import os                  
import PyPDF2 as pdf      
import json               
import re                 
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
    'under', 'again', 'further', 'then', 'once'
}

# Enhanced skill categories with weighted importance
SKILL_CATEGORIES = {
    'Programming Languages': {
        'weight': 1.5,
        'skills': ['python', 'java', 'javascript', 'c++', 'go', 'rust']
    },
    'Web Technologies': {
        'weight': 1.3,
        'skills': ['react', 'django', 'spring', 'node.js', 'flask']
    },
    'Database': {
        'weight': 1.2,
        'skills': ['sql', 'mongodb', 'postgresql', 'redis']
    },
    'DevOps': {
        'weight': 1.4,
        'skills': ['docker', 'kubernetes', 'aws', 'azure', 'ci/cd']
    },
    'Data Science': {
        'weight': 1.3,
        'skills': ['machine learning', 'deep learning', 'nlp', 'pandas', 'tensorflow']
    },
    'Soft Skills': {
        'weight': 0.8,
        'skills': ['leadership', 'communication', 'teamwork', 'problem solving']
    }
}

EDUCATION_TERMS = [

    'bachelor', 'master', 'phd', 'degree', 
    'b.tech', 'b.e', 'm.tech', 'bsc', 'msc',
    'computer science', 'cs', 'information technology', 'it',
    'engineering', 'technology', 'computer engineering',
    'software engineering', 'information systems',
    'artificial intelligence', 'data science',
    'university', 'college', 'institute',
    'gpa', 'cgpa', 'grade', 'honors', 'distinction'
]

def extract_skills_and_keywords(text):
    """Enhanced skill extraction with resume-specific analysis"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    
    # Initialize categories with empty dicts
    categorized_skills = {category: {} for category in SKILL_CATEGORIES}
    
    # First pass - identify mentioned technologies
    for category, data in SKILL_CATEGORIES.items():
        for skill in data['skills']:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                # Find all occurrences with context
                matches = list(re.finditer(r'\b' + re.escape(skill) + r'\b', text))
                contexts = [text[max(0,m.start()-50):min(len(text),m.end()+50)] for m in matches]
                
                # Calculate proficiency score (0-1)
                score = 0.5  # base
                
                # Boost score based on context indicators
                for ctx in contexts:
                    if any(w in ctx for w in ['expert', 'proficient', 'experienced']):
                        score = max(score, 1.0)
                    elif any(w in ctx for w in ['knowledge', 'familiar']):
                        score = max(score, 0.7)
                    elif any(w in ctx for w in ['project', 'built', 'developed']):
                        score = max(score, 0.9)
                
                # Boost if mentioned multiple times
                if len(matches) > 1:
                    score = min(1.0, score + 0.1 * len(matches))
                    
                categorized_skills[category][skill] = score
    
    # Extract keywords
    stop_words = STOP_WORDS
    words = tokenize_text(text)
    keywords = [word for word in words if word not in stop_words]
    
    return keywords, categorized_skills

def calculate_match_percentage(resume_text, jd_text):
    try:
        resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
        
        # Calculate category matches
        category_scores = {}
        
        for category in SKILL_CATEGORIES:
            jd_skills = jd_categories.get(category, {})
            resume_skills = resume_categories.get(category, {})
            
            if not jd_skills:  # Skip if JD doesn't require this category
                continue
                
            # Calculate match percentage for this category
            matched_score = sum(
                min(jd_skills[skill], resume_skills[skill]) 
                for skill in jd_skills if skill in resume_skills
            )
            
            total_score = sum(jd_skills.values())
            
            if total_score > 0:
                match_percent = (matched_score / total_score) * 100
                category_scores[category] = round(match_percent, 1)
        
        # Calculate overall score (weighted average)
        if not category_scores:
            return {'score': 0, 'category_scores': {}}
            
        overall_score = sum(
            score * SKILL_CATEGORIES[category]['weight'] 
            for category, score in category_scores.items()
        ) / sum(
            SKILL_CATEGORIES[category]['weight'] 
            for category in category_scores
        )
        
        return {
            'score': round(overall_score, 1),
            'category_scores': category_scores
        }
        
    except Exception as e:
        print(f"Error in calculate_match_percentage: {str(e)}")
        return {'score': 0, 'category_scores': {}}

def extract_pdf_text(uploaded_file):
    
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def analyze_education(text):
    text_lower = text.lower()
    
    sentences = text_lower.split('.')
    edu_sentences = [s.strip() for s in sentences if any(term in s for term in EDUCATION_TERMS)]
    
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
            if category in ['Programming Languages', 'Web Technologies']:
                rec = f"Develop {category.lower()} skills: Consider learning {gaps[0]} through online courses " \
                      f"or projects. Resources: " \
                      f"{'Udemy' if 'python' in gaps[0].lower() else 'Codecademy' if 'javascript' in gaps[0].lower() else 'Coursera'}"
            elif category == 'Database':
                rec = f"Gain {gaps[0]} experience: Try building a small project using {gaps[0]} " \
                      f"with a frontend framework you know."
            elif category == 'DevOps':
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

@st.cache_data 
def get_ats_feedback(resume_text, jd_text):
    try:
        # Initialize all variables with defaults
        profile_summary = ""
        matched_keywords = []
        missing_keywords = []
        category_scores = {}
        skill_gaps = {}
        matched_skills = {}
        recommendations = []
        
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
            
            # Rest of your analysis code...
            match_percentage = calculate_match_percentage(resume_text, jd_text)
            
            education = analyze_education(resume_text)
            experience = analyze_experience(resume_text)
            
            if education:
                profile_summary = education.strip().capitalize()
            if experience:
                if profile_summary:
                    profile_summary += " | "
                profile_summary += experience
            
            # Skill category analysis...
            
        return {
            "JD Match": f"{match_percentage['score']}%" if 'match_percentage' in locals() else "0%",
            "Profile Summary": profile_summary,
            "Key Strengths": [kw[0] for kw in matched_keywords[:5]],
            "Missing Keywords": [kw[0] for kw in missing_keywords[:5]],
            "Education": analyze_education(resume_text) if 'resume_text' in locals() else "No education details found",
            "Experience": analyze_experience(resume_text) if 'resume_text' in locals() else "No experience details found",
            "Projects": analyze_projects(resume_text)[:3] if 'resume_text' in locals() and analyze_projects(resume_text) else [],
            "Achievements": analyze_achievements(resume_text)[:3] if 'resume_text' in locals() and analyze_achievements(resume_text) else [],
            "Category Matches": match_percentage['category_scores'],
            "Skill Gaps": skill_gaps,
            "Matched Skills": matched_skills,
            "Recommendations": generate_recommendations({"Projects": analyze_projects(resume_text), "Achievements": analyze_achievements(resume_text)}, jd_text, matched_skills, skill_gaps)
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
    # Check if both inputs are provided
    if uploaded_resume and jd_input.strip():
        # Show loading spinner while processing
        with st.spinner("Analyzing Resume..."):
            resume_text = extract_pdf_text(uploaded_resume)
            ats_response = get_ats_feedback(resume_text, jd_input)
            
            if ats_response:
                st.markdown("---")
                st.markdown("###  ATS Evaluation Results")
        
        # Handle case where response is already parsed or needs parsing
        if isinstance(ats_response, str):
            results = json.loads(ats_response)
        else:
            results = ats_response
        
        # Display match percentage with color coding
        # Green: ≥80%, Orange: ≥60%, Red: <60%
        match_pct = float(results.get('JD Match', '0%').strip('%'))
        color = 'green' if match_pct >= 80 else 'orange' if match_pct >= 60 else 'red'
        st.markdown(f"<h2 style='color: {color}; text-align: center;'>Overall Match: {results.get('JD Match', 'N/A')}</h2>", unsafe_allow_html=True)
        
        # Create three tabs for organized results display
        tab1, tab2, tab3 = st.tabs(["Overview", "Skills Analysis", "Recommendations"])
        
        # Tab 1: Overview - Display basic profile information
        with tab1:
            # Show profile summary
            st.markdown("### Profile Summary")
            st.info(results.get('Profile Summary', ''))
            
            # Display education and experience in two columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Education")
                st.write(results.get('Education', ''))
            with col2:
                st.markdown("### Experience")
                st.write(results.get('Experience', ''))
            
            # Display projects and achievements if available
            if results.get('Projects', []) or results.get('Achievements', []):
                st.markdown("### Key Highlights")
                
                # Show top 3 projects
                if results.get('Projects', []):
                    st.markdown("#### Notable Projects")
                    for project in results['Projects'][:3]:
                        st.markdown(f"* {project.capitalize()}")
                
                # Show top 3 achievements
                if results.get('Achievements', []):
                    st.markdown("#### Key Achievements")
                    for achievement in results['Achievements'][:3]:
                        st.markdown(f"* {achievement.capitalize()}")
        
        # Tab 2: Skills Analysis - Show detailed skill matching and gaps
        with tab2:
            st.markdown("### Skills Analysis")
            
            # Enhanced skill categories display
            if 'Category Matches' in results:
                for category in ['Technical', 'Soft', 'Domain']:
                    match_pct = results['Category Matches'].get(category, 0)
                    progress_color = 'green' if match_pct >= 80 else 'orange' if match_pct >= 60 else 'red'
                    
                    # Create expandable section for each category
                    with st.expander(f"{category} Skills - {match_pct}% Match", expanded=True):
                        # Progress bar with match details
                        st.progress(match_pct/100)
                        
                        # Show matched skills if available
                        if 'Matched Skills' in results and category in results['Matched Skills']:
                            st.markdown(f"**Your strong {category.lower()} skills:**")
                            cols = st.columns(3)
                            for i, skill in enumerate(results['Matched Skills'][category][:6]):
                                cols[i%3].success(f" {skill}")
                        
                        # Show missing skills if available
                        if 'Skill Gaps' in results and category in results['Skill Gaps'] and results['Skill Gaps'][category]:
                            st.markdown(f"**Recommended {category.lower()} skills to add:**")
                            for skill in results['Skill Gaps'][category][:5]:
                                st.error(f"- {skill}")
            
            # Strengths vs Areas to Improve
            st.markdown("### Strengths vs Areas for Improvement")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Your Key Strengths")
                for strength in results.get('Key Strengths', [])[:5]:
                    st.success(f" {strength}")
            with col2:
                st.markdown("#### Priority Areas")
                for keyword in results.get('Missing Keywords', [])[:5]:
                    st.error(f" {keyword}")
        
        # Tab 3: Recommendations - Enhanced actionable feedback
        with tab3:
            st.markdown("### Personalized Recommendations")
            
            # Resume Structure Recommendations
            with st.expander("Resume Structure", expanded=True):
                if results.get('Recommendations', []):
                    for i, rec in enumerate(results['Recommendations'][:3], 1):
                        st.markdown(f"{i}. {rec}")
                else:
                    st.info("No specific structure recommendations available")
            
            # Skill Development Plan
            with st.expander("Skill Development Plan", expanded=True):
                if 'Skill Gaps' in results:
                    st.markdown("**Focus on developing these skills:**")
                    for category in ['Technical', 'Soft', 'Domain']:
                        if category in results['Skill Gaps'] and results['Skill Gaps'][category]:
                            st.markdown(f"**{category}:** {', '.join(results['Skill Gaps'][category][:3])}")
                else:
                    st.success("Your skills match well with the job requirements!")
            
            # General Tips
            with st.expander("General Resume Tips", expanded=True):
                tips = [
                    "Use strong action verbs (e.g., 'developed', 'managed', 'optimized')",
                    "Quantify achievements with numbers where possible",
                    "Keep resume to 1-2 pages maximum",
                    "Use consistent formatting throughout",
                    "Tailor your resume for each job application"
                ]
                for tip in tips:
                    st.markdown(f"* {tip}")
    # Show warning if inputs are missing
    else:
        st.warning("Please upload a resume and enter a job description.")
