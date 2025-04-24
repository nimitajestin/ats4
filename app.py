import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Verq ATS Evaluator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

import os
import PyPDF2 as pdf
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Skill categories for better analysis
SKILL_CATEGORIES = {
    'Programming Languages': ['python', 'java', 'javascript', 'js', 'typescript', 'ts', 'c++', 'c#', 'csharp', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab', 'c', 'cpp'],
    'Web Technologies': ['html', 'html5', 'css', 'css3', 'react', 'reactjs', 'angular', 'vue', 'nodejs', 'node.js', 'django', 'flask', 'express', 'jquery', 'bootstrap', 'sass', 'less', 'webpack', 'vite', 'nextjs', 'graphql', 'rest api', 'restful'],
    'Database': ['sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'mongo', 'oracle', 'redis', 'elasticsearch', 'dynamodb', 'firebase', 'cassandra', 'mariadb', 'sqlite', 'nosql'],
    'Cloud & DevOps': ['aws', 'amazon', 'azure', 'microsoft azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s', 'jenkins', 'terraform', 'ci/cd', 'cicd', 'git', 'github', 'gitlab', 'bitbucket', 'linux', 'unix', 'bash', 'shell'],
    'Data Science': ['machine learning', 'ml', 'deep learning', 'dl', 'nlp', 'natural language processing', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras', 'computer vision', 'cv', 'ai', 'artificial intelligence', 'data mining', 'statistics'],
    'Soft Skills': ['leadership', 'communication', 'teamwork', 'team player', 'problem solving', 'analytical', 'project management', 'agile', 'scrum', 'time management', 'collaboration', 'critical thinking', 'attention to detail', 'multitasking']
}

# Education related terms
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

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('punkt')
    except LookupError:
        with st.spinner('Downloading required language data (punkt)...'):
            nltk.download('punkt')
    
    try:
        nltk.data.find('stopwords')
    except LookupError:
        with st.spinner('Downloading required language data (stopwords)...'):
            nltk.download('stopwords')

# Download NLTK data
download_nltk_data()

# Function to get response from OpenAI's GPT model
@st.cache_data
def extract_skills_and_keywords(text):
    # Convert to lowercase
    text = text.lower()
    
    # First, look for exact matches of multi-word skills
    categorized_skills = {category: [] for category in SKILL_CATEGORIES}
    
    # Look for exact matches first (especially for multi-word terms)
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if ' ' in skill:  # Multi-word skill
                if skill in text:
                    categorized_skills[category].append(skill)
    
    # Now process individual words
    # Split on whitespace and remove punctuation
    words = [word.strip('.,!?()[]{}:;"\'') for word in text.split()]
    words = [word for word in words if word]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Extract common technical terms and skills (2-gram phrases)
    phrases = []
    for i in range(len(words)-1):
        phrase = f"{words[i]} {words[i+1]}"
        phrases.append(phrase)
    
    # Combine single words and phrases
    all_terms = words + phrases
    
    # Get terms by frequency
    term_freq = Counter(all_terms)
    terms = [term for term, freq in term_freq.most_common(50)]
    
    # Add single-word skills to categories
    for category, skills in SKILL_CATEGORIES.items():
        single_word_matches = [term for term in terms 
                             if term in skills and term not in categorized_skills[category]]
        categorized_skills[category].extend(single_word_matches)
    
    # Remove empty categories
    categorized_skills = {k: v for k, v in categorized_skills.items() if v}
    
    return terms, categorized_skills

def calculate_match_percentage(resume_text, jd_text):
    # Use TF-IDF vectorization for better matching
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 1)
    except:
        return 50  # Default fallback value

# Function to extract text from uploaded PDF
def extract_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to get ATS feedback based on resume and job description
def analyze_education(text):
    text_lower = text.lower()
    
    # First look for education-related sentences
    sentences = text_lower.split('.')
    edu_sentences = [s.strip() for s in sentences if any(term in s for term in EDUCATION_TERMS)]
    
    # Look for CS/IT related education
    is_cs = any(term in text_lower for term in ['computer science', 'cs', 'information technology', 'it', 'software engineering'])
    
    # Get the most relevant education sentence
    if edu_sentences:
        main_edu = edu_sentences[0]
        # Add CS/IT indicator if found
        if is_cs and 'computer science' not in main_edu and 'cs' not in main_edu:
            main_edu += ' (Computer Science/IT background)'
        return main_edu
    
    # If no clear education sentence but CS/IT terms found
    if is_cs:
        return "Computer Science/IT background detected"
    
    return ""

def analyze_experience(text):
    text_lower = text.lower()
    sentences = [s.strip() for s in text_lower.split('.')]
    
    # Look for different types of experience
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
    
    # Determine experience level
    years_exp = 0
    for sentence in experiences['work']:
        year_matches = re.findall(r'\d+\+?\s*(?:year|yr)', sentence)
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
    # Find matching keywords (strengths)
    strengths = list(set(resume_keywords) & set(jd_keywords))
    return sorted(strengths, key=lambda x: len(x), reverse=True)[:5]

def analyze_projects(text):
    text_lower = text.lower()
    sentences = [s.strip() for s in text_lower.split('.')]
    
    projects = [s for s in sentences if any(word in s for word in ['project', 'developed', 'built', 'created', 'implemented'])]
    return projects

def analyze_achievements(text):
    text_lower = text.lower()
    sentences = [s.strip() for s in text_lower.split('.')]
    
    achievements = [s for s in sentences if any(word in s for word in 
        ['achieved', 'awarded', 'won', 'recognized', 'selected', 'ranked', 'improved', 
         'increased', 'decreased', 'reduced', 'saved', 'delivered', 'led', 'managed'])]
    return achievements

@st.cache_data
def get_ats_feedback(resume_text, jd_text):
    # Extract keywords from both texts
    resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
    jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
    
    # Find missing keywords and strengths
    missing_keywords = list(set(jd_keywords) - set(resume_keywords))
    key_strengths = get_key_strengths(resume_keywords, jd_keywords)
    
    # Calculate match percentage
    match_percentage = calculate_match_percentage(resume_text, jd_text)
    
    # Get detailed analysis
    education = analyze_education(resume_text)
    experience = analyze_experience(resume_text)
    projects = analyze_projects(resume_text)
    achievements = analyze_achievements(resume_text)
    
    # Generate a contextual profile summary
    profile_parts = []
    if education:
        profile_parts.append(education.strip().capitalize())
    if experience:
        profile_parts.append(experience)
    profile_summary = ' | '.join(profile_parts)
    
    # Analyze skill gaps by category
    skill_gaps = {}
    for category in SKILL_CATEGORIES:
        jd_skills = set(jd_categories.get(category, []))
        resume_skills = set(resume_categories.get(category, []))
        if jd_skills:
            skill_gaps[category] = list(jd_skills - resume_skills)
    
    # Calculate category-wise match percentages
    category_matches = {}
    for category in SKILL_CATEGORIES:
        jd_skills = set(jd_categories.get(category, []))
        resume_skills = set(resume_categories.get(category, []))
        if jd_skills:
            match = len(jd_skills & resume_skills) / len(jd_skills) * 100
            category_matches[category] = round(match, 1)
    
    # Generate personalized recommendations based on actual content
    recommendations = []
    
    # Education-based recommendations
    if not education:
        recommendations.append("Add your educational background prominently")
    elif 'computer science' in education.lower() or 'cs' in education.lower():
        recommendations.append("Your CS background is relevant - highlight any specialized coursework or projects")
    
    # Experience-based recommendations
    if not experience:
        recommendations.append("Add any internships, projects, or relevant work experience")
    elif 'internship' in experience.lower():
        recommendations.append("Quantify your internship achievements with specific metrics")
    elif any(str(i) in experience.lower() for i in range(1, 6)):
        recommendations.append("Highlight leadership roles and team contributions in your experience")
    
    # Project-based recommendations
    if not projects:
        recommendations.append("Add relevant projects showcasing your technical skills")
    elif len(projects) < 3:
        recommendations.append("Consider adding more projects demonstrating your expertise")
    
    # Achievement-based recommendations
    if not achievements:
        recommendations.append("Add quantifiable achievements and metrics to strengthen your impact")
    
    # Skill-based recommendations
    tech_categories = ['Programming Languages', 'Web Technologies', 'Database', 'Cloud & DevOps']
    missing_tech = [cat for cat in tech_categories if cat in skill_gaps and skill_gaps[cat]]
    
    if missing_tech:
        for category in missing_tech[:2]:  # Limit to top 2 categories
            gaps = skill_gaps[category][:3]  # Limit to top 3 skills
            if gaps:
                recommendations.append(f"Add {category} skills: {', '.join(gaps)}")
    
    # Format-based recommendations
    if len(resume_text.split()) < 200:
        recommendations.append("Your resume seems concise - consider adding more detail to your experiences")
    
    # Create detailed response
    response = {
        "JD Match": f"{match_percentage}%",
        "Profile Summary": profile_summary,
        "Key Strengths": key_strengths,
        "Missing Keywords": missing_keywords[:5],
        "Education": education.strip().capitalize() if education else "No education details found",
        "Experience": experience.strip().capitalize() if experience else "No experience details found",
        "Projects": projects[:3] if projects else [], 
        "Achievements": achievements[:3] if achievements else [], 
        "Category Matches": category_matches,
        "Skill Gaps": skill_gaps,
        "Recommendations": recommendations
    }
    
    return json.dumps(response, indent=2)

# Streamlit App

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
            resume_text = extract_pdf_text(uploaded_resume)
            ats_response = get_ats_feedback(resume_text, jd_input)
            
            if ats_response:
                st.markdown("---")
                st.markdown("###  ATS Evaluation Results")
        
        # Parse the JSON response
        results = json.loads(ats_response)
        
        
        match_pct = float(results['JD Match'].strip('%'))
        color = 'green' if match_pct >= 80 else 'orange' if match_pct >= 60 else 'red'
        st.markdown(f"<h2 style='color: {color}; text-align: center;'>Overall Match: {results['JD Match']}</h2>", unsafe_allow_html=True)
        
       
        tab1, tab2, tab3 = st.tabs(["Overview", "Skills Analysis", "Recommendations"])
        
        with tab1:
            
            st.markdown("### Profile Summary")
            st.info(results['Profile Summary'])
            
            # Education and Experience
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Education")
                st.write(results['Education'])
            with col2:
                st.markdown("### Experience")
                st.write(results['Experience'])
            
            # Projects and Achievements
            if results['Projects'] or results['Achievements']:
                st.markdown("### Key Highlights")
                
                if results['Projects']:
                    st.markdown("#### Notable Projects")
                    for project in results['Projects']:
                        st.markdown(f"* {project.capitalize()}")
                
                if results['Achievements']:
                    st.markdown("#### Key Achievements")
                    for achievement in results['Achievements']:
                        st.markdown(f"* {achievement.capitalize()}")
        
        with tab2:
            
            st.markdown("### Skills by Category")
            for category, match in results['Category Matches'].items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress_color = 'green' if match >= 80 else 'orange' if match >= 60 else 'red'
                    st.markdown(f"**{category}**")
                    st.progress(match/100)
                with col2:
                    st.markdown(f"<h4 style='color: {progress_color}'>{match}%</h4>", unsafe_allow_html=True)
                
                # Show skill gaps if any
                if category in results['Skill Gaps'] and results['Skill Gaps'][category]:
                    st.caption(f"Missing: {', '.join(results['Skill Gaps'][category])}")
            
            # Key Strengths and Missing Keywords
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Key Strengths")
                for strength in results['Key Strengths']:
                    st.markdown(f"+ {strength}")
            with col2:
                st.markdown("### Areas to Add")
                for keyword in results['Missing Keywords']:
                    st.markdown(f"- {keyword}")
        
        with tab3:
            
            st.markdown("### Detailed Recommendations")
            for i, rec in enumerate(results['Recommendations'], 1):
                st.markdown(f"{i}. {rec}")
            
            # Additional Tips
            st.markdown("### Pro Tips")
            st.info("""
            - Use industry-standard section headings
            - Include relevant certifications
            - Highlight achievements with metrics
            - Keep formatting simple and consistent
            """)
    else:
        st.warning("Please upload a resume and enter a job description.")
