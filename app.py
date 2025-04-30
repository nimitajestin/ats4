import streamlit as st

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Verq ATS Evaluator", 
    layout="centered",               
    initial_sidebar_state="expanded"  
)

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
from nltk.tokenize import word_tokenize, sent_tokenize  # For breaking text into words and sentences
from nltk.corpus import stopwords                       # For removing common words (e.g., 'the', 'is', 'at')


SKILL_CATEGORIES = {
    'Programming Languages': ['python', 'java', 'javascript', 'js', 'typescript', 'ts', 'c++', 'c#', 'csharp', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab', 'c', 'cpp'],
    
    'Web Technologies': ['html', 'html5', 'css', 'css3', 'react', 'reactjs', 'angular', 'vue', 'nodejs', 'node.js', 'django', 'flask', 'express', 'jquery', 'bootstrap', 'sass', 'less', 'webpack', 'vite', 'nextjs', 'graphql', 'rest api', 'restful'],
    
    'Database': ['sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'mongo', 'oracle', 'redis', 'elasticsearch', 'dynamodb', 'firebase', 'cassandra', 'mariadb', 'sqlite', 'nosql'],
    
 
    'Cloud & DevOps': ['aws', 'amazon', 'azure', 'microsoft azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s', 'jenkins', 'terraform', 'ci/cd', 'cicd', 'git', 'github', 'gitlab', 'bitbucket', 'linux', 'unix', 'bash', 'shell'],
    
 
    'Data Science': ['machine learning', 'ml', 'deep learning', 'dl', 'nlp', 'natural language processing', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras', 'computer vision', 'cv', 'ai', 'artificial intelligence', 'data mining', 'statistics'],
    
    'Soft Skills': ['leadership', 'communication', 'teamwork', 'team player', 'problem solving', 'analytical', 'project management', 'agile', 'scrum', 'time management', 'collaboration', 'critical thinking', 'attention to detail', 'multitasking']
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

download_nltk_data()

@st.cache_data
def extract_skills_and_keywords(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    
    categorized_skills = {category: {} for category in SKILL_CATEGORIES}
    
    def find_skill_variations(skill):
        variations = [skill]
        skill_variations = {
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'cpp': 'c++',
            'react': 'reactjs',
            'vue': 'vuejs',
            'node': 'nodejs',
            'aws': 'amazon web services',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'db': 'database',
            'ui': 'user interface',
            'ux': 'user experience',
            'api': 'application programming interface'
        }
        if skill in skill_variations:
            variations.append(skill_variations[skill])
        for abbr, full in skill_variations.items():
            if skill == full:
                variations.append(abbr)
        return variations
    
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            variations = find_skill_variations(skill)
            for variation in variations:
                pattern = r'\b' + re.escape(variation) + r'\b'
                matches = re.finditer(pattern, text)
                for match in matches:
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]
                    
                    confidence = 0.8  # Base confidence
                    tech_indicators = ['developed', 'implemented', 'built', 'created', 'designed', 'managed', 'led']
                    if any(indicator in context for indicator in tech_indicators):
                        confidence = 0.9
                    if any(tech in context for tech in ['project', 'application', 'system', 'software']):
                        confidence = 1.0
                    
                    categorized_skills[category][skill] = max(
                        confidence,
                        categorized_skills[category].get(skill, 0)
                    )
    
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    
    bigrams = [' '.join(pair) for pair in zip(words[:-1], words[1:])]
    trigrams = [' '.join(triple) for triple in zip(words[:-2], words[1:-1], words[2:])]
    
    all_terms = words + bigrams + trigrams
    term_freq = Counter(all_terms)
    
    terms = [term for term, freq in term_freq.most_common(100) 
             if len(term) > 2 or freq > 2]  # Filter out short, infrequent terms
    
    categorized_skills = {k: [skill for skill, conf in v.items() if conf >= 0.8] 
                         for k, v in categorized_skills.items()}
    
    categorized_skills = {k: v for k, v in categorized_skills.items() if v}
    
    return terms, categorized_skills
def calculate_match_percentage(resume_text, jd_text):
    try:
        resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
        
        tech_categories = ['Programming Languages', 'Web Technologies', 'Database', 'Cloud & DevOps', 'Data Science']
        tech_scores = []
        
        for category in tech_categories:
            if category in jd_categories and jd_categories[category]:
                jd_skills = set(jd_categories[category])
                resume_skills = set(resume_categories.get(category, []))
                
                exact_matches = len(jd_skills & resume_skills)
                partial_matches = sum(1 for js in jd_skills for rs in resume_skills 
                                    if js in rs or rs in js)
                
                match_score = (exact_matches + 0.5 * partial_matches) / len(jd_skills)
                tech_scores.append(match_score)
        
        tech_similarity = sum(tech_scores) / len(tech_scores) if tech_scores else 0.5
        
        soft_skills_score = 0.0
        if 'Soft Skills' in jd_categories and jd_categories['Soft Skills']:
            jd_soft_skills = set(jd_categories['Soft Skills'])
            resume_soft_skills = set(resume_categories.get('Soft Skills', []))
            soft_skills_score = len(jd_soft_skills & resume_soft_skills) / len(jd_soft_skills)
        
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            stop_words = set(stopwords.words('english'))
            words = [w for w in text.split() if w not in stop_words]
            return ' '.join(words)
        
        processed_resume = preprocess_text(resume_text)
        processed_jd = preprocess_text(jd_text)
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_jd])
        keyword_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        exp_edu_score = 0.0
        education = analyze_education(resume_text)
        experience = analyze_experience(resume_text)
        
        if education and any(term in education.lower() for term in ['computer science', 'it', 'software', 'engineering']):
            exp_edu_score += 0.5
     
        if experience:
            exp_words = experience.lower()
 
            years_pattern = r'\b\d+\s*(?:\+\s*)?years?\b'
            if re.search(years_pattern, exp_words):
                exp_edu_score += 0.5
   
        final_score = (
            0.45 * tech_similarity +
            0.15 * soft_skills_score +
            0.25 * keyword_similarity +
            0.15 * exp_edu_score
        )
        
        if final_score > 0.6:
            final_score = 0.6 + (final_score - 0.6) * 1.5
        elif final_score < 0.4:
            final_score = 0.4 * (final_score / 0.4)

        final_score = max(0, min(1, final_score))
        
        return round(final_score * 100, 1)
        
    except Exception as e:
        print(f"Error in calculate_match_percentage: {str(e)}")
        return 50
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
    resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
    jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
    
    missing_keywords = list(set(jd_keywords) - set(resume_keywords))
    key_strengths = get_key_strengths(resume_keywords, jd_keywords)
    
    match_percentage = calculate_match_percentage(resume_text, jd_text)
    
    education = analyze_education(resume_text)
    experience = analyze_experience(resume_text)
    projects = analyze_projects(resume_text)
    achievements = analyze_achievements(resume_text)

    profile_parts = []
    if education:
        profile_parts.append(education.strip().capitalize())
    if experience:
        profile_parts.append(experience)
    profile_summary = ' | '.join(profile_parts)
    
    skill_gaps = {}
    for category in SKILL_CATEGORIES:
        jd_skills = set(jd_categories.get(category, []))
        resume_skills = set(resume_categories.get(category, []))
        if jd_skills:
            skill_gaps[category] = list(jd_skills - resume_skills)
    
    category_matches = {}
    for category in SKILL_CATEGORIES:
        jd_skills = set(jd_categories.get(category, []))
        resume_skills = set(resume_categories.get(category, []))
        if jd_skills:
            match = len(jd_skills & resume_skills) / len(jd_skills) * 100
            category_matches[category] = round(match, 1)
    
    recommendations = []
    
    if not education:
        recommendations.append("Add your educational background prominently")
    elif 'computer science' in education.lower() or 'cs' in education.lower():
        recommendations.append("Your CS background is relevant - highlight any specialized coursework or projects")
    
    if not experience:
        recommendations.append("Add any internships, projects, or relevant work experience")
    elif 'internship' in experience.lower():
        recommendations.append("Quantify your internship achievements with specific metrics")
    elif any(str(i) in experience.lower() for i in range(1, 6)):
        recommendations.append("Highlight leadership roles and team contributions in your experience")
    
    if not projects:
        recommendations.append("Add relevant projects showcasing your technical skills")
    elif len(projects) < 3:
        recommendations.append("Consider adding more projects demonstrating your expertise")
    
    if not achievements:
        recommendations.append("Add quantifiable achievements and metrics to strengthen your impact")
    
    tech_categories = ['Programming Languages', 'Web Technologies', 'Database', 'Cloud & DevOps']
    missing_tech = [cat for cat in tech_categories if cat in skill_gaps and skill_gaps[cat]]
    
    if missing_tech:
        for category in missing_tech[:2]:  # Suggest skills from top 2 categories
            gaps = skill_gaps[category][:3]  # Suggest top 3 missing skills
            if gaps:
                recommendations.append(f"Add {category} skills: {', '.join(gaps)}")
   
    if len(resume_text.split()) < 200:
        recommendations.append("Your resume seems concise - consider adding more detail to your experiences")
    
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
        
        # Convert JSON response to Python dictionary
        results = json.loads(ats_response)
        
        # Display match percentage with color coding
        # Green: ≥80%, Orange: ≥60%, Red: <60%
        match_pct = float(results['JD Match'].strip('%'))
        color = 'green' if match_pct >= 80 else 'orange' if match_pct >= 60 else 'red'
        st.markdown(f"<h2 style='color: {color}; text-align: center;'>Overall Match: {results['JD Match']}</h2>", unsafe_allow_html=True)
        
        # Create three tabs for organized results display
        tab1, tab2, tab3 = st.tabs(["Overview", "Skills Analysis", "Recommendations"])
        
        # Tab 1: Overview - Display basic profile information
        with tab1:
            # Show profile summary
            st.markdown("### Profile Summary")
            st.info(results['Profile Summary'])
            
            # Display education and experience in two columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Education")
                st.write(results['Education'])
            with col2:
                st.markdown("### Experience")
                st.write(results['Experience'])
            
            # Display projects and achievements if available
            if results['Projects'] or results['Achievements']:
                st.markdown("### Key Highlights")
                
                # Show top 3 projects
                if results['Projects']:
                    st.markdown("#### Notable Projects")
                    for project in results['Projects']:
                        st.markdown(f"* {project.capitalize()}")
                
                # Show top 3 achievements
                if results['Achievements']:
                    st.markdown("#### Key Achievements")
                    for achievement in results['Achievements']:
                        st.markdown(f"* {achievement.capitalize()}")
        
        # Tab 2: Skills Analysis - Show detailed skill matching and gaps
        with tab2:
            # Display skill categories with match percentages
            st.markdown("### Skills by Category")
            for category, match in results['Category Matches'].items():
                # Create a progress bar layout with 75-25 split
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Color code the progress bars based on match percentage
                    progress_color = 'green' if match >= 80 else 'orange' if match >= 60 else 'red'
                    st.markdown(f"**{category}**")
                    st.progress(match/100)  # Show progress bar
                with col2:
                    # Display match percentage with color coding
                    st.markdown(f"<h4 style='color: {progress_color}'>{match}%</h4>", unsafe_allow_html=True)
                
                # Show missing skills in each category
                if category in results['Skill Gaps'] and results['Skill Gaps'][category]:
                    st.caption(f"Missing: {', '.join(results['Skill Gaps'][category])}")
            
            # Display strengths and areas for improvement side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Key Strengths")
                for strength in results['Key Strengths']:
                    st.markdown(f"+ {strength}")  # Use bullet points for strengths
            with col2:
                st.markdown("### Areas to Add")
                for keyword in results['Missing Keywords']:
                    st.markdown(f"- {keyword}")  # Use minus for missing skills
        
        # Tab 3: Recommendations - Provide actionable feedback
        with tab3:
            # Show personalized recommendations
            st.markdown("### Detailed Recommendations")
            for i, rec in enumerate(results['Recommendations'], 1):
                st.markdown(f"{i}. {rec}")  # Numbered list of recommendations
            
            # Display general resume improvement tips
            st.markdown("### Pro Tips")
            st.info("""
            - Use industry-standard section headings
            - Include relevant certifications
            - Highlight achievements with metrics
            - Keep formatting simple and consistent
            """)
    # Show warning if inputs are missing
    else:
        st.warning("Please upload a resume and enter a job description.")
