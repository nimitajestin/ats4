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
    
    stop_words = STOP_WORDS
    words = tokenize_text(text)
    words = [word for word in words if word.lower() not in stop_words]
    
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
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            stop_words = STOP_WORDS
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
            "JD Match": f"{match_percentage}%" if 'match_percentage' in locals() else "0%",
            "Profile Summary": profile_summary,
            "Key Strengths": [kw[0] for kw in matched_keywords[:5]],
            "Missing Keywords": [kw[0] for kw in missing_keywords[:5]],
            "Education": analyze_education(resume_text) if 'resume_text' in locals() else "No education details found",
            "Experience": analyze_experience(resume_text) if 'resume_text' in locals() else "No experience details found",
            "Projects": analyze_projects(resume_text)[:3] if 'resume_text' in locals() and analyze_projects(resume_text) else [],
            "Achievements": analyze_achievements(resume_text)[:3] if 'resume_text' in locals() and analyze_achievements(resume_text) else [],
            "Category Matches": category_scores,
            "Skill Gaps": skill_gaps,
            "Matched Skills": matched_skills,
            "Recommendations": recommendations
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

    # Display basic results
    st.markdown(f"#### Overall Match: {results.get('JD Match', '0%')}")
    
    # Show key strengths and missing keywords
    st.markdown("##### Key Strengths")
    for strength in results.get('Key Strengths', [])[:5]:
        st.markdown(f"- {strength}")
        
    st.markdown("##### Areas for Improvement")
    for keyword in results.get('Missing Keywords', [])[:5]:
        st.markdown(f"- {keyword}")
        
    # Show basic recommendations
    if results.get('Recommendations'):
        st.markdown("##### Recommendations")
        for rec in results['Recommendations'][:3]:
            st.markdown(f"- {rec}")
