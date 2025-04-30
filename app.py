import streamlit as st
import os                  
import PyPDF2 as pdf      
import json               
import re                 
from collections import Counter  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity      
import plotly.express as px

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

SKILL_CATEGORIES = {
    'Technical': {
        'weight': 1.5,
        'subcategories': ['Programming', 'Frameworks', 'Databases', 'DevOps', 'Tools']
    },
    'Domain': {
        'weight': 1.2,
        'subcategories': ['Industry', 'Regulatory', 'Methodologies', 'Standards']
    },
    'Soft': {
        'weight': 0.8,
        'subcategories': ['Communication', 'Leadership', 'Teamwork', 'Problem Solving']
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

INDUSTRY_SKILLS = {
    'Software Engineering': {
        'Programming': ['Python', 'Java', 'C++', 'JavaScript'],
        'Frameworks': ['React', 'Django', 'Spring', 'Node.js']
    },
    'Data Science': {
        'Programming': ['Python', 'R', 'SQL'],
        'Tools': ['Pandas', 'TensorFlow', 'PyTorch', 'Tableau']
    },
    'DevOps': {
        'Tools': ['Docker', 'Kubernetes', 'AWS', 'Terraform'],
        'Methodologies': ['CI/CD', 'Infrastructure as Code']
    }
}

def detect_industry(resume_text):
    """Detect industry based on resume content"""
    text = resume_text.lower()
    if any(term in text for term in ['data scien', 'machine learn', 'ai ', 'artificial intel']):
        return 'Data Science'
    elif any(term in text for term in ['devops', 'cloud', 'aws', 'azure', 'infrastructure']):
        return 'DevOps'
    return 'Software Engineering'  # Default

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
    
    for category, config in SKILL_CATEGORIES.items():
        for subcat in config['subcategories']:
            for skill in ['python', 'java', 'javascript', 'js', 'typescript', 'ts', 'c++', 'c#', 'csharp', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab', 'c', 'cpp']:
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
            match_percentage = 0
            
            education = analyze_education(resume_text)
            experience = analyze_experience(resume_text)
            
            if education:
                profile_summary = education.strip().capitalize()
            if experience:
                if profile_summary:
                    profile_summary += " | "
                profile_summary += experience
            
            # Skill category analysis...
            for category, config in SKILL_CATEGORIES.items():
                jd_skills = set()
                resume_skills = set()
                
                # Aggregate skills with context awareness
                for subcat in config['subcategories']:
                    jd_skills.update(skill for skill in jd_categories.get(subcat, []) 
                                   if skill.lower() not in STOP_WORDS)
                    resume_skills.update(skill for skill in resume_categories.get(subcat, []) 
                                       if skill.lower() not in STOP_WORDS)
                
                # Calculate matches with partial matching (e.g., 'Python' matches 'Python 3')
                matched = []
                for jd_skill in jd_skills:
                    for resume_skill in resume_skills:
                        if jd_skill.lower() in resume_skill.lower() or \
                           resume_skill.lower() in jd_skill.lower():
                            matched.append(jd_skill)
                            break
                
                gaps = list(jd_skills - set(matched))
                
                # Store matches and gaps with context
                matched_skills[category] = sorted(matched, 
                    key=lambda x: (jd_keywords.count(x), len(x)), reverse=True)[:10]
                skill_gaps[category] = sorted(gaps, 
                    key=lambda x: (jd_keywords.count(x), len(x)), reverse=True)[:5]
                
                # Enhanced scoring with partial credit for similar skills
                if jd_skills:
                    base_score = len(matched) / len(jd_skills) * 100
                    # Bonus for having extra relevant skills
                    extra_skills = len(resume_skills - jd_skills)
                    base_score = min(base_score + (extra_skills * 2), 100)
                else:
                    base_score = min(len(resume_skills) * 5, 30)
                    
                category_scores[category] = min(round(base_score * config['weight'], 1), 100)
            
            # Generate more actionable recommendations
            recommendations = generate_custom_recommendations(resume_text, jd_text, {
                "JD Match": f"{match_percentage}%",
                "Profile Summary": profile_summary,
                "Key Strengths": [kw[0] for kw in matched_keywords[:5]],
                "Missing Keywords": [kw[0] for kw in missing_keywords[:5]],
                "Education": analyze_education(resume_text),
                "Experience": analyze_experience(resume_text),
                "Projects": analyze_projects(resume_text)[:3],
                "Achievements": analyze_achievements(resume_text)[:3],
                "Category Matches": category_scores,
                "Skill Gaps": skill_gaps,
                "Matched Skills": matched_skills,
                "Recommendations": []
            })
            
            # Calculate overall match percentage
            tech_similarity = sum(category_scores.values()) / len(category_scores) if category_scores else 0.5
            
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

def generate_custom_recommendations(resume_text, jd_text, results):
    """Generate personalized recommendations"""
    industry = detect_industry(resume_text)
    recommendations = []
    
    # 1. Industry-specific suggestions
    if industry in INDUSTRY_SKILLS:
        top_skills = INDUSTRY_SKILLS[industry]
        for category, skills in top_skills.items():
            missing = [s for s in skills 
                      if s.lower() not in resume_text.lower() 
                      and s.lower() in jd_text.lower()]
            if missing:
                recommendations.append(
                    f"Industry Standard: Add {category} skills like {', '.join(missing[:3])}"
                )
    
    # 2. Experience-based suggestions
    exp_years = len(re.findall(r'\d{4}\s*[-–]\s*\d{4}', resume_text))
    if exp_years < 2:
        recommendations.append("Highlight academic projects and certifications")
    elif exp_years > 5:
        recommendations.append("Focus on leadership and complex project experience")
    
    # 3. Education level suggestions
    if 'phd' in resume_text.lower():
        recommendations.append("Emphasize research and publications")
    elif 'master' in resume_text.lower():
        recommendations.append("Highlight specialized coursework")
    
    return results['Recommendations'] + recommendations

def display_enhanced_results(results):
    """Improved results visualization"""
    with st.container():
        # Score breakdown
        cols = st.columns(3)
        with cols[0]:
            st.metric("Overall Match", f"{results.get('JD Match', '0%')}")
        with cols[1]:
            st.metric("Technical Fit", 
                     f"{results.get('Category Matches', {}).get('Technical', 0)}%")
        with cols[2]:
            st.metric("Experience Relevance", 
                     f"{len(results.get('Experience', '').split('|'))} yrs")
        
        # Skill radar chart
        if 'Category Matches' in results:
            categories = list(results['Category Matches'].keys())
            scores = list(results['Category Matches'].values())
            fig = px.line_polar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                line_close=True,
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Priority recommendations
        st.subheader("Priority Improvements")
        for rec in results.get('Recommendations', [])[:3]:
            st.markdown(f"- {rec}")
        
        # Detailed skill analysis
        with st.expander("Detailed Skill Analysis"):
            for category in ['Technical', 'Domain', 'Soft']:
                if category in results.get('Matched Skills', {}):
                    st.markdown(f"**{category} Skills Present:**")
                    st.write(", ".join(results['Matched Skills'][category][:8]))
                if category in results.get('Skill Gaps', {}):
                    st.markdown(f"**Missing {category} Skills:**")
                    st.write(", ".join(results['Skill Gaps'][category][:5]))

@st.cache_data 
def get_ats_feedback(resume_text, jd_text):
    try:
        # Enhanced keyword extraction with weights
        resume_keywords, resume_categories = extract_skills_and_keywords(resume_text)
        jd_keywords, jd_categories = extract_skills_and_keywords(jd_text)
        
        # Calculate keyword frequencies
        resume_kw_freq = {kw: resume_keywords.count(kw) for kw in set(resume_keywords)}
        jd_kw_freq = {kw: jd_keywords.count(kw) for kw in set(jd_keywords)}
        
        # Improved matching with weighted scores
        missing_keywords = []
        matched_keywords = []
        
        for kw in set(jd_keywords):
            if kw in resume_keywords:
                matched_keywords.append((kw, jd_kw_freq[kw]))  # (keyword, importance)
            else:
                missing_keywords.append((kw, jd_kw_freq[kw]))
        
        # Sort by importance (frequency in JD)
        matched_keywords.sort(key=lambda x: x[1], reverse=True)
        missing_keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Enhanced skill category analysis
        skill_gaps = {}
        matched_skills = {}
        category_scores = {}
        
        # Define skill category weights
        CATEGORY_WEIGHTS = {
            'Technical': 1.2,
            'Domain': 1.1,
            'Soft': 0.8
        }
        
        for category in SKILL_CATEGORIES:
            jd_skills = set(jd_categories.get(category, []))
            resume_skills = set(resume_categories.get(category, []))
            
            matched = list(jd_skills & resume_skills)
            gaps = list(jd_skills - resume_skills)
            
            # Calculate weighted match score
            if jd_skills:
                base_score = len(matched) / len(jd_skills) * 100
                weighted_score = base_score * CATEGORY_WEIGHTS.get(category, 1.0)
            else:
                # Credit for additional resume skills (diminishing returns)
                weighted_score = min(len(resume_skills) * 5, 30)  # Max 30% bonus
                
            matched_skills[category] = matched[:8]  # Show top 8 matches
            skill_gaps[category] = gaps[:5]        # Show top 5 gaps
            category_scores[category] = min(round(weighted_score, 1), 100)
        
        # Calculate overall match percentage (weighted average)
        total_score = sum(category_scores.values())
        max_possible = sum(CATEGORY_WEIGHTS.get(c, 1.0) * 100 for c in SKILL_CATEGORIES)
        match_percentage = round((total_score / max_possible) * 100, 1)
        
        # Enhanced recommendation engine
        recommendations = []
        
        # 1. Overall match quality
        if match_percentage < 40:
            recommendations.append(" Major Improvement Needed: Your resume shows significant gaps compared to the job requirements")
        elif match_percentage < 65:
            recommendations.append(" Moderate Improvement Needed: Several key areas need enhancement")
        elif match_percentage < 85:
            recommendations.append(" Minor Improvements: Your resume is good but could be stronger")
        else:
            recommendations.append(" Strong Match: Your resume aligns well with the job requirements")
        
        # 2. Category-specific recommendations
        for category in SKILL_CATEGORIES:
            score = category_scores[category]
            gaps = skill_gaps[category]
            
            if score < 50:
                rec = f" Focus on {category} skills: "
                if gaps:
                    rec += f"Add {', '.join(gaps[:3])}"
                else:
                    rec += f"Highlight your {category.lower()} skills more prominently"
                recommendations.append(rec)
            
        # 3. Content completeness checks
        sections_missing = []
        if not analyze_education(resume_text):
            sections_missing.append("education")
        if not analyze_experience(resume_text):
            sections_missing.append("work experience")
        if sections_missing:
            recommendations.append(f" Add missing sections: {', '.join(sections_missing)}")
        
        # 4. Impactful writing suggestions
        if len(analyze_achievements(resume_text)) < 2:
            recommendations.append(" Add more achievements with quantifiable results (e.g., 'Increased sales by 30%')")
        
        # 5. Skill demonstration
        projects = analyze_projects(resume_text)
        if not projects:
            recommendations.append(" Add projects demonstrating your technical skills")
        elif len(projects) < 2:
            recommendations.append(" Include more projects that showcase relevant skills")
        
        return {
            "JD Match": f"{match_percentage}%",
            "Profile Summary": profile_summary,
            "Key Strengths": [kw[0] for kw in matched_keywords[:5]],
            "Missing Keywords": [kw[0] for kw in missing_keywords[:5]],
            "Education": analyze_education(resume_text) or "",
            "Experience": analyze_experience(resume_text) or "",
            "Projects": projects[:3],
            "Achievements": analyze_achievements(resume_text)[:3],
            "Category Matches": category_scores,
            "Skill Gaps": skill_gaps,
            "Matched Skills": matched_skills,
            "Recommendations": recommendations
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
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
            try:
                results = json.loads(ats_response)
            except json.JSONDecodeError:
                st.error("Failed to parse analysis results")
                return
        else:
            results = ats_response
        
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
    else:
        st.warning("Please upload a resume and enter a job description.")
