# Enhanced skill categories with industry-specific mappings
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
                    st.write(", ".join(results['Skill Gaps'][category]))

# Streamlit UI Components
st.title('ATS Resume Evaluator')

# File uploader
uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Job description input
jd_input = st.text_area("Paste Job Description", height=200)

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
            
            # Display results using enhanced visualization
            display_enhanced_results(results)
else:
    st.warning("Please upload a resume and enter a job description.")