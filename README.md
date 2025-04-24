# Resume ATS Tracking System

A modern ATS (Applicant Tracking System) built with Python that helps evaluate resumes against job descriptions using advanced NLP techniques.

## Features

- **Job Description Match:** 
  - Evaluates resume-job description compatibility
  - Provides percentage match scores
  - Category-wise skill analysis
  - Smart recommendations based on matches

- **Skills Analysis:**
  - Identifies technical skills, frameworks, and technologies
  - Categorizes skills by domain (Programming, Web, Database, etc.)
  - Highlights missing critical skills
  - Shows strengths and areas for improvement

- **Smart Recommendations:**
  - Personalized feedback based on profile analysis
  - Suggestions for skill improvements
  - Format and content optimization tips
  - Industry-standard best practices

- **Education & Experience Analysis:**
  - Detects educational qualifications
  - Analyzes work experience
  - Identifies projects and achievements
  - Provides context-aware suggestions

## Requirements
- Python 3.10+
- Streamlit
- NLTK
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nimitajestin/ats-resume-analyzer.git
cd ats-resume-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload your resume (PDF format)
2. Enter or paste the job description
3. Click "Analyze" to get:
   - Match percentage
   - Skills analysis
   - Missing keywords
   - Personalized recommendations

## Contributing
Feel free to open issues and pull requests for any improvements.

## License
MIT License
