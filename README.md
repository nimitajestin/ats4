# Verq ATS Resume Analyzer

A modern Applicant Tracking System (ATS) compliance checker that evaluates resumes against job descriptions using:
- TF-IDF vectorization
- Cosine similarity
- Domain-specific skill categorization
- NLP-powered text analysis


## Key Features

### Smart Resume Analysis
- Automatic section detection (Education, Experience, Skills)
- PDF text extraction with layout preservation
- Context-aware skill scoring
- Multi-domain categorization (Programming, Data Science, etc.)

### Advanced Matching
- Weighted scoring system (critical vs optional skills)
- Semantic similarity for synonymous terms
- Color-coded match indicators (Green ≥80%, Orange ≥60%, Red <60%)

### Technical Highlights
- Streamlit-based interactive UI
- NLTK for NLP processing
- Custom domain knowledge bases
- Extensible scoring algorithms

## Installation

### Prerequisites
- Python 3.10+
- Pip package manager

### Quick Start
```bash
# Clone repository
git clone https://github.com/nimitajestin/ats4.git
cd ats4

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

## Usage Guide

1. **Upload Resume**
   - PDF format recommended
   - Standard chronological format works best

2. **Enter Job Description**
   - Paste full job description text
   - Minimum 50 words for accurate analysis

3. **Review Analysis**
   - Match percentage with breakdown
   - Skills gap analysis
   - Personalized improvement suggestions


