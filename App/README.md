# AI-Based Resume Shortlisting API

This Flask API converts the Jupyter notebook code into a web service for resume shortlisting based on job descriptions.

## Features 

- **JD Upload**: Upload and process job description files (PDF/DOCX)
- **Resume Upload**: Upload multiple resume files (PDF/DOCX) 
- **Skill Matching**: Extract and match skills between JD and resumes
- **Semantic Analysis**: Use sentence transformers for semantic similarity
- **Experience Matching**: Calculate experience-based scores
- **Ranking System**: Comprehensive scoring and ranking of candidates

## API Endpoints

### 1. Upload Job Description
```
POST /upload_jd
Content-Type: multipart/form-data
Body: file (JD file)
```

### 2. Upload Resumes
```
POST /upload_resumes
Content-Type: multipart/form-data
Body: files[] (Multiple resume files)
```

### 3. Analyze All Resumes
```
POST /analyze_resumes
Returns: Complete analysis with scores for all resumes
```

### 4. Get Top Resumes
```
POST /get_top_resumes
Body: {"top_n": 5}
Returns: Top N ranked candidates
```

### 5. Check Status
```
GET /status
Returns: Current upload status and counts
```

## Installation

### Option 1: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy models:
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

3. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

### Option 2: Docker Installation (Recommended)

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Or build manually:
```bash
# Build the image
docker build -t resume-api .

# Run the container
docker run -p 5000:5000 -v $(pwd)/uploads:/app/uploads -v $(pwd)/resumes:/app/resumes resume-api
```

The server will be accessible at `http://localhost:5000` from your local system.

## Usage Example

1. Upload a JD file
2. Upload multiple resume files  
3. Analyze resumes to get ranked results
4. Get top candidates based on scores

## Scoring Algorithm

The final score is calculated as:
- **50%** Skill Matching (exact + semantic)
- **30%** Semantic Similarity (text embeddings)
- **20%** Experience Score

## File Structure

- `app.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `uploads/` - Temporary JD uploads
- `resumes/` - Processed resume files

## Supported File Formats

- PDF (.pdf)
- Microsoft Word (.docx)
