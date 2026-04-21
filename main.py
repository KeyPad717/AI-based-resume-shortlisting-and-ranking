from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
import tempfile
from pipeline import process_resumes

app = FastAPI(title="Resume ATS Pipeline API")

# Allow Frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/score")
async def score_candidates(
    jd_file: UploadFile = File(...),
    resume_files: List[UploadFile] = File(...)
):
    if not jd_file.filename:
        raise HTTPException(status_code=400, detail="JD File missing")
    if not resume_files or len(resume_files) == 0:
        raise HTTPException(status_code=400, detail="Resume files missing")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save JD
        jd_path = os.path.join(tmpdir, jd_file.filename)
        with open(jd_path, "wb") as f:
            shutil.copyfileobj(jd_file.file, f)
            
        # Save Resumes
        resume_paths = []
        for r_file in resume_files:
            if not r_file.filename: continue
            path = os.path.join(tmpdir, r_file.filename)
            with open(path, "wb") as f:
                shutil.copyfileobj(r_file.file, f)
            resume_paths.append(path)
            
        # Run Pipeline
        try:
            results = process_resumes(jd_path, resume_paths)
            return {"status": "success", "data": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
