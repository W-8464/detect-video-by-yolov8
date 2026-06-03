import os
import subprocess
import uuid
import shutil
import sys
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and outputs
ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve static files from the outputs directory
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

@app.post("/api/video/process")
async def process_video(video: UploadFile = File(...)):
    # 1. Save uploaded video
    file_id = str(uuid.uuid4())
    input_ext = Path(video.filename).suffix
    if not input_ext:
        input_ext = ".mp4"
    
    input_path = UPLOAD_DIR / f"{file_id}{input_ext}"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # 2. Prepare output paths
    output_filename = f"{file_id}_output.mp4"
    output_path = OUTPUT_DIR / output_filename
    detections_path = OUTPUT_DIR / f"{file_id}_detections.jsonl"
    
    # 3. Run inference_video.py
    # Use sys.executable to ensure we use the same virtual environment
    inference_script = ROOT / "inference_video.py"
    
    command = [
        sys.executable, str(inference_script),
        "--input", str(input_path),
        "--output", str(output_path),
        "--detections", str(detections_path)
    ]
    
    try:
        print(f"Running command: {' '.join(command)}")
        # Run the script
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # 4. Return the result URL
        # We return a relative URL that will be handled by the Angular proxy or direct access
        result_url = f"/outputs/{output_filename}"
        
        return {
            "resultUrl": result_url,
            "detectionsUrl": f"/outputs/{file_id}_detections.jsonl"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return JSONResponse(status_code=500, content={"message": "Error processing video", "error": e.stderr})
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Internal server error", "error": str(e)})

@app.get("/")
async def root():
    return {"message": "Video Processor API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
