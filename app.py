import os
import subprocess
import uuid
import shutil
import sys
import yaml
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Dict, Any

# Import render function from dynamic_sop_builder
# We'll need to make sure dynamic_sop_builder is importable or call it via CLI
# For simplicity and reliability, we'll add a helper to call it via CLI with a json file


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
    
    # New paths for SOP Builder
    sop_yaml_path = OUTPUT_DIR / f"{file_id}_sop.yaml"
    sop_video_filename = f"{file_id}_sop_output.mp4"
    sop_video_path = OUTPUT_DIR / sop_video_filename
    
    # 3. Run inference_video.py
    # Use sys.executable to ensure we use the same virtual environment
    inference_script = ROOT / "inference_video.py"
    
    inference_command = [
        sys.executable, str(inference_script),
        "--input", str(input_path),
        "--output", str(output_path),
        "--detections", str(detections_path)
    ]
    
    try:
        print(f"Running inference: {' '.join(inference_command)}")
        # Run inference
        subprocess.run(inference_command, capture_output=True, text=True, check=True)
        
        # 4. Run dynamic_sop_builder.py
        sop_builder_script = ROOT / "dynamic_sop_builder.py"
        templates = [
            "put_board.yaml", "take_only.yaml", "take_gasket.yaml",
            "apply_gasket_to_shielding.yaml", "attach_gasket_payload_to_board.yaml",
            "attach_only.yaml", "return_board.yaml"
        ]
        template_paths = [str(ROOT / t) for t in templates]
        global_config = str(ROOT / "sop_global_shared.yaml")
        
        sop_command = [
            sys.executable, str(sop_builder_script),
            "--templates"
        ] + template_paths + [
            "--detections", str(detections_path),
            "--global-config", global_config,
            "--output", str(sop_yaml_path),
            "--video", str(input_path), # Use original video as background
            "--overlay-output", str(sop_video_path)
        ]
        
        print(f"Running SOP builder: {' '.join(sop_command)}")
        subprocess.run(sop_command, capture_output=True, text=True, check=True)
        
        # Load the generated YAML to extract _inference_meta
        with open(sop_yaml_path, "r", encoding="utf-8") as f:
            sop_data = yaml.safe_load(f)
            
        inferred_actions = sop_data.get("_inference_meta", {}).get("matches", [])
        
        return {
            "fileId": file_id,
            "actions": inferred_actions,
            "inferenceVideoUrl": f"/outputs/{output_filename}",
            "detectionsUrl": f"/outputs/{file_id}_detections.jsonl",
            "sopYamlUrl": f"/outputs/{file_id}_sop.yaml"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return JSONResponse(status_code=500, content={"message": "Error processing video", "error": e.stderr})
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Internal server error", "error": str(e)})

@app.post("/api/video/render")
async def render_video(data: Dict[str, Any] = Body(...)):
    file_id = data.get("file_id")
    actions = data.get("actions")
    
    if not file_id or actions is None:
        return JSONResponse(status_code=400, content={"message": "Missing file_id or actions"})
        
    # 1. Prepare paths
    # We assume file_id exists in uploads and detections
    # Find original extension
    input_files = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not input_files:
        return JSONResponse(status_code=404, content={"message": f"Original video not found for {file_id}"})
    
    input_path = input_files[0]
    detections_path = OUTPUT_DIR / f"{file_id}_detections.jsonl"
    
    # Final output video path
    final_video_filename = f"{file_id}_final_output.mp4"
    final_video_path = OUTPUT_DIR / final_video_filename
    
    # Save actions to a temporary JSON file for dynamic_sop_builder
    json_timeline_path = OUTPUT_DIR / f"{file_id}_timeline.json"
    
    # Convert actions to timeline rows format expected by dynamic_sop_builder
    # timeline_rows = [{"action_id": ..., "start_frame": ..., "end_frame": ...}]
    timeline_rows = []
    for action in actions:
        timeline_rows.append({
            "action_id": action.get("action_id"),
            "start_frame": action.get("start_frame"),
            "end_frame": action.get("end_frame")
        })
        
    with open(json_timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_rows, f)
        
    # 2. Run dynamic_sop_builder.py with --json-timeline
    sop_builder_script = ROOT / "dynamic_sop_builder.py"
    
    # We provide required arguments to satisfy argparse, even if they aren't used in json mode
    sop_command = [
        sys.executable, str(sop_builder_script),
        "--templates", "put_board.yaml", 
        "--detections", str(detections_path),
        "--output", str(OUTPUT_DIR / f"{file_id}_dummy.yaml"), # Required by argparse
        "--video", str(input_path),
        "--overlay-output", str(final_video_path),
        "--json-timeline", str(json_timeline_path)
    ]
    
    try:
        print(f"Running Final Render: {' '.join(sop_command)}")
        subprocess.run(sop_command, capture_output=True, text=True, check=True)
        
        return {
            "resultUrl": f"/outputs/{final_video_filename}"
        }
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return JSONResponse(status_code=500, content={"message": "Error rendering video", "error": e.stderr})
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Internal server error", "error": str(e)})


@app.get("/")
async def root():
    return {"message": "Video Processor API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
