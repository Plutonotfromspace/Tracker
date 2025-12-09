"""
FastAPI router for video upload endpoints.

Provides REST API for uploading videos and creating processing jobs.
"""

import os
import shutil
import subprocess
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.tracker.data.database import get_session
from src.tracker.data import crud
from src.tracker.api.auth import require_non_demo_user
from src.tracker.utils.paths import get_backend_dir
from sqlmodel import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])


def start_video_processing(job_id: str, video_path: str, max_frames: Optional[int] = None):
    """
    Start video processing in a background process.
    
    This runs main.py with the job-id flag to process the video
    and update the job status in the database.
    
    Args:
        job_id: The job ID to use for processing
        video_path: Path to the video file
        max_frames: Optional limit on number of frames to process
        background_tasks: FastAPI background tasks for monitoring
    """
    try:
        # Get the Python executable from the virtual environment
        python_exe = sys.executable
        
        # Get backend directory using utility function
        backend_dir = get_backend_dir(__file__)
        main_script = backend_dir / "main.py"
        
        if not main_script.exists():
            raise FileNotFoundError(f"Could not find main.py in {backend_dir}")
        
        # Build command arguments
        cmd_args = [
            python_exe,
            str(main_script),
            "--video", video_path,
            "--headless",
            "--job-id", job_id,  # Pass the existing job ID
        ]
        
        # Add max frames if specified
        if max_frames and max_frames > 0:
            cmd_args.extend(["--max-frames", str(max_frames)])
        
        logger.info(f"Starting video processing for job {job_id}: {' '.join(cmd_args)}")
        
        # Create log file for this job
        backend_dir = get_backend_dir(__file__)
        log_dir = backend_dir / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{job_id}.log"
        
        # Start the processing in the background
        # Write output to log file instead of PIPE to avoid buffering issues
        with open(log_file, "w") as log_f:
            process = subprocess.Popen(
                cmd_args,
                stdout=log_f,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                cwd=str(backend_dir)  # backend directory
            )
        
        logger.info(f"Job {job_id} subprocess started with PID {process.pid}, logs: {log_file}")
        
        # Check immediately if process crashed on startup
        import time
        time.sleep(1)  # Give it a second
        poll_result = process.poll()
        if poll_result is not None and poll_result != 0:
            # Process already exited with error - likely crashed
            with open(log_file, "r") as log_f:
                error_msg = log_f.read()
            logger.error(f"Job {job_id} subprocess crashed immediately (exit code {poll_result}): {error_msg[-500:]}")
            
            # Update job to failed
            from src.tracker.data.database import get_session_context
            with get_session_context() as session:
                crud.update_job_status(session, job_id, "failed", error=f"Process crashed (code {poll_result}): {error_msg[-200:]}")
        
    except Exception as e:
        logger.error(f"Failed to start video processing for job {job_id}: {e}")
        
        # Update job to failed status
        from src.tracker.data.database import get_session_context
        with get_session_context() as session:
            crud.update_job_status(session, job_id, "failed", error=f"Failed to start: {str(e)}")


class UploadResponse(BaseModel):
    """Response model for upload requests."""
    job_id: str
    video_name: str
    message: str


@router.post("/", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    user = Depends(require_non_demo_user)  # This will block demo users
):
    """
    Upload a video file and create a processing job.
    
    **This endpoint is restricted for demo users.**
    Demo users will receive a 403 Forbidden response.
    
    Args:
        file: Video file to upload
        session: Database session (injected)
        user: Current authenticated user (injected, must not be demo)
    
    Returns:
        UploadResponse with job_id and status
        
    Raises:
        HTTPException: 400 if file is invalid, 403 if demo user
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only video files are allowed."
        )
    
    # Validate file extension
    valid_extensions = [".mp4", ".mpeg", ".mov", ".avi"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(valid_extensions)}"
        )
    
    # Create uploads directory
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    video_path = uploads_dir / safe_filename
    
    try:
        # Save file to disk
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    
    # Generate job ID
    from datetime import datetime
    import hashlib
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_input = f"{video_path}_{timestamp}".encode()
    hash_suffix = hashlib.md5(hash_input).hexdigest()[:4]
    job_id = f"{timestamp}_{hash_suffix}"
    
    # Create job in database
    job = crud.create_job(
        session=session,
        job_id=job_id,
        video_name=file.filename,
        video_path=str(video_path),
        status="created"  # Start as "created", will be "processing" when picked up
    )
    
    # Start video processing in the background
    start_video_processing(job.id, str(video_path))
    
    return UploadResponse(
        job_id=job.id,
        video_name=file.filename,
        message="Video uploaded successfully. Processing has started."
    )


@router.post("/demo", response_model=UploadResponse)
async def upload_demo_video(
    max_frames: Optional[int] = None,
    session: Session = Depends(get_session)
):
    """
    Create a processing job for the demo video (pre-loaded on server).
    
    This endpoint is specifically for demo users who cannot upload their own videos.
    It creates a job that references the server-side demo video file.
    
    Args:
        max_frames: Optional limit on number of frames to process (for faster testing)
        session: Database session (injected)
    
    Returns:
        UploadResponse with job_id and status
        
    Raises:
        HTTPException: 404 if demo video not found
    """
    # Path to the demo video
    backend_dir = get_backend_dir(__file__)
    demo_video_path = backend_dir / "videos" / "World Trade Bridge Oct 16th from 3.22PM to 3.51PM.mp4"
    
    if not demo_video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Demo video not found on server"
        )
    
    # Generate job ID
    from datetime import datetime
    import hashlib
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_input = f"{demo_video_path}_{timestamp}".encode()
    hash_suffix = hashlib.md5(hash_input).hexdigest()[:4]
    job_id = f"{timestamp}_{hash_suffix}"
    
    # Create job in database
    job = crud.create_job(
        session=session,
        job_id=job_id,
        video_name="World Trade Bridge Oct 16th from 3.22PM to 3.51PM.mp4",
        video_path=str(demo_video_path.absolute()),
        status="created"
    )
    
    # Start video processing in the background
    start_video_processing(job.id, str(demo_video_path.absolute()), max_frames)
    
    return UploadResponse(
        job_id=job.id,
        video_name="World Trade Bridge Oct 16th from 3.22PM to 3.51PM.mp4",
        message="Demo video job created successfully. Processing has started."
    )
