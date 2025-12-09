"""
FastAPI router for job endpoints.

Provides REST API for querying and managing video processing jobs.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session
from pydantic import BaseModel

from src.tracker.data.database import get_session
from src.tracker.data import crud
from src.tracker.data.models import Job


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobResponse(BaseModel):
    """Response model for job data."""
    id: str
    video_name: str
    video_path: str
    status: str
    created_at: str  # Maps to start_time
    started_at: Optional[str] = None  # Maps to start_time
    completed_at: Optional[str] = None  # Maps to end_time
    truck_count: int
    error: Optional[str]
    
    class Config:
        from_attributes = True


class ProgressResponse(BaseModel):
    """Response model for job progress data."""
    job_id: str
    status: str
    current_stage: Optional[str]
    progress_percent: float
    current_frame: int
    total_frames: int
    eta_seconds: Optional[int]
    stage_eta_seconds: Optional[int]
    
    class Config:
        from_attributes = True


@router.get("/", response_model=List[JobResponse])
def get_jobs(
    offset: int = 0,
    limit: int = Query(default=100, le=100),
    session: Session = Depends(get_session)
):
    """
    Get all jobs with pagination.
    
    Args:
        offset: Number of records to skip
        limit: Maximum number of records to return (max 100)
        session: Database session (injected)
    
    Returns:
        List of jobs
    """
    # Mark any stale jobs as failed before returning results
    crud.mark_stale_jobs_as_failed(session, timeout_minutes=60)
    
    jobs = crud.get_all_jobs(session, offset=offset, limit=limit)
    return [
        JobResponse(
            id=job.id,
            video_name=job.video_name,
            video_path=job.video_path,
            status=job.status,
            created_at=job.start_time.isoformat(),
            started_at=job.start_time.isoformat(),
            completed_at=job.end_time.isoformat() if job.end_time else None,
            truck_count=job.truck_count,
            error=job.error
        )
        for job in jobs
    ]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    session: Session = Depends(get_session)
):
    """
    Get a specific job by ID.
    
    Args:
        job_id: Unique job identifier
        session: Database session (injected)
    
    Returns:
        Job details
    
    Raises:
        HTTPException: If job not found
    """
    # Mark any stale jobs as failed before returning results
    crud.mark_stale_jobs_as_failed(session, timeout_minutes=60)
    
    job = crud.get_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobResponse(
        id=job.id,
        video_name=job.video_name,
        video_path=job.video_path,
        status=job.status,
        created_at=job.start_time.isoformat(),
        started_at=job.start_time.isoformat(),
        completed_at=job.end_time.isoformat() if job.end_time else None,
        truck_count=job.truck_count,
        error=job.error
    )


@router.get("/{job_id}/progress", response_model=ProgressResponse)
def get_job_progress(
    job_id: str,
    session: Session = Depends(get_session)
):
    """
    Get real-time progress information for a job.
    
    This endpoint is designed for polling to show live progress updates.
    
    Args:
        job_id: Unique job identifier
        session: Database session (injected)
    
    Returns:
        ProgressResponse with current progress information
    
    Raises:
        HTTPException: If job not found
    """
    job = crud.get_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ProgressResponse(
        job_id=job.id,
        status=job.status,
        current_stage=job.current_stage,
        progress_percent=job.progress_percent or 0.0,
        current_frame=job.current_frame or 0,
        total_frames=job.total_frames or 0,
        eta_seconds=job.eta_seconds,
        stage_eta_seconds=job.stage_eta_seconds
    )


@router.delete("/{job_id}")
def delete_job(
    job_id: str,
    session: Session = Depends(get_session)
):
    """
    Delete a job and all associated trucks.
    
    Args:
        job_id: Unique job identifier
        session: Database session (injected)
    
    Returns:
        Success message
    
    Raises:
        HTTPException: If job not found
    """
    success = crud.delete_job(session, job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return {"message": f"Job {job_id} deleted successfully"}
