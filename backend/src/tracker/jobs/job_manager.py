"""
Job manager for creating and tracking video processing jobs.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import Job, JobStatus
from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)


class JobManager:
    """
    Manages video processing jobs with unique IDs and isolated storage.
    
    Creates job directories, persists metadata, and provides job lookup.
    Uses data/jobs/ as the base directory for all job storage.
    
    Example:
        >>> manager = JobManager()
        >>> job = manager.create_job("videos/test.mp4")
        >>> print(job.id)  # "20251207_205845_a3f2"
        >>> print(job.output_dir)  # Path("data/jobs/20251207_205845_a3f2")
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the job manager.
        
        Args:
            base_dir: Base directory for job storage. Defaults to "data/jobs".
        """
        self.base_dir = base_dir or Path("data/jobs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for quick job lookup
        self.index_file = self.base_dir / "index.json"
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure index file exists."""
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps({"jobs": []}, indent=2))
    
    def _generate_job_id(self, video_path: str) -> str:
        """
        Generate unique job ID from timestamp and video path hash.
        
        Format: YYYYMMDD_HHMMSS_hash4
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Unique job ID string.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{video_path}_{timestamp}".encode()
        hash_suffix = hashlib.md5(hash_input).hexdigest()[:4]
        return f"{timestamp}_{hash_suffix}"
    
    def create_job(
        self, 
        video_path: str,
        metadata: Optional[Dict] = None
    ) -> Job:
        """
        Create a new job with unique ID and directory structure.
        
        Creates the following structure:
        data/jobs/{job_id}/
            ├── job.json          # Job metadata
            ├── trucks/           # Truck crop images (organized by Truck###/)
            ├── reports/          # Summary and cost reports
            └── logs/             # Log files
        
        Args:
            video_path: Path to the input video file.
            metadata: Optional metadata dictionary.
            
        Returns:
            Created Job object.
        """
        job_id = self._generate_job_id(video_path)
        output_dir = self.base_dir / job_id
        
        # Create job object
        job = Job(
            id=job_id,
            video_path=video_path,
            status=JobStatus.CREATED,
            output_dir=output_dir,
            metadata=metadata or {}
        )
        
        # Create directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        job.get_trucks_dir().mkdir(exist_ok=True)
        job.get_reports_dir().mkdir(exist_ok=True)
        job.get_logs_dir().mkdir(exist_ok=True)
        
        # Save job metadata
        self._save_job(job)
        
        # Update index
        self._add_to_index(job)
        
        logger.info(f"Created job {job_id} for video: {video_path}")
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.
        
        Args:
            job_id: Unique job identifier.
            
        Returns:
            Job object if found, None otherwise.
        """
        job_dir = self.base_dir / job_id
        metadata_file = job_dir / "job.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            data = json.loads(metadata_file.read_text())
            return Job.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load job {job_id}: {e}")
            return None
    
    def list_jobs(
        self, 
        status: Optional[JobStatus] = None,
        limit: Optional[int] = None
    ) -> List[Job]:
        """
        List all jobs, optionally filtered by status.
        
        Args:
            status: Filter by job status.
            limit: Maximum number of jobs to return.
            
        Returns:
            List of Job objects.
        """
        try:
            index = json.loads(self.index_file.read_text())
            jobs = []
            
            for job_entry in index.get("jobs", []):
                job = self.get_job(job_entry["id"])
                if job:
                    if status is None or job.status == status:
                        jobs.append(job)
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            
            if limit:
                jobs = jobs[:limit]
            
            return jobs
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    def update_status(
        self, 
        job_id: str, 
        status: JobStatus,
        error: Optional[str] = None
    ) -> bool:
        """
        Update job status.
        
        Args:
            job_id: Job identifier.
            status: New status.
            error: Optional error message if status is FAILED.
            
        Returns:
            True if successful, False otherwise.
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return False
        
        job.status = status
        if error:
            job.error = error
        
        # Update timestamps
        if status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.now()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = datetime.now()
        
        self._save_job(job)
        logger.info(f"Updated job {job_id} status to {status.value}")
        return True
    
    def _save_job(self, job: Job):
        """Save job metadata to disk."""
        metadata_file = job.get_metadata_file()
        metadata_file.write_text(json.dumps(job.to_dict(), indent=2))
    
    def _add_to_index(self, job: Job):
        """Add job to index file."""
        try:
            index = json.loads(self.index_file.read_text())
            
            # Check if job already in index
            if not any(j["id"] == job.id for j in index.get("jobs", [])):
                index.setdefault("jobs", []).append({
                    "id": job.id,
                    "video_path": job.video_path,
                    "created_at": job.created_at.isoformat(),
                })
                self.index_file.write_text(json.dumps(index, indent=2))
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
