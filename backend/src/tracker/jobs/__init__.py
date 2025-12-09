"""
Job management system for tracking video processing jobs.

Provides unique job IDs and isolated storage for multi-video dashboard support.
"""

from .models import Job, JobStatus
from .job_manager import JobManager

__all__ = ["Job", "JobStatus", "JobManager"]
