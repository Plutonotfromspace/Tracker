"""
Data models for job management system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


class JobStatus(str, Enum):
    """Job processing status."""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """
    Represents a video processing job with unique ID and isolated storage.
    
    Attributes:
        id: Unique job identifier (timestamp + hash).
        video_path: Path to the input video file.
        status: Current processing status.
        created_at: Job creation timestamp.
        started_at: Processing start timestamp.
        completed_at: Processing completion timestamp.
        output_dir: Base output directory for this job.
        metadata: Additional job metadata (video info, settings, etc).
        error: Error message if job failed.
    """
    id: str
    video_path: str
    status: JobStatus = JobStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_dir: Optional[Path] = None
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert job to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "video_path": self.video_path,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "metadata": self.metadata,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Job":
        """Create job from dictionary."""
        return cls(
            id=data["id"],
            video_path=data["video_path"],
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )
    
    def get_trucks_dir(self) -> Path:
        """Get directory for truck crop images."""
        if not self.output_dir:
            raise ValueError("Job output_dir not set")
        return self.output_dir / "trucks"
    
    def get_reports_dir(self) -> Path:
        """Get directory for reports (summary, cost analysis)."""
        if not self.output_dir:
            raise ValueError("Job output_dir not set")
        return self.output_dir / "reports"
    
    def get_logs_dir(self) -> Path:
        """Get directory for log files."""
        if not self.output_dir:
            raise ValueError("Job output_dir not set")
        return self.output_dir / "logs"
    
    def get_metadata_file(self) -> Path:
        """Get path to job metadata JSON file."""
        if not self.output_dir:
            raise ValueError("Job output_dir not set")
        return self.output_dir / "job.json"
