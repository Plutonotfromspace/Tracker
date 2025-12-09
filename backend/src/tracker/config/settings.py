"""
Configuration settings for the vehicle tracking system.

Supports loading from environment variables with fallback defaults.
Uses python-dotenv for .env file support.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.tracker.utils.logging_config import get_logger

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = get_logger(__name__)


def _get_env_list(key: str, default: List[int]) -> List[int]:
    """Get a list of integers from environment variable."""
    value = os.getenv(key)
    if value:
        return [int(x.strip()) for x in value.split(",")]
    return default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.getenv(key)
    return float(value) if value else default


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.getenv(key)
    return int(value) if value else default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


def _get_env_optional_int(key: str, default: Optional[int]) -> Optional[int]:
    """Get optional integer from environment variable."""
    value = os.getenv(key)
    if value is None or value.lower() == "none":
        return default
    return int(value)


@dataclass
class Settings:
    """
    Configuration settings for the vehicle tracking system.
    
    All settings can be overridden via environment variables.
    
    Attributes:
        video_source: Path to video file or RTSP stream URL.
        zone_polygon: Detection zone coordinates as list of [x, y] points.
        confidence_threshold: Minimum detection confidence (0-1).
        iou_threshold: IoU threshold for NMS (0-1).
        vehicle_classes: COCO class IDs to detect.
        show_video: Whether to display video window.
        frame_skip: Process every Nth frame.
        max_frames: Maximum frames to process (None for all).
        yolo_model: YOLO model file path.
        device: Compute device ("cuda" or "cpu").
        entry_line: Entry line coordinates [x1, y1, x2, y2].
        openai_api_key: OpenAI API key for advanced features.
        openai_model: OpenAI model for vision tasks.
        min_aspect_ratio: Minimum aspect ratio for truck filtering.
    """
    
    # Video source
    video_source: str = field(default_factory=lambda: os.getenv(
        "VIDEO_SOURCE", 
        r"videos/input.mp4"
    ))
    
    # Detection zone (list of [x, y] coordinates)
    zone_polygon: List[List[int]] = field(default_factory=lambda: [
        [328, 544], [1074, 709], [1681, 776], 
        [1675, 948], [899, 830], [250, 711]
    ])
    
    # Detection settings
    confidence_threshold: float = field(
        default_factory=lambda: _get_env_float("CONFIDENCE_THRESHOLD", 0.15)
    )
    iou_threshold: float = field(
        default_factory=lambda: _get_env_float("IOU_THRESHOLD", 0.4)
    )
    
    # COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes: List[int] = field(
        default_factory=lambda: _get_env_list("VEHICLE_CLASSES", [2, 5, 7])
    )
    
    # Display settings
    show_video: bool = field(
        default_factory=lambda: _get_env_bool("SHOW_VIDEO", False)
    )
    frame_skip: int = field(
        default_factory=lambda: _get_env_int("FRAME_SKIP", 1)
    )
    max_frames: Optional[int] = field(
        default_factory=lambda: _get_env_optional_int("MAX_FRAMES", None)
    )
    
    # Model settings
    # Using yolov8x (extra-large) for detection; SAM2 handles segmentation
    yolo_model: str = field(
        default_factory=lambda: os.getenv("YOLO_MODEL", "models/yolov8x.pt")
    )
    
    # Segmentation mask settings (uses SAM2 for pixel-perfect masks)
    save_segmentation_masks: bool = field(
        default_factory=lambda: _get_env_bool("SAVE_SEGMENTATION_MASKS", True)
    )
    # SAM2 model: facebook/sam2-hiera-tiny, small, base-plus, or large
    sam2_model: str = field(
        default_factory=lambda: os.getenv("SAM2_MODEL", "facebook/sam2-hiera-large")
    )
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", "cuda")
    )
    
    # Line crossing detection [x1, y1, x2, y2]
    entry_line: List[int] = field(default_factory=lambda: [800, 700, 600, 800])
    
    # OpenAI settings (for optional advanced features)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    )
    
    # xAI Grok settings (for VLM classification)
    xai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("XAI_API_KEY")
    )
    grok_model: str = field(
        default_factory=lambda: os.getenv("GROK_MODEL", "grok-4-fast-non-reasoning")
    )
    
    # Aspect ratio filter settings
    min_aspect_ratio: float = field(
        default_factory=lambda: _get_env_float("MIN_ASPECT_RATIO", 1.5)
    )
    
    # Small vehicle classification threshold
    # Vehicles with aspect ratio < this are categorized as "small_vehicle"
    # Vehicles with aspect ratio >= this are categorized as "full_truck"
    small_vehicle_aspect_threshold: float = field(
        default_factory=lambda: _get_env_float("SMALL_VEHICLE_ASPECT_THRESHOLD", 2.1)
    )
    
    # Small vehicle height threshold (pixels)
    # Vehicles with crop height < this are categorized as "small_vehicle" regardless of aspect ratio
    small_vehicle_height_threshold: int = field(
        default_factory=lambda: _get_env_int("SMALL_VEHICLE_HEIGHT_THRESHOLD", 160)
    )
    
    # Bounding box merger settings (for cab+trailer merging)
    # DISABLED by default - can cause multi-vehicle contamination
    bbox_merge_enabled: bool = field(
        default_factory=lambda: _get_env_bool("BBOX_MERGE_ENABLED", False)
    )
    bbox_merge_max_gap: int = field(
        default_factory=lambda: _get_env_int("BBOX_MERGE_MAX_GAP", 150)
    )
    bbox_merge_min_overlap: float = field(
        default_factory=lambda: _get_env_float("BBOX_MERGE_MIN_OVERLAP", 0.4)
    )
    
    # Crop expansion settings
    crop_expansion_pixels: int = field(
        default_factory=lambda: _get_env_int("CROP_EXPANSION_PIXELS", 100)
    )
    max_crop_iterations: int = field(
        default_factory=lambda: _get_env_int("MAX_CROP_ITERATIONS", 10)
    )
    
    # BBox padding settings (direction-aware padding for trailer capture)
    # DISABLED - bbox extension tends to capture adjacent trucks in busy scenes
    bbox_padding_enabled: bool = field(
        default_factory=lambda: _get_env_bool("BBOX_PADDING_ENABLED", False)
    )
    bbox_base_padding_percent: float = field(
        default_factory=lambda: _get_env_float("BBOX_BASE_PADDING_PERCENT", 0.05)
    )
    bbox_trailer_extension_percent: float = field(
        default_factory=lambda: _get_env_float("BBOX_TRAILER_EXTENSION_PERCENT", 0.3)
    )
    
    # Accept all trucks (no rejection based on aspect ratio)
    # When True, all trucks are accepted and categorized instead of rejected
    accept_all_trucks: bool = field(
        default_factory=lambda: _get_env_bool("ACCEPT_ALL_TRUCKS", True)
    )
    
    # Delayed capture settings (for flatbed trucks)
    # Waits after line crossing to find best capture frame with collision-aware bbox extension
    delayed_capture_enabled: bool = field(
        default_factory=lambda: _get_env_bool("DELAYED_CAPTURE_ENABLED", True)
    )
    delayed_capture_window_frames: int = field(
        default_factory=lambda: _get_env_int("DELAYED_CAPTURE_WINDOW_FRAMES", 10)
    )
    
    # ByteTrack settings
    track_activation_threshold: float = field(
        default_factory=lambda: _get_env_float("TRACK_ACTIVATION_THRESHOLD", 0.25)
    )
    lost_track_buffer: int = field(
        default_factory=lambda: _get_env_int("LOST_TRACK_BUFFER", 60)
    )
    minimum_matching_threshold: float = field(
        default_factory=lambda: _get_env_float("MINIMUM_MATCHING_THRESHOLD", 0.8)
    )
    tracker_frame_rate: int = field(
        default_factory=lambda: _get_env_int("TRACKER_FRAME_RATE", 8)
    )
    
    # Job-based processing (optional)
    # When set, outputs are isolated to data/jobs/{job_id}/ instead of outputs/
    job_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if not DOTENV_AVAILABLE:
            logger.warning(
                "python-dotenv not installed. Environment variables will be used "
                "directly. Install with: pip install python-dotenv"
            )
        
        if self.openai_api_key:
            logger.info("OpenAI API key configured")
        else:
            logger.info("OpenAI API key not configured - advanced features disabled")
    
    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create settings instance from environment variables.
        
        Returns:
            Settings instance with values from environment.
        """
        return cls()
    
    def validate(self) -> bool:
        """
        Validate all settings.
        
        Returns:
            True if all settings are valid.
        
        Raises:
            ValueError: If any setting is invalid.
        """
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, "
                f"got {self.confidence_threshold}"
            )
        
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError(
                f"iou_threshold must be between 0 and 1, "
                f"got {self.iou_threshold}"
            )
        
        if self.frame_skip < 1:
            raise ValueError(
                f"frame_skip must be >= 1, got {self.frame_skip}"
            )
        
        if len(self.entry_line) != 4:
            raise ValueError(
                f"entry_line must have 4 values [x1, y1, x2, y2], "
                f"got {len(self.entry_line)}"
            )
        
        if self.min_aspect_ratio <= 0:
            raise ValueError(
                f"min_aspect_ratio must be > 0, got {self.min_aspect_ratio}"
            )
        
        return True
    
    def get_output_paths(self) -> Dict[str, Path]:
        """
        Get output directory paths based on job_id.
        
        When job_id is set, returns job-based paths: data/jobs/{job_id}/
        When job_id is None, returns legacy paths: outputs/
        
        Returns:
            Dictionary with keys:
                - trucks: Directory for truck crop images
                - reports: Directory for summary/cost reports
                - logs: Directory for log files
                - debug_images: Directory for debug images (trucks/)
        """
        if self.job_id:
            # Job-based paths
            base = Path("data") / "jobs" / self.job_id
            return {
                "trucks": base / "trucks",
                "reports": base / "reports",
                "logs": base / "logs",
                "debug_images": base / "trucks",  # Alias for backward compatibility
            }
        else:
            # Legacy paths
            return {
                "trucks": Path("outputs") / "debug_images",
                "reports": Path("outputs") / "reports",
                "logs": Path("outputs") / "logs",
                "debug_images": Path("outputs") / "debug_images",
            }


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Creates settings on first call, returns cached instance thereafter.
    
    Returns:
        Global Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
