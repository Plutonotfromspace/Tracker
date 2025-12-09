"""
Delayed capture manager for intelligent truck capture.

This module implements a delayed capture strategy that:
1. Waits for N frames after line crossing before capturing
2. Tracks bbox changes to find the best (widest) detection
3. Categorizes as full_truck or small_vehicle based on aspect ratio

The key insight is that YOLO may detect different portions of a truck
across different frames. By waiting and selecting the best frame,
we can capture more complete images of trucks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptureCandidate:
    """
    Tracks a pending capture during the delayed capture window.
    
    Attributes:
        tracker_id: ByteTrack tracker ID
        vehicle_id: Sequential vehicle ID for logging
        start_frame: Frame number when capture window started
        best_bbox: Best (widest) bbox seen so far
        best_aspect_ratio: Aspect ratio of best bbox
        best_frame: Frame image with best bbox
        best_time: Timestamp of best frame
        velocity_x: Horizontal velocity for direction detection
        best_mask: Binary segmentation mask (H x W, 0=bg, 255=object) if available
    """
    tracker_id: int
    vehicle_id: int
    start_frame: int
    best_bbox: np.ndarray
    best_aspect_ratio: float
    best_frame: np.ndarray
    best_time: datetime
    velocity_x: float = 0.0
    best_mask: Optional[np.ndarray] = None


@dataclass
class CaptureResult:
    """
    Result of a completed delayed capture.
    
    Attributes:
        tracker_id: ByteTrack tracker ID
        vehicle_id: Sequential vehicle ID
        bbox: Final bounding box
        frame: Frame to capture from
        frame_time: Timestamp
        aspect_ratio: Aspect ratio of the bbox
        category: 'full_truck' or 'small_vehicle' based on aspect ratio
        mask: Binary segmentation mask if available
    """
    tracker_id: int
    vehicle_id: int
    bbox: np.ndarray
    frame: np.ndarray
    frame_time: datetime
    aspect_ratio: float = 0.0
    category: str = "unknown"
    mask: Optional[np.ndarray] = None


@dataclass 
class DelayedCaptureConfig:
    """Configuration for delayed capture behavior."""
    enabled: bool = True
    capture_delay_frames: int = 30  # ~1 second at 30fps
    small_vehicle_aspect_threshold: float = 2.1  # Below this, categorize as small_vehicle
    small_vehicle_height_threshold: int = 160  # Below this height (pixels), categorize as small_vehicle


class DelayedCaptureManager:
    """
    Manages delayed capture windows for intelligent truck capture.
    
    Instead of capturing immediately at line crossing, this manager:
    1. Opens a capture window and tracks bbox changes
    2. Selects the frame with the best (widest) bbox
    3. Categorizes as full_truck or small_vehicle based on aspect ratio
    
    Example:
        >>> manager = DelayedCaptureManager(config)
        >>> # When truck crosses line:
        >>> manager.start_capture_window(tracker_id, vehicle_id, frame, bbox, time)
        >>> # Each frame update:
        >>> result = manager.update(tracker_id, frame, bbox, time, all_bboxes, frame_num)
        >>> if result:
        ...     # Capture window closed, use result
        ...     save_image(result.frame, result.bbox)
    """
    
    def __init__(self, config: Optional[DelayedCaptureConfig] = None):
        """
        Initialize the delayed capture manager.
        
        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or DelayedCaptureConfig()
        self.pending_captures: Dict[int, CaptureCandidate] = {}
        self.completed_count = 0
        
        logger.info(
            f"DelayedCaptureManager initialized: enabled={self.config.enabled}, "
            f"delay_frames={self.config.capture_delay_frames}"
        )
    
    def start_capture_window(
        self,
        tracker_id: int,
        vehicle_id: int,
        frame: np.ndarray,
        bbox: np.ndarray,
        frame_time: datetime,
        current_frame: int,
        velocity_x: float = 0.0,
        mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Start a delayed capture window for a vehicle.
        
        Called when a vehicle crosses the entry line.
        
        Args:
            tracker_id: ByteTrack tracker ID
            vehicle_id: Sequential vehicle ID
            frame: Current video frame
            bbox: Current bounding box
            frame_time: Current timestamp
            current_frame: Current frame number
            velocity_x: Horizontal velocity for direction detection
            mask: Binary segmentation mask if available
        """
        if not self.config.enabled:
            return
        
        aspect_ratio = self._calc_aspect_ratio(bbox)
        
        self.pending_captures[tracker_id] = CaptureCandidate(
            tracker_id=tracker_id,
            vehicle_id=vehicle_id,
            start_frame=current_frame,
            best_bbox=bbox.copy(),
            best_aspect_ratio=aspect_ratio,
            best_frame=frame.copy(),
            best_time=frame_time,
            velocity_x=velocity_x,
            best_mask=mask.copy() if mask is not None else None
        )
        
        logger.debug(
            f"Started capture window for Truck {vehicle_id} "
            f"(tracker {tracker_id}), initial aspect ratio: {aspect_ratio:.2f}, "
            f"mask: {'yes' if mask is not None else 'no'}"
        )
    
    def update(
        self,
        tracker_id: int,
        frame: np.ndarray,
        bbox: np.ndarray,
        frame_time: datetime,
        current_frame: int,
        all_tracked_bboxes: Dict[int, np.ndarray],
        frame_width: int,
        velocity_x: float = 0.0,
        mask: Optional[np.ndarray] = None
    ) -> Optional[CaptureResult]:
        """
        Update a pending capture with new frame data.
        
        Should be called every frame for vehicles with open capture windows.
        
        Args:
            tracker_id: ByteTrack tracker ID
            frame: Current video frame
            bbox: Current bounding box
            frame_time: Current timestamp
            current_frame: Current frame number
            all_tracked_bboxes: Dict of all current tracker_id -> bbox
            frame_width: Width of frame for boundary checking
            velocity_x: Current horizontal velocity
            mask: Binary segmentation mask if available
            
        Returns:
            CaptureResult if window closed, None if still pending
        """
        if not self.config.enabled:
            return None
        
        if tracker_id not in self.pending_captures:
            return None
        
        candidate = self.pending_captures[tracker_id]
        aspect_ratio = self._calc_aspect_ratio(bbox)
        
        # Update velocity (use exponential moving average)
        candidate.velocity_x = 0.7 * candidate.velocity_x + 0.3 * velocity_x
        
        # Check if this frame has a better (wider) bbox
        if aspect_ratio > candidate.best_aspect_ratio:
            candidate.best_bbox = bbox.copy()
            candidate.best_aspect_ratio = aspect_ratio
            candidate.best_frame = frame.copy()
            candidate.best_time = frame_time
            candidate.best_mask = mask.copy() if mask is not None else None
            logger.debug(
                f"Truck {candidate.vehicle_id}: Updated best bbox, "
                f"new aspect ratio: {aspect_ratio:.2f}"
            )
        
        # Check if capture window should close
        frames_elapsed = current_frame - candidate.start_frame
        if frames_elapsed >= self.config.capture_delay_frames:
            return self._finalize_capture(
                tracker_id, all_tracked_bboxes, frame_width
            )
        
        return None
    
    def force_capture(
        self,
        tracker_id: int,
        all_tracked_bboxes: Dict[int, np.ndarray],
        frame_width: int
    ) -> Optional[CaptureResult]:
        """
        Force immediate capture for a pending vehicle.
        
        Useful when vehicle leaves tracking before window closes.
        
        Args:
            tracker_id: ByteTrack tracker ID
            all_tracked_bboxes: Dict of all current tracker_id -> bbox
            frame_width: Width of frame for boundary checking
            
        Returns:
            CaptureResult if vehicle was pending, None otherwise
        """
        if tracker_id in self.pending_captures:
            return self._finalize_capture(
                tracker_id, all_tracked_bboxes, frame_width
            )
        return None
    
    def has_pending(self, tracker_id: int) -> bool:
        """Check if a vehicle has a pending capture window."""
        return tracker_id in self.pending_captures
    
    def _finalize_capture(
        self,
        tracker_id: int,
        all_tracked_bboxes: Dict[int, np.ndarray],
        frame_width: int
    ) -> CaptureResult:
        """
        Finalize a capture.
        
        Args:
            tracker_id: ByteTrack tracker ID
            all_tracked_bboxes: Dict of all current tracker_id -> bbox (unused, kept for API compatibility)
            frame_width: Width of frame (unused, kept for API compatibility)
            
        Returns:
            CaptureResult with final bbox and metadata
        """
        candidate = self.pending_captures.pop(tracker_id)
        self.completed_count += 1
        
        aspect_ratio = candidate.best_aspect_ratio
        
        # Calculate crop height from best bbox (including padding that will be applied)
        # Padding is 20px on each side, so add 40px to raw bbox height
        x1, y1, x2, y2 = candidate.best_bbox
        raw_height = int(y2 - y1)
        crop_height = raw_height + 40  # Account for 20px padding on top and bottom
        
        # Categorize based on aspect ratio AND height
        # Small vehicle if: aspect < threshold OR height < height_threshold
        if aspect_ratio < self.config.small_vehicle_aspect_threshold or crop_height < self.config.small_vehicle_height_threshold:
            category = "small_vehicle"
            reason = f"aspect={aspect_ratio:.2f}" if aspect_ratio < self.config.small_vehicle_aspect_threshold else f"height={crop_height}px"
        else:
            category = "full_truck"
            reason = f"aspect={aspect_ratio:.2f}, height={crop_height}px"
        
        return CaptureResult(
            tracker_id=tracker_id,
            vehicle_id=candidate.vehicle_id,
            bbox=candidate.best_bbox,
            frame=candidate.best_frame,
            frame_time=candidate.best_time,
            aspect_ratio=aspect_ratio,
            category=category,
            mask=candidate.best_mask
        )
    
    @staticmethod
    def _calc_aspect_ratio(bbox: np.ndarray) -> float:
        """Calculate aspect ratio (width/height) of a bbox."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width / height if height > 0 else 0.0
    
    def get_pending_tracker_ids(self) -> List[int]:
        """Get list of tracker IDs with pending captures."""
        return list(self.pending_captures.keys())
    
    def finalize_all_pending(
        self,
        all_tracked_bboxes: Dict[int, np.ndarray],
        frame_width: int
    ) -> List[CaptureResult]:
        """
        Finalize all pending captures immediately.
        
        Called at end of video or when processing stops.
        
        Args:
            all_tracked_bboxes: Dict of all current tracker_id -> bbox (unused, kept for API)
            frame_width: Width of frame (unused, kept for API)
            
        Returns:
            List of CaptureResults for all pending vehicles
        """
        results = []
        pending_ids = list(self.pending_captures.keys())
        
        for tracker_id in pending_ids:
            result = self._finalize_capture(
                tracker_id, all_tracked_bboxes, frame_width
            )
            results.append(result)
            logger.info(
                f"Force-finalized pending capture for Truck {result.vehicle_id}"
            )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture manager statistics."""
        return {
            "enabled": self.config.enabled,
            "pending_captures": len(self.pending_captures),
            "completed_captures": self.completed_count,
        }
    
    def reset(self) -> None:
        """Reset manager state."""
        self.pending_captures.clear()
        self.completed_count = 0
