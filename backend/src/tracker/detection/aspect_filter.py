"""
Aspect ratio-based categorization for truck detection.

Categorizes detections based on bounding box aspect ratio to identify
vehicle types (full_truck, small_vehicle, cab_only) without rejecting any data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)


class TruckCategory(Enum):
    """Categories for truck detection based on aspect ratio."""
    FULL_TRUCK = "full_truck"      # Ratio >= 2.0 - cab + trailer clearly visible
    SMALL_VEHICLE = "small_vehicle" # Ratio 1.0-2.0 - cab only or short trailer
    CAB_ONLY = "cab_only"          # Ratio < 1.0 - narrow detection, likely partial


@dataclass
class CropResult:
    """Result of a crop operation with aspect ratio categorization."""
    
    image: Optional[np.ndarray]
    accepted: bool
    aspect_ratio: float
    category: TruckCategory = TruckCategory.FULL_TRUCK
    
    @property
    def is_trailer_visible(self) -> bool:
        """Check if the trailer is considered visible based on aspect ratio."""
        return self.category == TruckCategory.FULL_TRUCK


class AspectRatioFilter:
    """
    Aspect ratio-based categorization for truck crops.
    
    Categorizes vehicles based on aspect ratio without rejecting any.
    Trucks with trailers are wide (~2.0+), small vehicles are narrower (~1.0-2.0).
    
    When accept_all=True (default), all trucks are accepted and categorized.
    When accept_all=False, narrow trucks are rejected (legacy behavior).
    
    Attributes:
        min_aspect_ratio: Threshold for categorizing as full_truck.
        accept_all: When True, accept all trucks regardless of ratio.
        accepted_count: Number of accepted crops.
        rejected_count: Number of rejected crops.
        category_counts: Count per TruckCategory.
    
    Example:
        >>> filter = AspectRatioFilter(min_aspect_ratio=2.0, accept_all=True)
        >>> result = filter.get_truck_crop(frame, bbox)
        >>> print(f"Category: {result.category.value}")  # "small_vehicle" or "full_truck"
    """
    
    DEFAULT_MIN_ASPECT_RATIO = 2.0
    DEFAULT_PADDING = 20
    
    def __init__(
        self, 
        min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO,
        accept_all: bool = True
    ) -> None:
        """
        Initialize the aspect ratio filter.
        
        Args:
            min_aspect_ratio: Aspect ratio threshold for full_truck category.
            accept_all: If True, accept all trucks (categorize, do not reject).
        """
        self.min_aspect_ratio = min_aspect_ratio
        self.accept_all = accept_all
        self.accepted_count = 0
        self.rejected_count = 0
        self.category_counts = {cat: 0 for cat in TruckCategory}
    
    def _categorize(self, aspect_ratio: float) -> TruckCategory:
        """Determine truck category based on aspect ratio."""
        if aspect_ratio >= self.min_aspect_ratio:
            return TruckCategory.FULL_TRUCK
        elif aspect_ratio >= 1.0:
            return TruckCategory.SMALL_VEHICLE
        else:
            return TruckCategory.CAB_ONLY
    
    def get_truck_crop(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        padding: int = DEFAULT_PADDING
    ) -> CropResult:
        """
        Get a crop of the truck with aspect ratio categorization.
        
        Args:
            frame: Full video frame as numpy array.
            bbox: Bounding box as [x1, y1, x2, y2] array.
            padding: Pixels to add around the bounding box.
        
        Returns:
            CropResult with the cropped image, acceptance status,
            aspect ratio, and truck category.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate crop bounds with padding
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(frame_w, x2 + padding)
        crop_y2 = min(frame_h, y2 + padding)
        
        width = crop_x2 - crop_x1
        height = crop_y2 - crop_y1
        aspect_ratio = width / height if height > 0 else 0
        
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        category = self._categorize(aspect_ratio)
        self.category_counts[category] += 1
        
        # Determine acceptance based on accept_all setting
        if self.accept_all:
            # Accept all trucks, just categorize them
            self.accepted_count += 1
            return CropResult(
                image=cropped.copy(),
                accepted=True,
                aspect_ratio=aspect_ratio,
                category=category
            )
        else:
            # Legacy behavior: reject narrow trucks
            if aspect_ratio >= self.min_aspect_ratio:
                self.accepted_count += 1
                logger.info(
                    f"ACCEPTED: Aspect ratio {aspect_ratio:.2f} >= "
                    f"{self.min_aspect_ratio}"
                )
                return CropResult(
                    image=cropped.copy(),
                    accepted=True,
                    aspect_ratio=aspect_ratio,
                    category=category
                )
            else:
                self.rejected_count += 1
                logger.info(
                    f"REJECTED: Aspect ratio {aspect_ratio:.2f} < "
                    f"{self.min_aspect_ratio} - trailer not visible"
                )
                return CropResult(
                    image=cropped.copy(),  # Still provide image for analysis
                    accepted=False,
                    aspect_ratio=aspect_ratio,
                    category=category
                )
    
    def reset_counts(self) -> None:
        """Reset the accepted/rejected counters."""
        self.accepted_count = 0
        self.rejected_count = 0
        self.category_counts = {cat: 0 for cat in TruckCategory}
    
    @property
    def total_processed(self) -> int:
        """Total number of crops processed."""
        return self.accepted_count + self.rejected_count
    
    @property
    def acceptance_rate(self) -> float:
        """Percentage of accepted crops (0-100)."""
        total = self.total_processed
        return (self.accepted_count / total * 100) if total > 0 else 0
    
    def log_summary(self) -> None:
        """Log a summary of accepted/rejected counts and categories."""
        pass
    
    def get_stats(self) -> dict:
        """
        Get filter statistics as a dictionary.
        
        Returns:
            Dictionary with filter statistics.
        """
        return {
            "min_aspect_ratio": self.min_aspect_ratio,
            "accept_all": self.accept_all,
            "total_processed": self.total_processed,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "acceptance_rate": self.acceptance_rate,
            "categories": {cat.value: count for cat, count in self.category_counts.items()}
        }
