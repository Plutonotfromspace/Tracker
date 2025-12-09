"""
Direction-aware bounding box padding for truck capture.

This module expands bounding boxes to capture trailers that YOLO may miss,
particularly flatbed trailers that aren't recognized as part of the truck.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class PaddingConfig:
    """Configuration for bounding box padding."""
    enabled: bool = True
    base_padding_percent: float = 0.05  # 5% padding on all sides
    trailer_extension_percent: float = 0.8  # 80% extension toward trailer
    max_extension_percent: float = 1.0  # Max 100% extension to prevent capturing other trucks


class BBoxPadder:
    """
    Direction-aware bounding box padding.
    
    Expands bounding boxes based on vehicle movement direction to capture
    trailers that may be behind the detected cab.
    
    For trucks moving right (positive velocity_x): trailer is to the LEFT
    For trucks moving left (negative velocity_x): trailer is to the RIGHT
    """
    
    def __init__(self, config: Optional[PaddingConfig] = None):
        """
        Initialize the BBoxPadder.
        
        Args:
            config: PaddingConfig instance with padding parameters.
                   If None, uses default values.
        """
        self.config = config or PaddingConfig()
        logger.info(
            f"BBoxPadder initialized: enabled={self.config.enabled}, "
            f"base_padding={self.config.base_padding_percent:.0%}, "
            f"trailer_extension={self.config.trailer_extension_percent:.0%}"
        )
    
    def pad_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        velocity_x: float,
        frame_width: int,
        frame_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Apply direction-aware padding to a bounding box.
        
        Args:
            bbox: Original bounding box as (x1, y1, x2, y2)
            velocity_x: Horizontal velocity of the vehicle.
                       Positive = moving right, trailer is LEFT
                       Negative = moving left, trailer is RIGHT
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            Padded bounding box as (x1, y1, x2, y2), clamped to frame boundaries
        """
        if not self.config.enabled:
            return bbox
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        original_bbox = bbox
        
        # Calculate base padding (all sides)
        pad_x = int(width * self.config.base_padding_percent)
        pad_y = int(height * self.config.base_padding_percent)
        
        # Calculate trailer extension (capped at max)
        trailer_extension = int(width * min(
            self.config.trailer_extension_percent,
            self.config.max_extension_percent
        ))
        
        # Direction-aware trailer extension
        # velocity_x > 0 = moving right = trailer is LEFT (extend x1 leftward)
        # velocity_x < 0 = moving left = trailer is RIGHT (extend x2 rightward)
        # velocity_x == 0 = unknown direction, extend both ways moderately
        
        if velocity_x > 0:
            # Moving right, extend left for trailer
            x1 -= trailer_extension
            x2 += pad_x  # Just base padding on right
            logger.debug(f"Vehicle moving right, extending left by {trailer_extension}px")
        elif velocity_x < 0:
            # Moving left, extend right for trailer
            x1 -= pad_x  # Just base padding on left
            x2 += trailer_extension
            logger.debug(f"Vehicle moving left, extending right by {trailer_extension}px")
        else:
            # Unknown direction, extend both ways with half the trailer extension
            half_extension = trailer_extension // 2
            x1 -= half_extension
            x2 += half_extension
            logger.debug(f"Unknown direction, extending both ways by {half_extension}px")
        
        # Apply vertical base padding
        y1 -= pad_y
        y2 += pad_y
        
        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)
        
        padded_bbox = (int(x1), int(y1), int(x2), int(y2))
        
        # Log significant changes
        original_width = original_bbox[2] - original_bbox[0]
        new_width = padded_bbox[2] - padded_bbox[0]
        expansion_ratio = new_width / original_width if original_width > 0 else 1.0
        
        if expansion_ratio > 1.2:  # Log if expanded more than 20%
            logger.debug(
                f"BBox expanded: {original_bbox} -> {padded_bbox} "
                f"(width: {original_width} -> {new_width}, {expansion_ratio:.1%})"
            )
        
        return padded_bbox
    
    def calculate_velocity_x(
        self,
        current_pos: Tuple[float, float],
        last_pos: Optional[Tuple[float, float]]
    ) -> float:
        """
        Calculate horizontal velocity from position history.
        
        Args:
            current_pos: Current center position (x, y)
            last_pos: Previous center position (x, y), or None if no history
            
        Returns:
            Horizontal velocity (positive = moving right, negative = moving left)
        """
        if last_pos is None:
            return 0.0
        
        return current_pos[0] - last_pos[0]
