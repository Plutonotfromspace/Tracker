"""
Segmentation mask generator for truck detection.

Converts YOLO segmentation polygon contours to binary masks
for pixel-perfect truck isolation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MaskData:
    """
    Container for segmentation mask data associated with a detection.
    
    Attributes:
        polygon: Original polygon contour from YOLO (N x 2 array of x,y points)
        binary_mask: Full-frame binary mask (H x W, 0=background, 255=object)
        bbox: Associated bounding box [x1, y1, x2, y2]
        tracker_id: Tracker ID if available
    """
    polygon: np.ndarray
    binary_mask: np.ndarray
    bbox: np.ndarray
    tracker_id: Optional[int] = None


class MaskGenerator:
    """
    Generates binary segmentation masks from YOLO polygon outputs.
    
    Converts polygon contours (result.masks.xy) to filled binary masks
    that can be saved as images or used for further processing.
    
    Example:
        >>> generator = MaskGenerator()
        >>> # After YOLO inference with seg model
        >>> masks_data = generator.extract_masks_from_results(results, detections)
        >>> for mask_data in masks_data.values():
        ...     cv2.imwrite("mask.png", mask_data.binary_mask)
    """
    
    def __init__(self) -> None:
        """Initialize the mask generator."""
        self.masks_generated = 0
        self.masks_failed = 0
    
    def polygon_to_binary_mask(
        self,
        polygon: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert a polygon contour to a binary mask.
        
        Args:
            polygon: Polygon points as Nx2 array of (x, y) coordinates.
            frame_shape: Shape of the frame (height, width).
        
        Returns:
            Binary mask with 255 for object pixels, 0 for background.
        """
        height, width = frame_shape[:2]
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if polygon is None or len(polygon) < 3:
            logger.warning("Invalid polygon: insufficient points")
            return mask
        
        # Convert polygon to proper format for cv2.drawContours
        # Expected format: (N, 1, 2) where N is number of points
        contour = polygon.astype(np.int32).reshape(-1, 1, 2)
        
        # Fill the polygon
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        
        return mask
    
    def extract_masks_from_results(
        self,
        yolo_results,
        detections,
        frame_shape: Tuple[int, int]
    ) -> Dict[int, MaskData]:
        """
        Extract segmentation masks from YOLO results and associate with tracker IDs.
        
        Args:
            yolo_results: Raw YOLO results object from model inference.
            detections: Supervision Detections object with tracker IDs.
            frame_shape: Shape of the frame (height, width).
        
        Returns:
            Dictionary mapping tracker_id to MaskData objects.
        """
        masks_by_tracker: Dict[int, MaskData] = {}
        
        # Check if masks are available
        if yolo_results.masks is None:
            logger.debug("No masks in YOLO results (using detection-only model?)")
            return masks_by_tracker
        
        # Get polygon masks from results
        polygons = yolo_results.masks.xy  # List of Nx2 arrays
        
        if len(polygons) == 0:
            logger.debug("No polygon masks found")
            return masks_by_tracker
        
        # Match masks to detections by index
        # YOLO returns masks in same order as boxes
        num_masks = len(polygons)
        num_detections = len(detections.xyxy)
        
        # Note: num_masks != num_detections is normal when filtering detections
        # (e.g., filtering to trucks only while YOLO sees all objects)
        if num_masks != num_detections:
            logger.debug(
                f"Mask/detection count: {num_masks} masks, "
                f"{num_detections} filtered detections"
            )
        
        # Process each mask
        for i, polygon in enumerate(polygons):
            if i >= num_detections:
                break
            
            # Get tracker ID if available
            tracker_id = None
            if detections.tracker_id is not None and i < len(detections.tracker_id):
                tracker_id = detections.tracker_id[i]
            
            if tracker_id is None:
                continue
            
            try:
                # Generate binary mask
                binary_mask = self.polygon_to_binary_mask(polygon, frame_shape)
                
                # Create MaskData
                mask_data = MaskData(
                    polygon=polygon,
                    binary_mask=binary_mask,
                    bbox=detections.xyxy[i],
                    tracker_id=tracker_id
                )
                
                masks_by_tracker[tracker_id] = mask_data
                self.masks_generated += 1
                
            except Exception as e:
                logger.error(f"Failed to generate mask for detection {i}: {e}")
                self.masks_failed += 1
        
        return masks_by_tracker
    
    def crop_mask_to_bbox(
        self,
        mask: np.ndarray,
        bbox: np.ndarray,
        padding: int = 20
    ) -> np.ndarray:
        """
        Crop a full-frame mask to the bounding box region.
        
        Args:
            mask: Full-frame binary mask.
            bbox: Bounding box [x1, y1, x2, y2].
            padding: Pixels to add around the bounding box.
        
        Returns:
            Cropped mask matching the crop dimensions.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        frame_h, frame_w = mask.shape[:2]
        
        # Calculate crop bounds with padding
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(frame_w, x2 + padding)
        crop_y2 = min(frame_h, y2 + padding)
        
        # Crop the mask
        cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        
        return cropped_mask.copy()
    
    def get_stats(self) -> Dict[str, int]:
        """Get mask generation statistics."""
        return {
            "masks_generated": self.masks_generated,
            "masks_failed": self.masks_failed,
            "total_attempts": self.masks_generated + self.masks_failed
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.masks_generated = 0
        self.masks_failed = 0
    
    def reset(self) -> None:
        """Reset the mask generator state (alias for reset_stats)."""
        self.reset_stats()
