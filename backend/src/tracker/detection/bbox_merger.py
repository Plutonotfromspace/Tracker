"""
Bounding Box Merger for combining horizontally adjacent truck detections.

This module addresses the issue where YOLO detects truck cab and trailer
as separate objects. By merging horizontally adjacent "truck" detections,
we can capture the full truck+trailer in a single bounding box.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv

from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MergeConfig:
    """Configuration for bounding box merging."""
    
    max_horizontal_gap: int = 150
    """Maximum horizontal gap (pixels) between boxes to consider merging."""
    
    min_vertical_overlap_ratio: float = 0.4
    """Minimum vertical overlap ratio (0-1) required for merging."""
    
    truck_class_id: int = 7
    """COCO class ID for trucks."""
    
    enabled: bool = True
    """Whether merging is enabled."""


class BoundingBoxMerger:
    """
    Merges horizontally adjacent truck bounding boxes.
    
    When YOLO detects a truck cab and trailer as separate objects,
    this merger combines them into a single bounding box covering
    the entire truck.
    
    Algorithm:
    1. Filter detections to truck class only
    2. For each pair of truck boxes, check if they are:
       - Horizontally adjacent (gap < max_horizontal_gap)
       - Vertically aligned (overlap ratio > min_vertical_overlap_ratio)
    3. If criteria met, merge boxes using union of coordinates
    4. Keep highest confidence score from merged boxes
    
    Example:
        >>> merger = BoundingBoxMerger()
        >>> detections = sv.Detections.from_ultralytics(results)
        >>> merged = merger.merge_adjacent_boxes(detections)
    """
    
    def __init__(self, config: Optional[MergeConfig] = None) -> None:
        """
        Initialize the bounding box merger.
        
        Args:
            config: Merge configuration. Uses defaults if None.
        """
        self.config = config or MergeConfig()
        self.merge_count = 0
        self.total_processed = 0
    
    def _calculate_vertical_overlap(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """
        Calculate the vertical overlap ratio between two boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2].
            box2: Second bounding box [x1, y1, x2, y2].
        
        Returns:
            Overlap ratio (0-1) based on the smaller box's height.
        """
        # Get y ranges
        y1_min, y1_max = box1[1], box1[3]
        y2_min, y2_max = box2[1], box2[3]
        
        # Calculate overlap
        overlap_start = max(y1_min, y2_min)
        overlap_end = min(y1_max, y2_max)
        overlap = max(0, overlap_end - overlap_start)
        
        # Calculate ratio based on smaller box height
        height1 = y1_max - y1_min
        height2 = y2_max - y2_min
        min_height = min(height1, height2)
        
        if min_height <= 0:
            return 0.0
        
        return overlap / min_height
    
    def _calculate_horizontal_gap(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """
        Calculate the horizontal gap between two boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2].
            box2: Second bounding box [x1, y1, x2, y2].
        
        Returns:
            Horizontal gap in pixels. Negative if boxes overlap horizontally.
        """
        # box1 is to the left of box2
        if box1[2] <= box2[0]:
            return box2[0] - box1[2]
        # box2 is to the left of box1
        elif box2[2] <= box1[0]:
            return box1[0] - box2[2]
        # Boxes overlap horizontally
        else:
            return -1  # Overlapping
    
    def _should_merge(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> bool:
        """
        Determine if two boxes should be merged.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2].
            box2: Second bounding box [x1, y1, x2, y2].
        
        Returns:
            True if boxes should be merged.
        """
        # Check horizontal proximity
        h_gap = self._calculate_horizontal_gap(box1, box2)
        
        # Allow slight overlap (negative gap) or gap within threshold
        if h_gap > self.config.max_horizontal_gap:
            return False
        
        # Check vertical alignment
        v_overlap = self._calculate_vertical_overlap(box1, box2)
        if v_overlap < self.config.min_vertical_overlap_ratio:
            return False
        
        return True
    
    def _merge_two_boxes(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> np.ndarray:
        """
        Merge two bounding boxes into one.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2].
            box2: Second bounding box [x1, y1, x2, y2].
        
        Returns:
            Merged bounding box [x1, y1, x2, y2].
        """
        return np.array([
            min(box1[0], box2[0]),  # x1: leftmost
            min(box1[1], box2[1]),  # y1: topmost
            max(box1[2], box2[2]),  # x2: rightmost
            max(box1[3], box2[3])   # y2: bottommost
        ])
    
    def _find_merge_groups(
        self,
        truck_indices: List[int],
        xyxy: np.ndarray
    ) -> List[List[int]]:
        """
        Find groups of boxes that should be merged together.
        
        Uses a greedy approach to group adjacent boxes.
        
        Args:
            truck_indices: Indices of truck detections.
            xyxy: All bounding boxes.
        
        Returns:
            List of index groups, where each group should be merged.
        """
        if len(truck_indices) < 2:
            return [[i] for i in truck_indices]
        
        # Build adjacency list
        n = len(truck_indices)
        merged = [False] * n
        groups = []
        
        for i in range(n):
            if merged[i]:
                continue
            
            # Start a new group
            group = [truck_indices[i]]
            merged[i] = True
            
            # Find all boxes that should merge with this group
            for j in range(i + 1, n):
                if merged[j]:
                    continue
                
                # Check if box j should merge with any box in the group
                box_j = xyxy[truck_indices[j]]
                for idx in group:
                    box_i = xyxy[idx]
                    if self._should_merge(box_i, box_j):
                        group.append(truck_indices[j])
                        merged[j] = True
                        break
            
            groups.append(group)
        
        return groups
    
    def merge_adjacent_boxes(
        self,
        detections: sv.Detections
    ) -> sv.Detections:
        """
        Merge horizontally adjacent truck detections.
        
        Args:
            detections: Input detections from YOLO.
        
        Returns:
            Detections with adjacent truck boxes merged.
        """
        if not self.config.enabled or len(detections) == 0:
            return detections
        
        self.total_processed += 1
        
        # Find truck detections
        if detections.class_id is None or detections.confidence is None:
            return detections
        
        truck_mask = detections.class_id == self.config.truck_class_id
        truck_indices = np.where(truck_mask)[0].tolist()
        
        if len(truck_indices) < 2:
            return detections
        
        # Find groups to merge
        groups = self._find_merge_groups(truck_indices, detections.xyxy)
        
        # Check if any merging is needed
        if all(len(g) == 1 for g in groups):
            return detections
        
        # Build new detection arrays
        new_xyxy: List[np.ndarray] = []
        new_confidence: List[float] = []
        new_class_id: List[int] = []
        new_tracker_id: Optional[List] = [] if detections.tracker_id is not None else None
        
        # Track which original indices are kept/merged
        processed_indices: set = set()
        
        # Process merge groups
        for group in groups:
            processed_indices.update(group)
            
            if len(group) == 1:
                # Single box, keep as-is
                idx = group[0]
                new_xyxy.append(detections.xyxy[idx])
                new_confidence.append(float(detections.confidence[idx]))
                new_class_id.append(int(detections.class_id[idx]))
                if new_tracker_id is not None and detections.tracker_id is not None:
                    new_tracker_id.append(detections.tracker_id[idx])
            else:
                # Merge multiple boxes
                self.merge_count += 1
                
                # Merge bounding boxes
                merged_box = detections.xyxy[group[0]]
                for idx in group[1:]:
                    merged_box = self._merge_two_boxes(merged_box, detections.xyxy[idx])
                
                # Take maximum confidence
                max_conf = float(max(detections.confidence[idx] for idx in group))
                
                # Keep truck class
                merged_class = self.config.truck_class_id
                
                # Use tracker_id of largest box (likely the cab)
                if new_tracker_id is not None and detections.tracker_id is not None:
                    areas = [(detections.xyxy[i][2] - detections.xyxy[i][0]) * 
                             (detections.xyxy[i][3] - detections.xyxy[i][1]) 
                             for i in group]
                    largest_idx = group[int(np.argmax(areas))]
                    merged_tracker_id = detections.tracker_id[largest_idx]
                    new_tracker_id.append(merged_tracker_id)
                
                new_xyxy.append(merged_box)
                new_confidence.append(max_conf)
                new_class_id.append(merged_class)
                
                logger.debug(
                    f"Merged {len(group)} truck boxes into one "
                    f"(gap detection enabled)"
                )
        
        # Add non-truck detections unchanged
        for i in range(len(detections)):
            if i not in processed_indices:
                new_xyxy.append(detections.xyxy[i])
                new_confidence.append(float(detections.confidence[i]))
                new_class_id.append(int(detections.class_id[i]))
                if new_tracker_id is not None and detections.tracker_id is not None:
                    new_tracker_id.append(detections.tracker_id[i])
        
        # Create new Detections object
        return sv.Detections(
            xyxy=np.array(new_xyxy) if new_xyxy else np.empty((0, 4)),
            confidence=np.array(new_confidence) if new_confidence else np.empty(0),
            class_id=np.array(new_class_id) if new_class_id else np.empty(0, dtype=int),
            tracker_id=np.array(new_tracker_id) if new_tracker_id else None
        )
    
    def get_stats(self) -> dict:
        """Get merger statistics."""
        return {
            "total_frames_processed": self.total_processed,
            "total_merges": self.merge_count,
            "config": {
                "max_horizontal_gap": self.config.max_horizontal_gap,
                "min_vertical_overlap_ratio": self.config.min_vertical_overlap_ratio,
                "enabled": self.config.enabled
            }
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.merge_count = 0
        self.total_processed = 0
