"""
SAM2 (Segment Anything Model 2) mask generator for pixel-perfect truck segmentation.

Uses Meta's SAM2 model with bounding box prompts from YOLO detections
to generate high-quality, pixel-perfect segmentation masks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MaskData:
    """
    Container for segmentation mask data associated with a detection.
    
    Attributes:
        binary_mask: Full-frame binary mask (H x W, 0=background, 255=object)
        bbox: Associated bounding box [x1, y1, x2, y2]
        tracker_id: Tracker ID if available
        iou_score: SAM2 IoU prediction score (quality confidence)
    """
    binary_mask: np.ndarray
    bbox: np.ndarray
    tracker_id: Optional[int] = None
    iou_score: float = 0.0


class SAM2MaskGenerator:
    """
    Generates pixel-perfect segmentation masks using SAM2 with bounding box prompts.
    
    SAM2 (Segment Anything Model 2) provides significantly better mask quality
    compared to YOLO-seg by using a dedicated segmentation architecture
    optimized for precise boundary detection.
    
    Example:
        >>> generator = SAM2MaskGenerator()
        >>> # After YOLO detection
        >>> masks_data = generator.generate_masks(frame, detections)
        >>> for mask_data in masks_data.values():
        ...     cv2.imwrite("mask.png", mask_data.binary_mask)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/sam2-hiera-large",
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the SAM2 mask generator.
        
        Args:
            model_name: Hugging Face model ID. Options:
                - "facebook/sam2-hiera-tiny" (faster, lower quality)
                - "facebook/sam2-hiera-small"
                - "facebook/sam2-hiera-base-plus"
                - "facebook/sam2-hiera-large" (best quality, slower)
            device: Device to run on ("cuda", "cpu", or None for auto-detect).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self._initialized = False
        
        # Statistics
        self.masks_generated = 0
        self.masks_failed = 0
        
        logger.info(f"SAM2MaskGenerator created (model: {model_name}, device: {self.device})")
    
    def _lazy_init(self) -> None:
        """Lazily initialize the SAM2 model on first use."""
        if self._initialized:
            return
        
        logger.info(f"Loading SAM2 model: {self.model_name}...")
        
        try:
            # Suppress SAM2's verbose logging (it logs to root logger)
            import logging
            root_logger = logging.getLogger()
            original_level = root_logger.level
            root_logger.setLevel(logging.WARNING)
            
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
            self._initialized = True
            
            # Restore original log level
            root_logger.setLevel(original_level)
            
            logger.info("SAM2 model loaded successfully")
            
        except ImportError as e:
            logger.error(
                f"SAM2 not installed. Install with: pip install sam2\n"
                f"Error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            raise
    
    def generate_masks(
        self,
        frame: np.ndarray,
        detections,
    ) -> Dict[int, MaskData]:
        """
        Generate pixel-perfect masks for all detections using SAM2.
        
        Args:
            frame: BGR frame (from cv2.VideoCapture).
            detections: Supervision Detections object with tracker IDs and bboxes.
        
        Returns:
            Dictionary mapping tracker_id to MaskData objects.
        """
        self._lazy_init()
        
        masks_by_tracker: Dict[int, MaskData] = {}
        
        if detections is None or len(detections.xyxy) == 0:
            return masks_by_tracker
        
        # Convert BGR to RGB for SAM2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Collect boxes and tracker IDs
        boxes = []
        tracker_ids = []
        
        for i, bbox in enumerate(detections.xyxy):
            tracker_id = None
            if detections.tracker_id is not None and i < len(detections.tracker_id):
                tracker_id = detections.tracker_id[i]
            
            if tracker_id is not None:
                boxes.append(bbox)
                tracker_ids.append(tracker_id)
        
        if not boxes:
            return masks_by_tracker
        
        # Convert to numpy array
        boxes_array = np.array(boxes, dtype=np.float32)
        
        try:
            # Use autocast for mixed precision on CUDA
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        masks_by_tracker = self._predict_masks(
                            frame_rgb, boxes_array, tracker_ids, detections.xyxy
                        )
                else:
                    masks_by_tracker = self._predict_masks(
                        frame_rgb, boxes_array, tracker_ids, detections.xyxy
                    )
                    
        except Exception as e:
            logger.error(f"SAM2 prediction failed: {e}")
            self.masks_failed += len(boxes)
        
        return masks_by_tracker
    
    def _predict_masks(
        self,
        frame_rgb: np.ndarray,
        boxes: np.ndarray,
        tracker_ids: List[int],
        all_bboxes: np.ndarray
    ) -> Dict[int, MaskData]:
        """
        Internal method to run SAM2 prediction.
        
        Args:
            frame_rgb: RGB frame.
            boxes: Array of bounding boxes [N, 4] in xyxy format.
            tracker_ids: List of tracker IDs corresponding to boxes.
            all_bboxes: Original bbox array for reference.
        
        Returns:
            Dictionary mapping tracker_id to MaskData.
        """
        masks_by_tracker: Dict[int, MaskData] = {}
        
        # Temporarily suppress SAM2's verbose logging during inference
        import logging
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.WARNING)
        
        try:
            # Set the image (computes embeddings once)
            self.predictor.set_image(frame_rgb)
            
            # Predict masks for all boxes at once
            masks, iou_scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False  # Single best mask per box
            )
        finally:
            # Restore original log level
            root_logger.setLevel(original_level)
        
        # Process results
        for i, (mask, iou_score, tracker_id) in enumerate(zip(masks, iou_scores, tracker_ids)):
            try:
                # SAM2 returns boolean mask, convert to uint8
                # mask shape is (1, H, W) or (H, W)
                if mask.ndim == 3:
                    binary_mask = (mask[0] * 255).astype(np.uint8)
                else:
                    binary_mask = (mask * 255).astype(np.uint8)
                
                # Get the corresponding bbox
                bbox_idx = tracker_ids.index(tracker_id)
                bbox = boxes[bbox_idx]
                
                # Create MaskData
                mask_data = MaskData(
                    binary_mask=binary_mask,
                    bbox=bbox,
                    tracker_id=tracker_id,
                    iou_score=float(iou_score[0]) if iou_score.ndim > 0 else float(iou_score)
                )
                
                masks_by_tracker[tracker_id] = mask_data
                self.masks_generated += 1
                
                logger.debug(
                    f"Generated SAM2 mask for tracker {tracker_id}, "
                    f"IoU score: {mask_data.iou_score:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Failed to process mask for tracker {tracker_id}: {e}")
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
        """Reset the mask generator state."""
        self.reset_stats()
