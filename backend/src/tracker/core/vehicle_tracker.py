"""
Vehicle tracking system with LINE CROSSING detection.

Provides more reliable entry/exit logging compared to zone-based detection.
"""

import os
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from src.tracker.capture.delayed_capture import (
    CaptureResult,
    DelayedCaptureConfig,
    DelayedCaptureManager,
)
from src.tracker.config.settings import Settings, get_settings
from src.tracker.data.database import get_session_context
from src.tracker.data import crud
from src.tracker.data import (
    TruckMetadata,
    MetadataWriter,
    CaptureInfo,
    FileInfo,
    DetectionInfo,
    CropInfo,
    MaskInfo,
    CropCategory,
)
from src.tracker.detection.aspect_filter import AspectRatioFilter, TruckCategory
from src.tracker.detection.bbox_merger import BoundingBoxMerger, MergeConfig
from src.tracker.detection.bbox_padder import BBoxPadder, PaddingConfig
from src.tracker.detection.sam2_mask_generator import SAM2MaskGenerator, MaskData
from src.tracker.utils.logging_config import get_logger, setup_logging
from src.tracker.utils.timestamp_reader import VideoTimestampReader

logger = get_logger(__name__)


class VehicleTracker:
    """
    Vehicle tracking system using YOLO detection and ByteTrack.
    
    Tracks vehicles crossing an entry line and logs their entry times.
    Saves debug images with bounding boxes for validation.
    
    Attributes:
        model: YOLO detection model.
        byte_tracker: ByteTrack tracker for multi-object tracking.
        aspect_filter: Filter for validating truck aspect ratios.
        timestamp_reader: OCR reader for extracting video timestamps.
        vehicle_log: List of logged vehicle entries.
        vehicle_counter: Counter for assigning sequential vehicle IDs.
    
    Example:
        >>> tracker = VehicleTracker()
        >>> frame = cv2.imread("frame.jpg")
        >>> annotated = tracker.process_frame(frame, datetime.now())
        >>> tracker.save_log("output.csv")
    """
    
    def __init__(self, settings: Optional[Settings] = None, enable_database: bool = True) -> None:
        """
        Initialize the vehicle tracker.
        
        Args:
            settings: Configuration settings. Uses global settings if None.
            enable_database: Whether to write results to database (default: True).
        """
        self.settings = settings or get_settings()
        self.enable_database = enable_database
        
        # Set up logging
        setup_logging()
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {self.settings.yolo_model}")
        self.model = YOLO(self.settings.yolo_model)
        
        # Set device (CUDA/CPU)
        logger.info(f"Using device: {self.settings.device}")
        self.device = self.settings.device
        
        # Initialize ByteTrack tracker
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=self.settings.track_activation_threshold,
            lost_track_buffer=self.settings.lost_track_buffer,
            minimum_matching_threshold=self.settings.minimum_matching_threshold,
            frame_rate=self.settings.tracker_frame_rate
        )
        
        # Entry line configuration
        self.entry_line_start = np.array(self.settings.entry_line[:2])
        self.entry_line_end = np.array(self.settings.entry_line[2:])
        
        # Track vehicle states
        self.vehicle_states: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "crossed_entry": False,
                "entry_time": None,
                "last_pos": None,
                "velocity_x": 0.0  # For direction-aware bbox padding
            }
        )
        
        # Vehicle log for CSV output
        self.vehicle_log: List[Dict[str, Any]] = []
        
        # Annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        
        # Counter for unique vehicles that crossed entry
        self.vehicle_counter = 0
        self.tracker_to_vehicle_id: Dict[int, int] = {}
        
        # Debug images directory (supports job-based or legacy paths)
        output_paths = self.settings.get_output_paths()
        self.debug_dir = str(output_paths["debug_images"])
        os.makedirs(self.debug_dir, exist_ok=True)
        logger.info(f"Debug images will be saved to: {self.debug_dir}/")
        if self.settings.job_id:
            logger.info(f"Job ID: {self.settings.job_id}")
        logger.info(f"Entry line: {self.settings.entry_line}")
        
        # Initialize aspect ratio filter
        self.aspect_filter = AspectRatioFilter(
            min_aspect_ratio=self.settings.min_aspect_ratio,
            accept_all=self.settings.accept_all_trucks
        )
        logger.info(
            f"Aspect ratio filter: min ratio = {self.aspect_filter.min_aspect_ratio}, "
            f"accept_all = {self.aspect_filter.accept_all}"
        )
        
        # Initialize bounding box padder for direction-aware trailer capture
        padding_config = PaddingConfig(
            enabled=self.settings.bbox_padding_enabled,
            base_padding_percent=self.settings.bbox_base_padding_percent,
            trailer_extension_percent=self.settings.bbox_trailer_extension_percent
        )
        self.bbox_padder = BBoxPadder(config=padding_config)
        
        # Initialize bounding box merger for combining cab+trailer detections
        merge_config = MergeConfig(
            max_horizontal_gap=self.settings.bbox_merge_max_gap,
            min_vertical_overlap_ratio=self.settings.bbox_merge_min_overlap,
            truck_class_id=7,  # COCO truck class
            enabled=self.settings.bbox_merge_enabled
        )
        self.bbox_merger = BoundingBoxMerger(config=merge_config)
        logger.info(
            f"BBox merger: enabled={merge_config.enabled}, "
            f"max_gap={merge_config.max_horizontal_gap}px"
        )
        
        # Initialize delayed capture manager for intelligent capture
        delayed_config = DelayedCaptureConfig(
            enabled=self.settings.delayed_capture_enabled,
            capture_delay_frames=self.settings.delayed_capture_window_frames,
            small_vehicle_aspect_threshold=self.settings.small_vehicle_aspect_threshold,
            small_vehicle_height_threshold=self.settings.small_vehicle_height_threshold,
        )
        self.delayed_capture = DelayedCaptureManager(config=delayed_config)
        
        # Frame counter for delayed capture timing
        self.current_frame_number = 0
        
        # Initialize timestamp reader for extracting time from video overlay
        self.timestamp_reader = VideoTimestampReader()
        self.current_video_time: Optional[datetime] = None
        
        # Initialize SAM2 mask generator for pixel-perfect segmentation masks
        self.mask_generator = None  # Lazy loaded to avoid slow startup
        self.save_masks = self.settings.save_segmentation_masks
        self.current_frame_masks: Dict[int, MaskData] = {}
        
        if self.save_masks:
            logger.info(
                f"SAM2 segmentation enabled (model: {self.settings.sam2_model}). "
                f"Model will be loaded on first detection."
            )
        
        # Initialize metadata writer for structured JSON output
        self.metadata_writer = MetadataWriter()
        logger.info("Metadata writer initialized")
        
        # Track video source for metadata
        self.current_video_source: Optional[str] = None
    
    def set_video_source(self, video_source: str) -> None:
        """
        Set the current video source for metadata tracking.
        
        Args:
            video_source: Path or URL of the video being processed.
        """
        self.current_video_source = video_source
        logger.debug(f"Video source set to: {video_source}")
    
    def get_vehicle_id(self, tracker_id: int) -> int:
        """
        Get or create a sequential vehicle ID for a tracker ID.
        
        Args:
            tracker_id: ByteTrack tracker ID.
        
        Returns:
            Sequential vehicle ID.
        """
        if tracker_id not in self.tracker_to_vehicle_id:
            self.vehicle_counter += 1
            self.tracker_to_vehicle_id[tracker_id] = self.vehicle_counter
        return self.tracker_to_vehicle_id[tracker_id]
    
    @staticmethod
    def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        """Check if three points are in counter-clockwise order."""
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    def _lines_intersect(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray
    ) -> bool:
        """Check if line segment AB intersects with line segment CD."""
        return (
            self._ccw(a, c, d) != self._ccw(b, c, d) and
            self._ccw(a, b, c) != self._ccw(a, b, d)
        )
    
    def _crossed_line(
        self,
        prev_pos: Optional[np.ndarray],
        curr_pos: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> bool:
        """Check if movement from prev_pos to curr_pos crosses the line."""
        if prev_pos is None:
            return False
        return self._lines_intersect(prev_pos, curr_pos, line_start, line_end)
    
    def _save_rejected_debug_image(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        vehicle_id: int,
        event_type: str,
        frame_time: datetime,
        crop_result: Any,
        truck_folder: str
    ) -> None:
        """Save debug images for rejected (cab-only) detections for analysis."""
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Draw bounding box in RED to indicate rejection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # Add rejection label
        label = f"REJECTED Truck {vehicle_id} (ratio={crop_result.aspect_ratio:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw entry line
        cv2.line(
            frame,
            tuple(self.entry_line_start.astype(int)),
            tuple(self.entry_line_end.astype(int)),
            (0, 255, 0), 3
        )
        
        # Save full frame
        filename = f"{truck_folder}/REJECTED_{event_type.lower()}_{frame_time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        
        # Also save the crop (even though rejected) for analysis
        frame_h, frame_w = frame.shape[:2]
        padding = 20
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(frame_w, x2 + padding)
        crop_y2 = min(frame_h, y2 + padding)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        crop_filename = f"{truck_folder}/REJECTED_{event_type.lower()}_{frame_time.strftime('%H%M%S')}_crop.jpg"
        cv2.imwrite(crop_filename, crop)
    
    def _save_debug_image(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        vehicle_id: int,
        event_type: str,
        frame_time: datetime
    ) -> bool:
        """
        Save a debug image with bounding box around the truck.
        
        Args:
            frame: Full video frame.
            bbox: Bounding box [x1, y1, x2, y2].
            vehicle_id: Sequential vehicle ID.
            event_type: Event type ("ENTRY" or "EXIT").
            frame_time: Timestamp of the frame.
        
        Returns:
            True if accepted, False if rejected.
        """
        # Check aspect ratio - now categorizes instead of rejecting (when accept_all=True)
        crop_result = self.aspect_filter.get_truck_crop(frame, bbox)
        
        # Create truck-specific folder
        debug_frame = frame.copy()
        truck_folder = f"{self.debug_dir}/Truck{vehicle_id:03d}"
        os.makedirs(truck_folder, exist_ok=True)
        
        if not crop_result.accepted:
            # Only reaches here if accept_all=False (legacy mode)
            logger.info(
                f"Truck {vehicle_id}: REJECTED "
                f"(aspect ratio {crop_result.aspect_ratio:.2f} < "
                f"{self.aspect_filter.min_aspect_ratio})"
            )
            self._save_rejected_debug_image(
                debug_frame, bbox, vehicle_id, event_type, frame_time, 
                crop_result, truck_folder
            )
            return False
        
        # Accepted - log with category
        category_str = crop_result.category.value
        logger.info(
            f"Truck {vehicle_id}: ACCEPTED [{category_str}] "
            f"(aspect ratio {crop_result.aspect_ratio:.2f}) - saved to {truck_folder}"
        )
        
        # Save the cropped truck image
        if crop_result.image is not None:
            crop_filename = (
                f"{truck_folder}/{event_type.lower()}_"
                f"{frame_time.strftime('%H%M%S')}_crop.jpg"
            )
            cv2.imwrite(crop_filename, crop_result.image)
        
        # Draw the entry line (green)
        cv2.line(
            debug_frame,
            tuple(self.entry_line_start.astype(int)),
            tuple(self.entry_line_end.astype(int)),
            (0, 255, 0),
            3
        )
        cv2.putText(
            debug_frame,
            "ENTRY",
            (int(self.entry_line_start[0]) + 10, int(self.entry_line_start[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Get bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Draw thick bounding box around the truck
        color = (0, 255, 0) if event_type == "ENTRY" else (0, 0, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 4)
        
        # Add label above the box
        label = f"Truck {vehicle_id} - {event_type}"
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        )
        cv2.rectangle(
            debug_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        cv2.putText(
            debug_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )
        
        # Add timestamp at top of image
        timestamp_str = frame_time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(
            debug_frame,
            f"{event_type}: {timestamp_str}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )
        
        # Save the full debug image
        filename = (
            f"{truck_folder}/{event_type.lower()}_"
            f"{frame_time.strftime('%H%M%S')}.jpg"
        )
        cv2.imwrite(filename, debug_frame)
        
        return True
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_time: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Process a single frame for vehicle detection and line crossing.
        
        Args:
            frame: Video frame as numpy array (BGR format).
            frame_time: Timestamp for the frame. Uses current time if None.
        
        Returns:
            Annotated frame with detections and tracking info.
        """
        if frame_time is None:
            frame_time = datetime.now()
        
        # Increment frame counter for delayed capture timing
        self.current_frame_number += 1
        
        # Run YOLO detection
        results = self.model(
            frame,
            conf=self.settings.confidence_threshold,
            iou=self.settings.iou_threshold,
            classes=self.settings.vehicle_classes,
            device=self.device,
            verbose=False
        )[0]
        
        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Merge adjacent truck boxes (cab + trailer)
        # This combines separate detections for cab and trailer into one bbox
        detections = self.bbox_merger.merge_adjacent_boxes(detections)
        
        # Apply tracking
        detections = self.byte_tracker.update_with_detections(detections)
        
        # Generate pixel-perfect segmentation masks with SAM2 if enabled
        frame_h, frame_w = frame.shape[:2]
        if self.save_masks and len(detections.xyxy) > 0:
            # Lazy load SAM2 model on first use
            if self.mask_generator is None:
                logger.info("Loading SAM2 model for first detection...")
                self.mask_generator = SAM2MaskGenerator(
                    model_name=self.settings.sam2_model,
                    device=self.settings.device
                )
            self.current_frame_masks = self.mask_generator.generate_masks(
                frame, detections
            )
        
        # Build dict of all currently tracked bboxes for collision detection
        all_tracked_bboxes: Dict[int, np.ndarray] = {}
        if detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is not None:
                    all_tracked_bboxes[tracker_id] = detections.xyxy[i]
        
        # Get frame dimensions for boundary checking
        frame_h, frame_w = frame.shape[:2]
        
        # Process each tracked vehicle
        if detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is None:
                    continue
                
                state = self.vehicle_states[tracker_id]
                bbox = detections.xyxy[i]
                
                # Get bottom center of bounding box (where wheels are)
                curr_pos = np.array([
                    (bbox[0] + bbox[2]) / 2,  # center x
                    bbox[3]  # bottom y
                ])
                
                prev_pos = state["last_pos"]
                
                # Calculate velocity for direction-aware processing
                if prev_pos is not None:
                    state["velocity_x"] = curr_pos[0] - prev_pos[0]
                
                # Check for ENTRY line crossing
                if not state["crossed_entry"]:
                    if self._crossed_line(
                        prev_pos, curr_pos,
                        self.entry_line_start, self.entry_line_end
                    ):
                        state["crossed_entry"] = True
                        state["entry_time"] = frame_time
                        vehicle_id = self.get_vehicle_id(tracker_id)
                        
                        # Get mask for this tracker if available
                        current_mask = None
                        if self.save_masks and tracker_id in self.current_frame_masks:
                            current_mask = self.current_frame_masks[tracker_id].binary_mask
                        
                        # Start delayed capture window for intelligent capture
                        if self.delayed_capture.config.enabled:
                            self.delayed_capture.start_capture_window(
                                tracker_id=tracker_id,
                                vehicle_id=vehicle_id,
                                frame=frame,
                                bbox=bbox,
                                frame_time=frame_time,
                                current_frame=self.current_frame_number,
                                velocity_x=state["velocity_x"],
                                mask=current_mask
                            )
                        else:
                            # Immediate capture (legacy mode)
                            self._immediate_capture(
                                frame, bbox, vehicle_id, "ENTRY", 
                                frame_time, tracker_id, state["velocity_x"]
                            )
                
                # Update delayed capture windows
                if self.delayed_capture.has_pending(tracker_id):
                    # Get current mask for this tracker if available
                    current_mask = None
                    if self.save_masks and tracker_id in self.current_frame_masks:
                        current_mask = self.current_frame_masks[tracker_id].binary_mask
                    
                    result = self.delayed_capture.update(
                        tracker_id=tracker_id,
                        frame=frame,
                        bbox=bbox,
                        frame_time=frame_time,
                        current_frame=self.current_frame_number,
                        all_tracked_bboxes=all_tracked_bboxes,
                        frame_width=frame_w,
                        velocity_x=state["velocity_x"],
                        mask=current_mask
                    )
                    
                    if result is not None:
                        # Capture window closed, process the result
                        self._process_capture_result(result)
                
                # Update last position
                state["last_pos"] = curr_pos
        
        # Check for lost vehicles with pending captures and force-capture them
        self._check_lost_vehicles(all_tracked_bboxes, frame_w)
        
        # Create labels for visualization
        labels = self._create_labels(detections)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, detections, labels)
        
        return annotated_frame
    
    def _check_lost_vehicles(
        self,
        current_tracked_bboxes: Dict[int, np.ndarray],
        frame_width: int
    ) -> None:
        """
        Check for vehicles with pending captures that are no longer tracked.
        
        Force-captures them immediately to avoid losing data.
        """
        pending_ids = self.delayed_capture.get_pending_tracker_ids()
        
        for tracker_id in pending_ids:
            if tracker_id not in current_tracked_bboxes:
                # Vehicle lost tracking, force capture
                result = self.delayed_capture.force_capture(
                    tracker_id, current_tracked_bboxes, frame_width
                )
                if result is not None:
                    self._process_capture_result(result)
    
    def finalize_all_pending(self, frame_width: int = 1920) -> None:
        """
        Finalize all pending captures at end of video.
        
        Called when video processing ends to ensure all crossed
        trucks are logged.
        
        Args:
            frame_width: Frame width for boundary checking
        """
        results = self.delayed_capture.finalize_all_pending(
            all_tracked_bboxes={},  # No other vehicles at end
            frame_width=frame_width
        )
        
        for result in results:
            self._process_capture_result(result)
        
        if results:
            logger.info(f"Finalized {len(results)} pending captures at end of video")
    
    def _immediate_capture(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        vehicle_id: int,
        event_type: str,
        frame_time: datetime,
        tracker_id: int,
        velocity_x: float
    ) -> None:
        """
        Perform immediate capture without delay (legacy mode).
        
        Used when delayed_capture_enabled=False.
        """
        frame_h, frame_w = frame.shape[:2]
        
        # Apply direction-aware bbox padding for trailer capture
        padded_bbox = self.bbox_padder.pad_bbox(
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            velocity_x=velocity_x,
            frame_width=frame_w,
            frame_height=frame_h
        )
        padded_bbox = np.array(padded_bbox)
        
        accepted = self._save_debug_image(
            frame, padded_bbox, vehicle_id, event_type, frame_time
        )
        if accepted:
            self.vehicle_log.append({
                "vehicle_id": vehicle_id,
                "tracker_id": tracker_id,
                "entry_time": frame_time.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    def _process_capture_result(self, result: CaptureResult) -> None:
        """
        Process a completed delayed capture result.
        
        Saves the captured image and logs the vehicle entry.
        """
        # Create truck-specific folder
        truck_folder = f"{self.debug_dir}/Truck{result.vehicle_id:03d}"
        os.makedirs(truck_folder, exist_ok=True)
        
        # Get crop using aspect filter for consistency
        crop_result = self.aspect_filter.get_truck_crop(
            result.frame, result.bbox
        )
        
        # Save the cropped truck image
        if crop_result.image is not None:
            crop_filename = (
                f"{truck_folder}/entry_"
                f"{result.frame_time.strftime('%H%M%S')}_crop.jpg"
            )
            cv2.imwrite(crop_filename, crop_result.image)
        
        # Save the masked image (truck pixels with background blacked out)
        if self.save_masks and result.mask is not None and crop_result.image is not None:
            # Crop the mask to the same region as the crop image
            cropped_mask = self.mask_generator.crop_mask_to_bbox(
                result.mask, result.bbox, padding=20
            )
            
            # Apply mask to crop: keep truck pixels, black out background
            # Ensure mask and crop have same dimensions
            if cropped_mask.shape[:2] == crop_result.image.shape[:2]:
                # Create 3-channel mask for color image
                mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)
                # Apply mask: where mask is white (255), keep original; where black (0), make black
                masked_crop = cv2.bitwise_and(crop_result.image, mask_3ch)
                
                mask_filename = (
                    f"{truck_folder}/entry_"
                    f"{result.frame_time.strftime('%H%M%S')}_mask.png"
                )
                cv2.imwrite(mask_filename, masked_crop)
                logger.debug(f"Truck {result.vehicle_id}: Saved masked truck image")
            else:
                logger.warning(
                    f"Truck {result.vehicle_id}: Mask/crop size mismatch "
                    f"({cropped_mask.shape} vs {crop_result.image.shape})"
                )
        
        # Save full debug frame
        debug_frame = result.frame.copy()
        
        # Draw the entry line (green)
        cv2.line(
            debug_frame,
            tuple(self.entry_line_start.astype(int)),
            tuple(self.entry_line_end.astype(int)),
            (0, 255, 0),
            3
        )
        
        # Get bounding box
        x1, y1, x2, y2 = result.bbox.astype(int)
        
        # Draw thick bounding box around the truck
        color = (0, 255, 0)  # Green for entry
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 4)
        
        # Add label above the box
        category_str = result.category if result.category else "unknown"
        label = f"Truck {result.vehicle_id} - ENTRY [{category_str}]"
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        cv2.rectangle(
            debug_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        cv2.putText(
            debug_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Add timestamp at top of image
        timestamp_str = result.frame_time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(
            debug_frame,
            f"ENTRY: {timestamp_str}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )
        
        # Save the full debug image
        frame_filename = (
            f"entry_{result.frame_time.strftime('%H%M%S')}.jpg"
        )
        cv2.imwrite(f"{truck_folder}/{frame_filename}", debug_frame)
        
        # Build filenames for metadata
        crop_basename = f"entry_{result.frame_time.strftime('%H%M%S')}_crop.jpg"
        mask_basename = f"entry_{result.frame_time.strftime('%H%M%S')}_mask.png" if self.save_masks else None
        
        # Build mask info if available
        mask_info = None
        if self.save_masks and result.mask is not None:
            # Calculate mask statistics
            mask_pixel_count = int(np.sum(result.mask > 0))
            crop_area = crop_result.image.shape[0] * crop_result.image.shape[1] if crop_result.image is not None else 1
            coverage = mask_pixel_count / crop_area if crop_area > 0 else 0.0
            
            mask_info = MaskInfo(
                iou_score=0.9,  # SAM2 doesn't expose IoU directly, using default high value
                pixel_count=mask_pixel_count,
                coverage_ratio=min(coverage, 1.0)
            )
        
        # Map crop category to schema enum
        crop_category_map = {
            "full_truck": CropCategory.FULL_TRUCK,
            "small_vehicle": CropCategory.SMALL_VEHICLE,
            "cab_only": CropCategory.CAB_ONLY,
            "flatbed_extended": CropCategory.FULL_TRUCK,  # Extended flatbeds are full captures
        }
        crop_cat = crop_category_map.get(result.category, CropCategory.FULL_TRUCK)
        
        # Create structured metadata
        truck_id_str = f"Truck{result.vehicle_id:03d}"
        unique_truck_id_str = f"{self.settings.job_id}_{truck_id_str}" if self.settings.job_id else None
        
        metadata = TruckMetadata(
            truck_id=truck_id_str,
            unique_truck_id=unique_truck_id_str,
            job_id=self.settings.job_id,
            vehicle_id=result.vehicle_id,
            capture=CaptureInfo(
                timestamp=result.frame_time,
                video_source=self.current_video_source or "unknown",
                video_timestamp=result.frame_time,  # Already extracted from video
                frame_number=self.current_frame_number
            ),
            files=FileInfo(
                frame=frame_filename,
                crop=crop_basename,
                mask=mask_basename
            ),
            detection=DetectionInfo(
                bbox=tuple(result.bbox.tolist()),
                confidence=0.8,  # ByteTrack doesn't preserve original confidence
                class_id=7,  # COCO truck class
                tracker_id=result.tracker_id
            ),
            crop=CropInfo(
                dimensions=(crop_result.image.shape[1], crop_result.image.shape[0]) if crop_result.image is not None else (0, 0),
                aspect_ratio=result.aspect_ratio,
                category=crop_cat,
                padding_applied=20
            ),
            mask=mask_info,
            custom={
                "aspect_ratio": result.aspect_ratio,
                "category": result.category
            }
        )
        
        # Write metadata to JSON file
        try:
            self.metadata_writer.write(metadata, truck_folder)
            logger.debug(f"Truck {result.vehicle_id}: Saved metadata.json")
        except Exception as e:
            logger.error(f"Truck {result.vehicle_id}: Failed to write metadata: {e}")
        
        # Log the entry
        self.vehicle_log.append({
            "vehicle_id": result.vehicle_id,
            "tracker_id": result.tracker_id,
            "entry_time": result.frame_time.strftime('%Y-%m-%d %H:%M:%S'),
            "category": result.category
        })
        
        # Write to database if enabled and job_id is set
        if self.enable_database and self.settings.job_id:
            self._write_truck_to_database(
                truck_id=f"Truck{result.vehicle_id:03d}",
                unique_truck_id=f"{self.settings.job_id}_Truck{result.vehicle_id:03d}",
                job_id=self.settings.job_id,
                vehicle_id="truck",  # YOLO class name
                timestamp=result.frame_time,
                confidence=0.8,
                bbox=result.bbox,
                body_type=None,  # Will be filled by classification pipeline
                axle_type=None,
                small_vehicle_type=None,
                crop_path=f"{truck_folder}/{crop_basename}" if crop_result.image is not None else None,
                full_frame_path=f"{truck_folder}/{frame_filename}",
                metadata_json=json.dumps({
                    "aspect_ratio": float(result.aspect_ratio),  # Convert numpy types
                    "category": result.category,
                    "tracker_id": int(result.tracker_id) if result.tracker_id is not None else None
                })
            )
    
    def _create_labels(self, detections: sv.Detections) -> List[str]:
        """Create labels for visualization."""
        labels = []
        if detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is not None:
                    state = self.vehicle_states[tracker_id]
                    if tracker_id in self.tracker_to_vehicle_id:
                        vehicle_id = self.tracker_to_vehicle_id[tracker_id]
                        status = "CROSSED" if state["crossed_entry"] else ""
                        labels.append(f"Truck {vehicle_id} {status}")
                    else:
                        labels.append(f"ID:{tracker_id}")
                else:
                    labels.append("")
        return labels
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: List[str]
    ) -> np.ndarray:
        """Annotate frame with detections, entry line, and stats."""
        annotated_frame = frame.copy()
        
        # Draw entry line (green)
        cv2.line(
            annotated_frame,
            tuple(self.entry_line_start.astype(int)),
            tuple(self.entry_line_end.astype(int)),
            (0, 255, 0),
            3
        )
        
        # Draw bounding boxes
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        
        # Draw labels
        if labels:
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
        
        # Add stats overlay
        entered_count = sum(
            1 for s in self.vehicle_states.values() if s["crossed_entry"]
        )
        cv2.putText(
            annotated_frame,
            f"Trucks Entered: {entered_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated_frame
    
    def save_log(self, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Save vehicle log to CSV file.
        
        Args:
            filepath: Output file path. Uses settings if None.
        
        Returns:
            Vehicle log as list of dictionaries.
        """
        if filepath is None:
            filepath = self.settings.output_csv
        
        if self.vehicle_log:
            df = pd.DataFrame(self.vehicle_log)
            df = df.drop_duplicates(subset=['vehicle_id'], keep='first')
            df = df.sort_values("vehicle_id")
            df.to_csv(filepath, index=False)
            logger.info(f"Vehicle log saved to: {filepath}")
            logger.info(f"Total trucks entered: {len(df)}")
        else:
            logger.info("No vehicles were logged.")
        
        return self.vehicle_log
    
    def _write_truck_to_database(
        self,
        truck_id: str,
        unique_truck_id: str,
        job_id: str,
        vehicle_id: str,
        timestamp: datetime,
        confidence: float,
        bbox: np.ndarray,
        body_type: Optional[str],
        axle_type: Optional[str],
        small_vehicle_type: Optional[str],
        crop_path: Optional[str],
        full_frame_path: Optional[str],
        metadata_json: Optional[str]
    ) -> None:
        """
        Write truck detection to database.
        
        Args:
            truck_id: Local truck ID (Truck001, etc.)
            unique_truck_id: Global unique ID (job_id_Truck001)
            job_id: Parent job ID
            vehicle_id: YOLO class name
            timestamp: Detection timestamp
            confidence: Detection confidence
            bbox: Bounding box array [x1, y1, x2, y2]
            body_type: Body type classification
            axle_type: Axle type classification
            small_vehicle_type: Small vehicle type
            crop_path: Path to crop image
            full_frame_path: Path to full frame
            metadata_json: Additional metadata as JSON string
        """
        try:
            with get_session_context() as session:
                crud.create_truck(
                    session=session,
                    truck_id=truck_id,
                    unique_truck_id=unique_truck_id,
                    job_id=job_id,
                    vehicle_id=vehicle_id,
                    timestamp=timestamp,
                    confidence=confidence,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    body_type=body_type,
                    axle_type=axle_type,
                    small_vehicle_type=small_vehicle_type,
                    crop_path=crop_path,
                    full_frame_path=full_frame_path,
                    metadata_json=metadata_json
                )
                logger.debug(f"Wrote truck {unique_truck_id} to database")
        except Exception as e:
            logger.error(f"Failed to write truck {unique_truck_id} to database: {e}")
    
    def reset(self) -> None:
        """Reset tracker state for reprocessing."""
        self.vehicle_states.clear()
        self.vehicle_log.clear()
        self.vehicle_counter = 0
        self.tracker_to_vehicle_id.clear()
        self.aspect_filter.reset_counts()
        self.bbox_merger.reset_stats()
        self.delayed_capture.reset()
        self.current_frame_number = 0
        self.byte_tracker.reset()
        if self.mask_generator is not None:
            self.mask_generator.reset_stats()
        self.current_frame_masks.clear()
        logger.info("Tracker state reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics.
        """
        mask_stats = (
            self.mask_generator.get_stats() 
            if self.mask_generator is not None 
            else {"masks_generated": 0, "masks_failed": 0}
        )
        return {
            "total_vehicles_tracked": len(self.vehicle_states),
            "vehicles_crossed_entry": sum(
                1 for s in self.vehicle_states.values() if s["crossed_entry"]
            ),
            "vehicles_logged": len(self.vehicle_log),
            "aspect_filter": self.aspect_filter.get_stats(),
            "bbox_merger": self.bbox_merger.get_stats(),
            "delayed_capture": self.delayed_capture.get_stats(),
            "mask_generator": mask_stats
        }
