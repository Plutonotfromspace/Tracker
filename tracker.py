"""
Vehicle tracking system with LINE CROSSING detection.
More reliable than zone-based detection for entry/exit logging.
"""

import cv2
import numpy as np
import pandas as pd
import os
import re
import shutil
import logging
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

import config

# Set up logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracker.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fix console encoding for Windows
import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class VideoTimestampReader:
    """Extracts timestamp from video frame overlay using OCR."""
    
    def __init__(self):
        if EASYOCR_AVAILABLE:
            logger.info("Initializing EasyOCR for timestamp extraction...")
            self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            logger.info("EasyOCR initialized successfully")
        else:
            logger.warning("EasyOCR not available. Install with: pip install easyocr")
            self.reader = None
        
        # Cache the last successful timestamp to avoid repeated OCR
        self.last_timestamp = None
        self.last_frame_num = -1
    
    def extract_timestamp(self, frame, frame_num=0):
        """
        Extract timestamp from the top-right corner of the frame.
        Expected format: "MM-DD-YYYY HH:MM:SS AM/PM Day"
        Example: "10-16-2025 03:23:21 PM Thu"
        """
        if self.reader is None:
            return None
        
        # Only run OCR every few frames to save processing time
        # But always run for the exact frame we need a timestamp for
        h, w = frame.shape[:2]
        
        # Crop top-right region where timestamp is located
        # Based on the image, timestamp is roughly in the right third, top 50px
        roi_x1 = int(w * 0.55)  # Start from 55% of width
        roi_y1 = 0
        roi_x2 = w
        roi_y2 = 80  # Top 80 pixels
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        try:
            # Run OCR on the region
            results = self.reader.readtext(roi, detail=0, paragraph=True)
            
            if not results:
                return self.last_timestamp
            
            # Join all text found
            text = ' '.join(results)
            
            # Parse the timestamp - format: "10-16-2025 03:23:21 PM Thu" or "10-16-2025 03.23.21 PM Thu"
            # Pattern: MM-DD-YYYY HH:MM:SS AM/PM (colons or periods)
            pattern = r'(\d{1,2})-(\d{1,2})-(\d{4})\s+(\d{1,2})[:\.](\d{2})[:\.](\d{2})\s*(AM|PM|am|pm)'
            match = re.search(pattern, text)
            
            if match:
                month = int(match.group(1))
                day = int(match.group(2))
                year = int(match.group(3))
                hour = int(match.group(4))
                minute = int(match.group(5))
                second = int(match.group(6))
                ampm = match.group(7).upper()
                
                # Convert to 24-hour format
                if ampm == 'PM' and hour != 12:
                    hour += 12
                elif ampm == 'AM' and hour == 12:
                    hour = 0
                
                timestamp = datetime(year, month, day, hour, minute, second)
                self.last_timestamp = timestamp
                self.last_frame_num = frame_num
                return timestamp
            else:
                # Try a simpler pattern without seconds (colons or periods)
                pattern2 = r'(\d{1,2})-(\d{1,2})-(\d{4})\s+(\d{1,2})[:\.](\d{2})\s*(AM|PM|am|pm)'
                match2 = re.search(pattern2, text)
                if match2:
                    month = int(match2.group(1))
                    day = int(match2.group(2))
                    year = int(match2.group(3))
                    hour = int(match2.group(4))
                    minute = int(match2.group(5))
                    ampm = match2.group(6).upper()
                    
                    if ampm == 'PM' and hour != 12:
                        hour += 12
                    elif ampm == 'AM' and hour == 12:
                        hour = 0
                    
                    timestamp = datetime(year, month, day, hour, minute, 0)
                    self.last_timestamp = timestamp
                    self.last_frame_num = frame_num
                    return timestamp
            
            return self.last_timestamp
            
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return self.last_timestamp


class AspectRatioFilter:
    """Simple aspect ratio-based filter for truck crops. Rejects narrow images that likely don't show the full trailer."""
    
    # Minimum aspect ratio (width/height) to accept a crop
    # Trucks with trailers are wide (~2.0+), cab-only is narrow (~1.0-1.5)
    MIN_ASPECT_RATIO = 2.0
    
    def __init__(self):
        self.accepted_count = 0
        self.rejected_count = 0
    
    def get_truck_crop(self, frame, bbox, truck_folder=None):
        """
        Get a crop of the truck. Returns (crop, accepted) tuple.
        If aspect ratio is too narrow, returns (None, False) - truck should be rejected.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        frame_h, frame_w = frame.shape[:2]
        padding = 20
        
        # Calculate crop bounds
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(frame_w, x2 + padding)
        crop_y2 = min(frame_h, y2 + padding)
        
        width = crop_x2 - crop_x1
        height = crop_y2 - crop_y1
        aspect_ratio = width / height if height > 0 else 0
        
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if aspect_ratio >= self.MIN_ASPECT_RATIO:
            self.accepted_count += 1
            logger.info(f"ACCEPTED: Aspect ratio {aspect_ratio:.2f} >= {self.MIN_ASPECT_RATIO}")
            return cropped, True, aspect_ratio
        else:
            self.rejected_count += 1
            logger.info(f"REJECTED: Aspect ratio {aspect_ratio:.2f} < {self.MIN_ASPECT_RATIO} - trailer not visible")
            return None, False, aspect_ratio
    
    def log_summary(self):
        """Log a summary of accepted/rejected counts."""
        total = self.accepted_count + self.rejected_count
        accept_rate = (self.accepted_count / total * 100) if total > 0 else 0
        logger.info("="*60)
        logger.info("ASPECT RATIO FILTER SUMMARY")
        logger.info("="*60)
        logger.info(f"Minimum aspect ratio: {self.MIN_ASPECT_RATIO}")
        logger.info(f"Total trucks detected: {total}")
        logger.info(f"Accepted (trailer visible): {self.accepted_count}")
        logger.info(f"Rejected (trailer cut off): {self.rejected_count}")
        logger.info(f"Acceptance rate: {accept_rate:.1f}%")
        logger.info("="*60)


class VehicleTracker:
    def __init__(self):
        # Load YOLO model
        logger.info(f"Loading YOLO model: {config.YOLO_MODEL}")
        self.model = YOLO(config.YOLO_MODEL)
        
        # Set device (CUDA/CPU)
        device = getattr(config, 'DEVICE', 'cuda')
        logger.info(f"Using device: {device}")
        self.device = device
        
        # Initialize ByteTrack tracker with higher thresholds for consistency
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,  # Keep tracks alive longer
            minimum_matching_threshold=0.8,
            frame_rate=8  # Match video FPS
        )
        
        # Entry line
        self.entry_line_start = np.array(config.ENTRY_LINE[:2])
        self.entry_line_end = np.array(config.ENTRY_LINE[2:])
        
        # Track vehicle states
        # {tracker_id: {"crossed_entry": bool, "entry_time": datetime, "last_pos": [x,y]}}
        self.vehicle_states = defaultdict(lambda: {
            "crossed_entry": False,
            "entry_time": None,
            "last_pos": None
        })
        
        # Vehicle log for CSV output
        self.vehicle_log = []
        
        # Annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        
        # Counter for unique vehicles that crossed entry
        self.vehicle_counter = 0
        self.tracker_to_vehicle_id = {}
        
        # Debug images directory
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)
        logger.info(f"Debug images will be saved to: {self.debug_dir}/")
        logger.info(f"Entry line: {config.ENTRY_LINE}")
        
        # Initialize aspect ratio filter (replaces AI-based trailer detection)
        self.aspect_filter = AspectRatioFilter()
        logger.info(f"Aspect ratio filter: min ratio = {self.aspect_filter.MIN_ASPECT_RATIO}")
        
        # Initialize timestamp reader for extracting time from video overlay
        self.timestamp_reader = VideoTimestampReader()
        self.current_video_time = None
        
    def get_vehicle_id(self, tracker_id):
        """Get or create a sequential vehicle ID for a tracker ID."""
        if tracker_id not in self.tracker_to_vehicle_id:
            self.vehicle_counter += 1
            self.tracker_to_vehicle_id[tracker_id] = self.vehicle_counter
        return self.tracker_to_vehicle_id[tracker_id]
    
    def _ccw(self, A, B, C):
        """Check if three points are in counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def _lines_intersect(self, A, B, C, D):
        """Check if line segment AB intersects with line segment CD."""
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)
    
    def _crossed_line(self, prev_pos, curr_pos, line_start, line_end):
        """Check if movement from prev_pos to curr_pos crosses the line."""
        if prev_pos is None:
            return False
        return self._lines_intersect(prev_pos, curr_pos, line_start, line_end)
    
    def _save_debug_image(self, frame, bbox, vehicle_id, event_type, frame_time):
        """
        Save a debug image with bounding box around the truck.
        Returns True if accepted (trailer visible), False if rejected.
        """
        # Check aspect ratio first - reject if trailer not visible
        cropped_truck, accepted, aspect_ratio = self.aspect_filter.get_truck_crop(frame, bbox)
        
        if not accepted:
            # Trailer not visible - reject this truck, don't save anything
            logger.info(f"Truck {vehicle_id}: REJECTED (aspect ratio {aspect_ratio:.2f} < {self.aspect_filter.MIN_ASPECT_RATIO})")
            return False
        
        # Accepted - create folder and save images
        debug_frame = frame.copy()
        
        # Create truck-specific folder
        truck_folder = f"{self.debug_dir}/Truck{vehicle_id:03d}"
        os.makedirs(truck_folder, exist_ok=True)
        
        # Save the cropped truck image
        crop_filename = f"{truck_folder}/{event_type.lower()}_{frame_time.strftime('%H%M%S')}_crop.jpg"
        cv2.imwrite(crop_filename, cropped_truck)
        logger.info(f"Truck {vehicle_id}: ACCEPTED (aspect ratio {aspect_ratio:.2f}) - saved to {truck_folder}")
        
        # Draw the entry line (green)
        cv2.line(debug_frame, 
                tuple(self.entry_line_start.astype(int)), 
                tuple(self.entry_line_end.astype(int)), 
                (0, 255, 0), 3)
        cv2.putText(debug_frame, "ENTRY", 
                   (int(self.entry_line_start[0]) + 10, int(self.entry_line_start[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Get bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Draw thick bounding box around the truck
        color = (0, 255, 0) if event_type == "ENTRY" else (0, 0, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 4)
        
        # Add label above the box
        label = f"Truck {vehicle_id} - {event_type}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(debug_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(debug_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Add timestamp at top of image
        timestamp_str = frame_time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(debug_frame, f"{event_type}: {timestamp_str}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Save the image
        filename = f"{truck_folder}/{event_type.lower()}_{frame_time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, debug_frame)
        
        return True  # Accepted
    
    def process_frame(self, frame, frame_time=None):
        """Process a single frame for vehicle detection and line crossing."""
        if frame_time is None:
            frame_time = datetime.now()
            
        # Run YOLO detection
        results = self.model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=config.VEHICLE_CLASSES,
            device=self.device,
            verbose=False
        )[0]
        
        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Apply tracking
        detections = self.byte_tracker.update_with_detections(detections)
        
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
                
                # Check for ENTRY line crossing
                if not state["crossed_entry"]:
                    if self._crossed_line(prev_pos, curr_pos, self.entry_line_start, self.entry_line_end):
                        state["crossed_entry"] = True
                        state["entry_time"] = frame_time
                        vehicle_id = self.get_vehicle_id(tracker_id)
                        logger.info(f"[ENTRY] Truck {vehicle_id} crossed entry line at {frame_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Only log if truck was accepted (trailer visible based on aspect ratio)
                        accepted = self._save_debug_image(frame, bbox, vehicle_id, "ENTRY", frame_time)
                        if accepted:
                            self.vehicle_log.append({
                                "vehicle_id": vehicle_id,
                                "tracker_id": tracker_id,
                                "entry_time": state["entry_time"].strftime('%Y-%m-%d %H:%M:%S')
                            })
                
                # Update last position
                state["last_pos"] = curr_pos
        
        # Create labels for visualization
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
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Draw entry line (green)
        cv2.line(annotated_frame, 
                tuple(self.entry_line_start.astype(int)), 
                tuple(self.entry_line_end.astype(int)), 
                (0, 255, 0), 3)
        
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        if labels:
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections, 
                labels=labels
            )
        
        # Add stats overlay
        entered_count = sum(1 for s in self.vehicle_states.values() if s["crossed_entry"])
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
    
    def save_log(self, filepath=None):
        """Save vehicle log to CSV file."""
        if filepath is None:
            filepath = config.OUTPUT_CSV
        
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
