"""
Vehicle Tracking System

A Python-based vehicle tracking system using YOLO object detection 
and ByteTrack for real-time vehicle tracking and counting.
"""

from src.tracker.core.vehicle_tracker import VehicleTracker
from src.tracker.detection.aspect_filter import AspectRatioFilter
from src.tracker.utils.timestamp_reader import VideoTimestampReader

__version__ = "1.0.0"
__all__ = ["VehicleTracker", "AspectRatioFilter", "VideoTimestampReader"]
