"""
Configuration for vehicle tracking system.
Adjust ZONE_POLYGON coordinates to match your camera view.
"""

# Video source - can be a file path, RTSP stream URL, or camera index (0, 1, etc.)
VIDEO_SOURCE = r"f:\Tracker\videos\World Trade Bridge Oct 16th from 3.22PM to 3.51PM.mp4"

# Detection zone polygon (x, y coordinates)
# These coordinates should match the yellow zone lines in your camera view
# Adjust these based on your actual camera frame dimensions
# Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4], ...]
ZONE_POLYGON = [[328, 544], [1074, 709], [1681, 776], [1675, 948], [899, 830], [250, 711]]

# Detection settings
CONFIDENCE_THRESHOLD = 0.15  # Lower threshold to detect trucks further away
IOU_THRESHOLD = 0.4          # IoU threshold for NMS

# Classes to detect (COCO dataset class IDs)
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
# Including car since semi-truck cabs often get detected as cars
VEHICLE_CLASSES = [2, 5, 7]

# Output CSV file
OUTPUT_CSV = "vehicle_log.csv"

# Display settings
SHOW_VIDEO = False          # Set to False for headless mode (no GUI)
FRAME_SKIP = 1              # Process every Nth frame (1 = all frames)
MAX_FRAMES = None           # Maximum frames to process (None for all frames)

# Model settings
YOLO_MODEL = "yolov8s.pt"   # Small model - better accuracy than nano
DEVICE = "cuda"             # Use "cuda" for GPU, "cpu" for CPU

# Line crossing detection
# Define entry line as [x1, y1, x2, y2]
# Entry line in middle of road where trucks are reliably detected
ENTRY_LINE = [800, 700, 600, 800]    # Middle of road - trucks detected before crossing

# OpenAI Vision API settings for trailer detection
OPENAI_API_KEY = "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"  # Set your OpenAI API key here or use environment variable
OPENAI_MODEL = "gpt-4.1-nano"  # Vision model for trailer detection (gpt-4.1-nano is faster and more reliable than o4-mini)
CROP_EXPANSION_PIXELS = 100  # How many pixels to expand crop each iteration
MAX_CROP_ITERATIONS = 10  # Maximum times to expand the crop

# API Pricing (per 1M tokens) - GPT-4.1-nano
# https://openai.com/api/pricing/
API_PRICE_INPUT_PER_1M = 0.10   # $0.10 per 1M input tokens
API_PRICE_OUTPUT_PER_1M = 0.40  # $0.40 per 1M output tokens
