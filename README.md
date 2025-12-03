# Vehicle Tracker

A Python-based vehicle tracking system using YOLO object detection and OpenCV for real-time vehicle tracking and counting.

## Features

- Real-time vehicle detection and tracking
- Configurable detection zones
- Vehicle counting and logging
- Support for multiple YOLO models (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8x)
- Video processing capabilities

## Requirements

See `requirements.txt` for Python dependencies.

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download YOLO model files (*.pt) and place them in the project root

## Usage

### Main Tracker
```
python main.py
```

### Calibrate Detection Zone
```
python calibrate_zone.py
```

### Compile Crops
```
python compile_crops.py
```

## Project Structure

- `main.py` - Main entry point for the tracker
- `tracker.py` - Core tracking logic
- `config.py` - Configuration settings
- `calibrate_zone.py` - Tool for calibrating detection zones
- `compile_crops.py` - Tool for compiling cropped images
- `videos/` - Input video directory (not tracked)
- `debug_images/` - Debug output directory (not tracked)
- `compiled_canvases/` - Compiled output directory (not tracked)

## Output

- `vehicle_log.csv` - Log of detected vehicles
- `tracker.log` - Application logs
