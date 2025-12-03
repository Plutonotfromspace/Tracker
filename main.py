"""
Main entry point for vehicle tracking system.
Processes video and logs vehicle entry/exit times to CSV.
"""

import cv2
import sys
from datetime import datetime, timedelta

import config
from tracker import VehicleTracker


def main():
    print("=" * 60)
    print("Vehicle Zone Entry/Exit Tracker")
    print("=" * 60)
    
    # Initialize tracker
    tracker = VehicleTracker()
    
    # Open video source
    print(f"\nOpening video source: {config.VIDEO_SOURCE}")
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {config.VIDEO_SOURCE}")
        print("\nPlease check:")
        print("  1. The file path is correct")
        print("  2. The video file exists")
        print("  3. For RTSP streams, ensure the URL is accessible")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps:.1f} FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    
    print(f"\nZone polygon: {config.ZONE_POLYGON}")
    print(f"Output CSV: {config.OUTPUT_CSV}")
    max_frames = getattr(config, 'MAX_FRAMES', None)
    if max_frames:
        print(f"Processing first {max_frames} frames only")
    print("\nPress Ctrl+C to quit")
    print("-" * 60)
    
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\nEnd of video stream.")
                break
            
            frame_count += 1
            
            # Check max frames limit
            max_frames = getattr(config, 'MAX_FRAMES', None)
            if max_frames and frame_count > max_frames:
                print(f"\nReached max frames limit ({max_frames})")
                break
            
            # Skip frames if configured
            if frame_count % config.FRAME_SKIP != 0:
                continue
            
            # Extract timestamp from video overlay (OCR)
            # This reads the timestamp burned into the video frame
            video_time = tracker.timestamp_reader.extract_timestamp(frame, frame_count)
            if video_time:
                frame_time = video_time
            else:
                # Fallback to calculated time if OCR fails
                frame_time = start_time + timedelta(seconds=frame_count / fps)
            
            # Process frame
            annotated_frame = tracker.process_frame(frame, frame_time)
            
            # Display frame (only if enabled and GUI available)
            if config.SHOW_VIDEO:
                try:
                    # Resize for display if needed
                    display_frame = annotated_frame
                    if width > 1280:
                        scale = 1280 / width
                        display_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
                    
                    cv2.imshow("Vehicle Tracker", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        tracker.save_log()
                except cv2.error:
                    print("Warning: GUI not available, running in headless mode...")
                    config.SHOW_VIDEO = False
            
            # Progress update
            if frame_count % 100 == 0:
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                else:
                    print(f"Processed {frame_count} frames...")
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    finally:
        # Cleanup
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # GUI not available
        
        # Save final log
        print("\n" + "=" * 60)
        print("Final Results")
        print("=" * 60)
        tracker.save_log()
        
        # Print aspect ratio filter summary
        tracker.aspect_filter.log_summary()
        
        # Print summary
        if tracker.vehicle_log:
            print("\n--- Vehicle Log Summary ---")
            for entry in sorted(tracker.vehicle_log, key=lambda x: x["vehicle_id"]):
                print(f"Truck {entry['vehicle_id']}: Entry={entry['entry_time']}")


if __name__ == "__main__":
    main()
