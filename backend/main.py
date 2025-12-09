"""
Main entry point for vehicle tracking system.

Processes video and logs vehicle entry/exit times to CSV.

Usage:
    python main.py                    # Use settings from config/environment
    python main.py --video path.mp4   # Override video source
    python main.py --headless         # Run without display
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from new package structure
from src.tracker.config import Settings, get_settings
from src.tracker.core import VehicleTracker
from src.tracker.utils import setup_logging, get_logger
from src.tracker.classification.batch_processor import run_batch_classification
from src.tracker.analytics import get_cost_tracker
from src.tracker.jobs import JobManager, JobStatus
from src.tracker.data.database import create_db_and_tables, get_session_context
from src.tracker.data import crud
from tools.generate_summary import generate_summary
from tools.generate_cost_report import generate_cost_report, print_cost_summary

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vehicle Zone Entry/Exit Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Video source (file path or RTSP URL)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without video display"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        help="Process every Nth frame"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["models/yolov8n.pt", "models/yolov8s.pt", "models/yolov8m.pt", "models/yolov8x.pt"],
        help="YOLO model to use"
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Enable SAM2 segmentation mask generation (slower, but saves masks)"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Use existing job ID (or pass 'auto' to create new one)"
    )
    return parser.parse_args()


def get_video_source(args: argparse.Namespace, settings: Settings) -> str:
    """Determine video source from args or settings."""
    if args.video:
        return args.video
    
    if settings.video_source:
        return settings.video_source
    
    logger.error("No video source specified")
    sys.exit(1)


def main() -> None:
    """Main entry point for the vehicle tracker."""
    args = parse_args()
    
    # Set up logging first
    setup_logging()
    
    # Initialize database
    try:
        create_db_and_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")
    
    print("=" * 60)
    print("Vehicle Zone Entry/Exit Tracker")
    print("=" * 60)
    
    # Load settings
    settings = get_settings()
    
    # Apply command line overrides
    video_source = get_video_source(args, settings)
    
    # Job management (optional)
    job_manager = None
    job = None
    if args.job_id:
        job_manager = JobManager()
        
        # Check if we should use existing job ID or create new one
        if args.job_id == "auto":
            # Auto-create new job ID
            job = job_manager.create_job(video_source)
            logger.info(f"Created job {job.id} for video processing")
            
            # Create database entry for this job
            try:
                with get_session_context() as session:
                    crud.create_job(
                        session=session,
                        job_id=job.id,
                        video_name=Path(video_source).name,
                        video_path=video_source,
                        status="processing"
                    )
                    logger.info(f"Created database entry for job {job.id}")
            except Exception as e:
                logger.error(f"Failed to create database entry for job: {e}")
        else:
            # Use provided job ID
            job_id = args.job_id
            logger.info(f"Using existing job ID: {job_id}")
            
            # Get existing job from database
            try:
                with get_session_context() as session:
                    db_job = crud.get_job(session, job_id)
                    if db_job:
                        # Update status to processing
                        crud.update_job_status(
                            session=session,
                            job_id=job_id,
                            status="processing"
                        )
                        logger.info(f"Updated job {job_id} status to processing")
                        
                        # Create JobManager job object from database job
                        from src.tracker.jobs import Job
                        job = Job(
                            id=job_id,
                            video_path=video_source,
                            status=JobStatus.PROCESSING,
                            output_dir=Path("data/jobs") / job_id
                        )
                    else:
                        logger.error(f"Job {job_id} not found in database")
                        sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to load job {job_id}: {e}")
                sys.exit(1)
        
        settings.job_id = job.id
    
    if args.headless:
        settings.show_video = False
    if args.max_frames:
        settings.max_frames = args.max_frames
    if args.frame_skip:
        settings.frame_skip = args.frame_skip
    if args.model:
        settings.yolo_model = args.model
    
    # Masks OFF by default, enable with --mask flag
    settings.save_segmentation_masks = args.mask
    if args.mask:
        logger.info("SAM2 mask generation enabled (--mask flag)")
    
    # Initialize tracker
    tracker = VehicleTracker(settings)
    
    # Open video source
    logger.info(f"Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    # Set video source for metadata tracking
    tracker.set_video_source(video_source)
    
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
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
    
    logger.info(f"Video properties: {width}x{height} @ {fps:.1f} FPS")
    if total_frames > 0:
        logger.info(f"Total frames: {total_frames}")
    
    if settings.max_frames:
        logger.info(f"Processing first {settings.max_frames} frames only")
    
    print("\nPress Ctrl+C to quit")
    print("-" * 60)
    
    # Update job status to processing and set total frames
    if job_manager and job:
        job_manager.update_status(job.id, JobStatus.PROCESSING)
        # Update total frames in database
        with get_session_context() as session:
            effective_total = settings.max_frames if settings.max_frames else total_frames
            crud.update_job_progress(
                session=session,
                job_id=job.id,
                total_frames=effective_total if effective_total > 0 else 0,
                current_stage="loading",
                progress_percent=0.0
            )
    
    frame_count = 0
    start_time = datetime.now()
    wall_clock_start = time.perf_counter()
    last_progress_update = time.perf_counter()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\nEnd of video stream.")
                break
            
            frame_count += 1
            
            # Check max frames limit
            if settings.max_frames and frame_count > settings.max_frames:
                logger.info(f"Reached max frames limit ({settings.max_frames})")
                break
            
            # Skip frames if configured
            if frame_count % settings.frame_skip != 0:
                continue
            
            # Extract timestamp from video overlay (OCR)
            video_time = tracker.timestamp_reader.extract_timestamp(
                frame, frame_count
            )
            if video_time:
                frame_time = video_time
            else:
                # Fallback to calculated time if OCR fails
                frame_time = start_time + timedelta(seconds=frame_count / fps)
            
            # Process frame
            annotated_frame = tracker.process_frame(frame, frame_time)
            
            # Update progress every 50 frames or every 2 seconds
            if job_manager and job:
                current_time = time.perf_counter()
                if frame_count % 50 == 0 or (current_time - last_progress_update) >= 2.0:
                    last_progress_update = current_time
                    
                    # Calculate stage progress (0-100% for detecting stage)
                    effective_total = settings.max_frames if settings.max_frames else total_frames
                    if effective_total > 0:
                        stage_progress = (frame_count / effective_total) * 100
                        
                        # Update database (no ETA calculations)
                        try:
                            with get_session_context() as session:
                                crud.update_job_progress(
                                    session=session,
                                    job_id=job.id,
                                    current_stage="detecting",
                                    progress_percent=min(stage_progress, 100.0),
                                    current_frame=frame_count,
                                    eta_seconds=None,
                                    stage_eta_seconds=None
                                )
                        except Exception as e:
                            logger.warning(f"Failed to update progress: {e}")
            
            # Display frame (only if enabled and GUI available)
            if settings.show_video:
                try:
                    # Resize for display if needed
                    display_frame = annotated_frame
                    if width > 1280:
                        scale = 1280 / width
                        display_frame = cv2.resize(
                            annotated_frame, None, fx=scale, fy=scale
                        )
                    
                    cv2.imshow("Vehicle Tracker", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                except cv2.error:
                    logger.warning(
                        "GUI not available, running in headless mode..."
                    )
                    settings.show_video = False
            
            # Progress update
            if frame_count % 100 == 0:
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"Progress: {progress:.1f}% "
                        f"({frame_count}/{total_frames} frames)"
                    )
                else:
                    print(f"Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    finally:
        # Finalize any pending captures before cleanup
        tracker.finalize_all_pending(frame_width=width)
        
        # Cleanup
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # GUI not available
        
        # Save final log
        
        # Print elapsed time
        wall_clock_end = time.perf_counter()
        elapsed_seconds = wall_clock_end - wall_clock_start
        elapsed_minutes = elapsed_seconds / 60
        print(f"\n--- Timing ---")
        print(f"Total runtime: {elapsed_seconds:.1f} seconds ({elapsed_minutes:.2f} minutes)")
        print(f"Frames processed: {frame_count}")
        if frame_count > 0:
            fps_actual = frame_count / elapsed_seconds
            print(f"Average processing speed: {fps_actual:.1f} frames/second")
        
        # Run classification on captured trucks
        print("=" * 60)
        print("STEP 2: BODY TYPE CLASSIFICATION (Grok VLM)")
        print("=" * 60)
        
        # Update progress: starting classification stage
        if job_manager and job:
            with get_session_context() as session:
                crud.update_job_progress(
                    session=session,
                    job_id=job.id,
                    current_stage="classifying",
                    progress_percent=0.0,  # Start of classification stage
                    stage_eta_seconds=None
                )
        
        try:
            # Use job-based or legacy paths
            output_paths = settings.get_output_paths()
            debug_dir = output_paths["debug_images"]
            
            # Define progress callback for classification stage
            def classification_progress_callback(stage: str, progress: float, message: str):
                """Update job progress during classification."""
                if job_manager and job:
                    try:
                        with get_session_context() as session:
                            crud.update_job_progress(
                                session=session,
                                job_id=job.id,
                                current_stage=stage,
                                progress_percent=progress,
                                stage_eta_seconds=None  # Can't estimate VLM API time accurately
                            )
                        logger.debug(f"Progress update: {progress:.1f}% - {message}")
                    except Exception as e:
                        logger.warning(f"Failed to update progress: {e}")
            
            classification_stats = run_batch_classification(
                base_dir=debug_dir,
                dry_run=False,
                limit=None,
                confidence_threshold=0.6,
                reprocess=False,
                progress_callback=classification_progress_callback if job_manager else None
            )
            
            # Generate summary JSON/CSV
            
            # Update progress: starting analysis stage
            if job_manager and job:
                with get_session_context() as session:
                    crud.update_job_progress(
                        session=session,
                        job_id=job.id,
                        current_stage="analyzing",
                        progress_percent=0.0,  # Start of analyzing stage
                        stage_eta_seconds=None
                    )
            
            count, json_path, csv_path = generate_summary(
                debug_dir=output_paths["trucks"],
                output_dir=output_paths["reports"],
                output_name="truck_summary"
            )
            
            if count > 0:
                print(f"\nSummary files generated:")
                print(f"  JSON: {json_path}")
                print(f"  CSV:  {csv_path}")
            
            # Update progress: summary complete (50% of analyzing stage)
            if job_manager and job:
                with get_session_context() as session:
                    crud.update_job_progress(
                        session=session,
                        job_id=job.id,
                        current_stage="analyzing",
                        progress_percent=50.0,
                        stage_eta_seconds=None
                    )
            
            # Generate operating cost report
            print("\n" + "=" * 60)
            print("STEP 4: OPERATING COST REPORT")
            print("=" * 60)
            
            print_cost_summary()
            
            cost_json, cost_csv = generate_cost_report(output_paths["reports"])
            if cost_json and cost_csv:
                print(f"\nCost report files generated:")
                print(f"  JSON: {cost_json}")
                print(f"  CSV:  {cost_csv}")
            
            # Update progress: complete
            if job_manager and job:
                with get_session_context() as session:
                    crud.update_job_progress(
                        session=session,
                        job_id=job.id,
                        current_stage="analyzing",
                        progress_percent=100.0,
                        stage_eta_seconds=None
                    )
            
            # Update job status to completed
            if job_manager and job:
                job_manager.update_status(job.id, JobStatus.COMPLETED)
                logger.info(f"Job {job.id} completed successfully")
                
                # Update database job status
                try:
                    with get_session_context() as session:
                        crud.update_job_status(
                            session=session,
                            job_id=job.id,
                            status="completed",
                            truck_count=len(tracker.vehicle_log)
                        )
                        logger.info(f"Updated database entry for job {job.id}")
                except Exception as e:
                    logger.error(f"Failed to update database entry for job: {e}")
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            print(f"\nClassification error: {e}")
            print("You can run classification manually with:")
            print("  python -m src.tracker.classification.batch_processor")
            
            # Update job status to failed
            if job_manager and job:
                job_manager.update_status(job.id, JobStatus.FAILED, error=str(e))
                logger.error(f"Job {job.id} failed: {e}")
                
                # Update database job status
                try:
                    with get_session_context() as session:
                        crud.update_job_status(
                            session=session,
                            job_id=job.id,
                            status="failed",
                            error=str(e)
                        )
                        logger.info(f"Updated database entry for job {job.id} to failed")
                except Exception as db_error:
                    logger.error(f"Failed to update database entry for job: {db_error}")


if __name__ == "__main__":
    main()
