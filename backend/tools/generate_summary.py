#!/usr/bin/env python3
"""
Generate summary JSON and CSV files from truck metadata.

Creates a central summary with:
- Truck ID
- Timestamp
- Files (with full clickable paths)
- Body type (if available)
- Axle type (if available)
- Small vehicle type (if available)

Usage:
    python tools/generate_summary.py [--dir DEBUG_DIR] [--output OUTPUT_NAME]
    
Examples:
    python tools/generate_summary.py
    python tools/generate_summary.py --dir debug_images --output truck_summary
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_metadata(truck_folder: Path) -> Optional[Dict[str, Any]]:
    """Load metadata.json from a truck folder."""
    metadata_file = truck_folder / "metadata.json"
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {metadata_file}: {e}")
        return None


def build_summary_record(truck_folder: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build a summary record from metadata, only including non-null fields."""
    record = {}
    
    # Truck ID (always included)
    record["truck_id"] = metadata.get("truck_id", truck_folder.name)
    
    # Timestamp
    capture = metadata.get("capture", {})
    timestamp = capture.get("timestamp") or capture.get("video_timestamp")
    if timestamp:
        record["timestamp"] = timestamp
    
    # Files with full paths
    files_info = metadata.get("files", {})
    files = {}
    
    if files_info.get("frame"):
        files["frame"] = str(truck_folder / files_info["frame"])
    if files_info.get("crop"):
        files["crop"] = str(truck_folder / files_info["crop"])
    if files_info.get("mask"):
        files["mask"] = str(truck_folder / files_info["mask"])
    
    if files:
        record["files"] = files
    
    # Classification fields (only if not null/unknown)
    classification = metadata.get("classification", {})
    
    body_type = classification.get("body_type")
    if body_type and body_type.lower() != "unknown":
        record["body_type"] = body_type
    
    axle_type = classification.get("axle_type")
    if axle_type and axle_type.lower() != "unknown":
        record["axle_type"] = axle_type
    
    small_vehicle_type = classification.get("small_vehicle_type")
    if small_vehicle_type and small_vehicle_type.lower() not in ("unknown", "null", "none"):
        record["small_vehicle_type"] = small_vehicle_type
    
    return record


def flatten_for_csv(record: Dict[str, Any]) -> Dict[str, str]:
    """Flatten a record for CSV output."""
    # Use unique_truck_id if available, fall back to truck_id
    truck_id = record.get("unique_truck_id") or record.get("truck_id", "")
    
    flat = {
        "truck_id": truck_id,
        "job_id": record.get("job_id", ""),
        "timestamp": record.get("timestamp", ""),
    }
    
    # Flatten files
    files = record.get("files", {})
    flat["frame_path"] = files.get("frame", "")
    flat["crop_path"] = files.get("crop", "")
    flat["mask_path"] = files.get("mask", "")
    
    # Classification fields
    flat["body_type"] = record.get("body_type", "")
    flat["axle_type"] = record.get("axle_type", "")
    flat["small_vehicle_type"] = record.get("small_vehicle_type", "")
    
    return flat


def generate_summary(
    debug_dir: Path,
    output_dir: Optional[Path] = None,
    output_name: str = "truck_summary"
) -> tuple[int, Optional[Path], Optional[Path]]:
    """
    Generate summary JSON and CSV files.
    
    Args:
        debug_dir: Directory containing Truck### folders
        output_dir: Optional output directory for reports (defaults to debug_dir.parent/reports)
        output_name: Base name for output files (without extension)
    
    Returns:
        Tuple of (record_count, json_path, csv_path)
    """
    # Find all truck folders
    truck_folders = sorted([
        d for d in debug_dir.iterdir()
        if d.is_dir() and d.name.startswith("Truck")
    ])
    
    if not truck_folders:
        return 0, None, None
    
    # Build records
    records = []
    for folder in truck_folders:
        metadata = load_metadata(folder)
        if metadata:
            record = build_summary_record(folder, metadata)
            records.append(record)
        else:
            print(f"Skipping {folder.name}: no metadata.json")
    
    if not records:
        print("No valid metadata found")
        return 0, None, None
    
    # Write JSON to reports directory
    reports_dir = output_dir if output_dir else (debug_dir.parent / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{output_name}.json"
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    # Write CSV
    csv_path = reports_dir / f"{output_name}.csv"
    csv_fields = [
        "truck_id", "job_id", "timestamp", 
        "frame_path", "crop_path", "mask_path",
        "body_type", "axle_type", "small_vehicle_type"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_for_csv(record))
    
    # Print summary stats
    body_type_count = sum(1 for r in records if "body_type" in r)
    axle_type_count = sum(1 for r in records if "axle_type" in r)
    small_vehicle_count = sum(1 for r in records if "small_vehicle_type" in r)
    
    return len(records), json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary JSON/CSV from truck metadata"
    )
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=Path("outputs/debug_images"),
        help="Directory containing Truck### folders (default: outputs/debug_images)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="truck_summary",
        help="Base name for output files (default: truck_summary)"
    )
    
    args = parser.parse_args()
    
    # Resolve to absolute path
    debug_dir = args.dir.resolve()
    
    if not debug_dir.exists():
        print(f"Error: Directory not found: {debug_dir}")
        return 1
    
    count, json_path, csv_path = generate_summary(debug_dir, args.output)
    
    if count == 0:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
