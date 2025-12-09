"""
FastAPI router for data export endpoints.

Provides REST API for exporting aggregated data across all jobs.
"""

import csv
import json
from io import StringIO
from typing import List
from fastapi import APIRouter, Depends, Query, Response
from sqlmodel import Session, select, desc

from src.tracker.data.database import get_session
from src.tracker.data.models import Truck, Job


router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/master-data")
def export_master_data(
    format: str = Query(default="csv", regex="^(csv|json)$"),
    session: Session = Depends(get_session)
):
    """
    Export all truck data from all completed jobs.
    
    Provides comprehensive dataset including:
    - Job metadata (job_id, video_name)
    - Truck identification (truck_id, unique_truck_id)
    - Classification data (body_type, axle_type, small_vehicle_type)
    - Confidence scores for all classifications
    - Frame data (first_seen, last_seen)
    
    Args:
        format: Export format ('csv' or 'json')
        session: Database session (injected)
    
    Returns:
        CSV or JSON file download
    """
    # Query all trucks from completed jobs
    statement = (
        select(Truck, Job)
        .join(Job, Truck.job_id == Job.id)
        .where(Job.status == "completed")
        .order_by(desc(Job.start_time), Truck.first_seen_frame)
    )
    
    results = session.exec(statement).all()
    
    # Prepare data
    export_data = []
    for truck, job in results:
        export_data.append({
            "job_id": job.id,
            "video_name": job.video_name,
            "job_date": job.start_time.isoformat(),
            "truck_id": truck.truck_id,
            "unique_truck_id": truck.unique_truck_id,
            "body_type": truck.body_type,
            "body_type_confidence": truck.body_type_confidence,
            "axle_type": truck.axle_type,
            "axle_type_confidence": truck.axle_type_confidence,
            "small_vehicle_type": truck.small_vehicle_type,
            "small_vehicle_confidence": truck.small_vehicle_confidence,
            "first_seen_frame": truck.first_seen_frame,
            "last_seen_frame": truck.last_seen_frame,
            "crop_path": truck.crop_path,
        })
    
    if format == "csv":
        # Generate CSV
        output = StringIO()
        if export_data:
            fieldnames = list(export_data[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_data)
        
        csv_content = output.getvalue()
        output.close()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=master-data.csv"
            }
        )
    else:  # json
        return Response(
            content=json.dumps(export_data, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=master-data.json"
            }
        )
