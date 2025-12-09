"""
FastAPI router for truck endpoints.

Provides REST API for querying truck detections across all jobs.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session
from pydantic import BaseModel

from src.tracker.data.database import get_session
from src.tracker.data import crud
from src.tracker.data.models import Truck


router = APIRouter(prefix="/api/trucks", tags=["trucks"])


class TruckResponse(BaseModel):
    """Response model for truck data."""
    id: int
    truck_id: str
    unique_truck_id: str
    job_id: str
    vehicle_id: str
    timestamp: str
    confidence: float
    body_type: Optional[str]
    axle_type: Optional[str]
    small_vehicle_type: Optional[str]
    crop_path: Optional[str]
    full_frame_path: Optional[str]
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    
    class Config:
        from_attributes = True


@router.get("/", response_model=List[TruckResponse])
def get_trucks(
    body_type: Optional[str] = None,
    axle_type: Optional[str] = None,
    small_vehicle_type: Optional[str] = None,
    offset: int = 0,
    limit: int = Query(default=100, le=1000),
    session: Session = Depends(get_session)
):
    """
    Get all trucks with optional filtering.
    
    Args:
        body_type: Filter by body type (optional)
        axle_type: Filter by axle type (optional)
        small_vehicle_type: Filter by small vehicle type (optional)
        offset: Number of records to skip
        limit: Maximum number of records to return (max 1000)
        session: Database session (injected)
    
    Returns:
        List of trucks matching filters
    """
    trucks = crud.get_all_trucks(
        session,
        body_type=body_type,
        axle_type=axle_type,
        small_vehicle_type=small_vehicle_type,
        offset=offset,
        limit=limit
    )
    
    return [
        TruckResponse(
            id=truck.id,
            truck_id=truck.truck_id,
            unique_truck_id=truck.unique_truck_id,
            job_id=truck.job_id,
            vehicle_id=truck.vehicle_id,
            timestamp=truck.timestamp.isoformat(),
            confidence=truck.confidence,
            body_type=truck.body_type,
            axle_type=truck.axle_type,
            small_vehicle_type=truck.small_vehicle_type,
            crop_path=truck.crop_path,
            full_frame_path=truck.full_frame_path,
            bbox_x1=truck.bbox_x1,
            bbox_y1=truck.bbox_y1,
            bbox_x2=truck.bbox_x2,
            bbox_y2=truck.bbox_y2
        )
        for truck in trucks
    ]


@router.get("/job/{job_id}", response_model=List[TruckResponse])
def get_trucks_for_job(
    job_id: str,
    offset: int = 0,
    limit: int = Query(default=1000, le=1000),
    session: Session = Depends(get_session)
):
    """
    Get all trucks for a specific job.
    
    Args:
        job_id: Job identifier
        offset: Number of records to skip
        limit: Maximum number of records to return (max 1000)
        session: Database session (injected)
    
    Returns:
        List of trucks for the job
    """
    trucks = crud.get_trucks_for_job(session, job_id, offset=offset, limit=limit)
    
    return [
        TruckResponse(
            id=truck.id,
            truck_id=truck.truck_id,
            unique_truck_id=truck.unique_truck_id,
            job_id=truck.job_id,
            vehicle_id=truck.vehicle_id,
            timestamp=truck.timestamp.isoformat(),
            confidence=truck.confidence,
            body_type=truck.body_type,
            axle_type=truck.axle_type,
            small_vehicle_type=truck.small_vehicle_type,
            crop_path=truck.crop_path,
            full_frame_path=truck.full_frame_path,
            bbox_x1=truck.bbox_x1,
            bbox_y1=truck.bbox_y1,
            bbox_x2=truck.bbox_x2,
            bbox_y2=truck.bbox_y2
        )
        for truck in trucks
    ]


@router.get("/{unique_truck_id}", response_model=TruckResponse)
def get_truck(
    unique_truck_id: str,
    session: Session = Depends(get_session)
):
    """
    Get a specific truck by its globally unique ID.
    
    Args:
        unique_truck_id: Globally unique truck identifier
        session: Database session (injected)
    
    Returns:
        Truck details
    
    Raises:
        HTTPException: If truck not found
    """
    truck = crud.get_truck_by_unique_id(session, unique_truck_id)
    if not truck:
        raise HTTPException(
            status_code=404,
            detail=f"Truck {unique_truck_id} not found"
        )
    
    return TruckResponse(
        id=truck.id,
        truck_id=truck.truck_id,
        unique_truck_id=truck.unique_truck_id,
        job_id=truck.job_id,
        vehicle_id=truck.vehicle_id,
        timestamp=truck.timestamp.isoformat(),
        confidence=truck.confidence,
        body_type=truck.body_type,
        axle_type=truck.axle_type,
        small_vehicle_type=truck.small_vehicle_type,
        crop_path=truck.crop_path,
        full_frame_path=truck.full_frame_path,
        bbox_x1=truck.bbox_x1,
        bbox_y1=truck.bbox_y1,
        bbox_x2=truck.bbox_x2,
        bbox_y2=truck.bbox_y2
    )
