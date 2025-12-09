"""
FastAPI server for the Tracker application.

Provides REST API endpoints for jobs, trucks, and authentication.

Usage:
    uvicorn server:app --reload
    # or
    python server.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path

from src.tracker.api import jobs_router, trucks_router, auth_router, upload_router, export_router
from src.tracker.data.database import create_db_and_tables

# Create FastAPI app
app = FastAPI(
    title="Truck Tracker API",
    description="REST API for truck detection and tracking system",
    version="1.0.0"
)

# Configure CORS - use environment variable for origins
# Default to localhost for development, but allow override for production
cors_origins = [
    origin.strip() 
    for origin in os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(jobs_router)
app.include_router(trucks_router)
app.include_router(upload_router)
app.include_router(export_router)

# Mount static files for images (crops and full frames)
# This allows serving images at /data/jobs/...
data_path = Path("data")
if data_path.exists():
    app.mount("/data", StaticFiles(directory="data"), name="data")


@app.on_event("startup")
def on_startup():
    """Initialize database on startup."""
    create_db_and_tables()
    print("✓ Database initialized")


@app.get("/")
def read_root():
    """Root endpoint - API information."""
    return {
        "name": "Truck Tracker API",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/api/auth/*",
            "jobs": "/api/jobs/*",
            "trucks": "/api/trucks/*",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
