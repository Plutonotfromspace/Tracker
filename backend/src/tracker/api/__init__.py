"""API routers for tracker application."""

from .jobs import router as jobs_router
from .trucks import router as trucks_router
from .auth import router as auth_router
from .upload import router as upload_router
from .export import router as export_router

__all__ = ["jobs_router", "trucks_router", "auth_router", "upload_router", "export_router"]
