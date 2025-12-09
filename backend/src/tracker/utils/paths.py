"""
Path resolution utilities for the tracker application.

Provides consistent path resolution that works in both development and Docker/AWS environments.
"""

import os
from pathlib import Path
from typing import Optional


def get_backend_dir(fallback_file: Optional[str] = None) -> Path:
    """
    Get the backend root directory.
    
    Priority order:
    1. BACKEND_DIR environment variable
    2. Current working directory (if it looks like backend root)
    3. Fallback to file-relative path calculation
    
    Args:
        fallback_file: Path to a file (usually __file__) to use for fallback calculation.
                      Should be provided when called from within the backend directory.
    
    Returns:
        Path to the backend root directory
        
    Examples:
        >>> # In Docker/AWS, uses working directory or BACKEND_DIR
        >>> backend_dir = get_backend_dir()
        
        >>> # With fallback for development
        >>> backend_dir = get_backend_dir(__file__)
    """
    # First try environment variable
    if "BACKEND_DIR" in os.environ:
        return Path(os.environ["BACKEND_DIR"])
    
    # Then try current working directory
    cwd = Path.cwd()
    
    # Check if current directory looks like backend root
    # (has main.py and src directory)
    if (cwd / "main.py").exists() and (cwd / "src").exists():
        return cwd
    
    # Fallback: calculate from file location if provided
    if fallback_file:
        file_path = Path(fallback_file).resolve()
        # Navigate up to find backend root (has main.py)
        for parent in [file_path.parent] + list(file_path.parents):
            if (parent / "main.py").exists() and (parent / "src").exists():
                return parent
    
    # Last resort: return current directory
    return cwd


def get_project_root(fallback_file: Optional[str] = None) -> Path:
    """
    Get the project root directory (same as backend root for this project).
    
    Alias for get_backend_dir() for compatibility with tool scripts.
    
    Args:
        fallback_file: Path to a file (usually __file__) to use for fallback calculation.
    
    Returns:
        Path to the project root directory
    """
    return get_backend_dir(fallback_file)
