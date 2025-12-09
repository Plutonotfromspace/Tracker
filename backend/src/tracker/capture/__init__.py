"""Capture management modules for delayed and intelligent capture strategies."""

from src.tracker.capture.delayed_capture import (
    CaptureCandidate,
    CaptureResult,
    DelayedCaptureManager,
)

__all__ = [
    "CaptureCandidate",
    "CaptureResult",
    "DelayedCaptureManager",
]
