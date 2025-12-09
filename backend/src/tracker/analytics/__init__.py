"""
Analytics module for tracking API usage and operating costs.

This module provides tools for monitoring and analyzing the costs
associated with running the vehicle classification pipeline.
"""

from src.tracker.analytics.cost_tracker import (
    CostTracker,
    APICallRecord,
    get_cost_tracker,
)
from src.tracker.analytics.pricing import (
    PRICING,
    calculate_cost,
    get_model_pricing,
)

__all__ = [
    "CostTracker",
    "APICallRecord", 
    "get_cost_tracker",
    "PRICING",
    "calculate_cost",
    "get_model_pricing",
]
