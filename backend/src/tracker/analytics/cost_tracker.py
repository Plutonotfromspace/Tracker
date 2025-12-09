"""
Cost tracker for monitoring API usage and calculating operating costs.

This module provides a singleton CostTracker class that records all API calls
made during a classification session and calculates the associated costs.

Usage:
    from src.tracker.analytics import get_cost_tracker
    
    tracker = get_cost_tracker()
    tracker.record_call(APICallRecord(...))
    
    # At end of session
    stats = tracker.get_session_stats()
    tracker.export_report(Path("debug_images"))
"""

import csv
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tracker.analytics.pricing import calculate_cost, format_cost, get_model_pricing
from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class APICallRecord:
    """
    Record of a single API call.
    
    Attributes:
        timestamp: When the call was made
        truck_id: Associated truck ID (e.g., "Truck001")
        classifier_type: Type of classifier ("body_type", "axle_type", "small_vehicle")
        call_purpose: Purpose of call ("reasoning", "extraction")
        model: Model name used
        prompt_tokens: Number of input prompt tokens
        completion_tokens: Number of output completion tokens
        image_tokens: Number of image tokens (for vision calls)
        reasoning_tokens: Number of reasoning tokens (for reasoning models)
        cached_tokens: Number of cached tokens
        total_tokens: Total tokens used
        cost_usd: Calculated cost in USD
    """
    timestamp: datetime
    truck_id: Optional[str]
    classifier_type: str
    call_purpose: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    
    def __post_init__(self):
        """Calculate cost if not provided."""
        if self.cost_usd == 0.0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            self.cost_usd = calculate_cost(
                model=self.model,
                input_tokens=self.prompt_tokens,
                output_tokens=self.completion_tokens,
                image_tokens=self.image_tokens,
                cached_tokens=self.cached_tokens
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["cost_usd"] = round(self.cost_usd, 8)
        return d


class CostTracker:
    """
    Singleton class for tracking API usage and costs.
    
    Thread-safe implementation that accumulates API call records
    throughout a session and provides summary statistics.
    
    Example:
        >>> tracker = CostTracker.instance()
        >>> tracker.record_call(APICallRecord(...))
        >>> stats = tracker.get_session_stats()
        >>> print(f"Total cost: ${stats['total_cost_usd']:.4f}")
    """
    
    _instance: Optional["CostTracker"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "CostTracker":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the tracker (only runs once due to singleton)."""
        if self._initialized:
            return
        
        self._records: List[APICallRecord] = []
        self._session_start: datetime = datetime.now()
        self._lock = threading.Lock()
        self._initialized = True
    
    @classmethod
    def instance(cls) -> "CostTracker":
        """Get the singleton instance."""
        return cls()
    
    def record_call(self, record: APICallRecord) -> None:
        """
        Record an API call.
        
        Args:
            record: APICallRecord with call details
        """
        with self._lock:
            self._records.append(record)
            logger.debug(
                f"Recorded API call: {record.classifier_type}/{record.call_purpose} "
                f"- {record.total_tokens} tokens, ${record.cost_usd:.6f}"
            )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics for the current session.
        
        Returns:
            Dictionary with session statistics
        """
        with self._lock:
            if not self._records:
                return {
                    "session_start": self._session_start.isoformat(),
                    "session_end": datetime.now().isoformat(),
                    "total_api_calls": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_image_tokens": 0,
                    "total_reasoning_tokens": 0,
                    "total_cached_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                }
            
            stats = {
                "session_start": self._session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
                "total_api_calls": len(self._records),
                "total_prompt_tokens": sum(r.prompt_tokens for r in self._records),
                "total_completion_tokens": sum(r.completion_tokens for r in self._records),
                "total_image_tokens": sum(r.image_tokens for r in self._records),
                "total_reasoning_tokens": sum(r.reasoning_tokens for r in self._records),
                "total_cached_tokens": sum(r.cached_tokens for r in self._records),
                "total_tokens": sum(r.total_tokens for r in self._records),
                "total_cost_usd": sum(r.cost_usd for r in self._records),
            }
            
            return stats
    
    def get_breakdown_by_classifier(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cost breakdown by classifier type.
        
        Returns:
            Dictionary with stats per classifier type
        """
        with self._lock:
            breakdown = {}
            
            for record in self._records:
                ctype = record.classifier_type
                if ctype not in breakdown:
                    breakdown[ctype] = {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "image_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                    }
                
                breakdown[ctype]["calls"] += 1
                breakdown[ctype]["prompt_tokens"] += record.prompt_tokens
                breakdown[ctype]["completion_tokens"] += record.completion_tokens
                breakdown[ctype]["image_tokens"] += record.image_tokens
                breakdown[ctype]["total_tokens"] += record.total_tokens
                breakdown[ctype]["cost_usd"] += record.cost_usd
            
            return breakdown
    
    def get_breakdown_by_truck(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cost breakdown by truck ID.
        
        Returns:
            Dictionary with stats per truck
        """
        with self._lock:
            breakdown = {}
            
            for record in self._records:
                truck_id = record.truck_id or "unknown"
                if truck_id not in breakdown:
                    breakdown[truck_id] = {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "image_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                        "classifiers_used": set(),
                    }
                
                breakdown[truck_id]["calls"] += 1
                breakdown[truck_id]["prompt_tokens"] += record.prompt_tokens
                breakdown[truck_id]["completion_tokens"] += record.completion_tokens
                breakdown[truck_id]["image_tokens"] += record.image_tokens
                breakdown[truck_id]["total_tokens"] += record.total_tokens
                breakdown[truck_id]["cost_usd"] += record.cost_usd
                breakdown[truck_id]["classifiers_used"].add(record.classifier_type)
            
            # Convert sets to lists for JSON serialization
            for truck_id in breakdown:
                breakdown[truck_id]["classifiers_used"] = list(
                    breakdown[truck_id]["classifiers_used"]
                )
            
            return breakdown
    
    def get_model_used(self) -> Optional[str]:
        """Get the model name used (assumes single model per session)."""
        with self._lock:
            if self._records:
                return self._records[0].model
            return None
    
    def export_report(
        self,
        output_dir: Path,
        output_name: str = "operating_costs"
    ) -> tuple[Optional[Path], Optional[Path]]:
        """
        Export cost report to JSON and CSV files.
        
        Args:
            output_dir: Directory to write files
            output_name: Base name for output files
        
        Returns:
            Tuple of (json_path, csv_path) or (None, None) if no data
        """
        stats = self.get_session_stats()
        
        if stats["total_api_calls"] == 0:
            logger.info("No API calls recorded, skipping cost report")
            return None, None
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build full report
        report = {
            "session_id": self._session_start.strftime("%Y%m%d_%H%M%S"),
            "generated_at": datetime.now().isoformat(),
            "model": self.get_model_used(),
            "summary": stats,
            "by_classifier": self.get_breakdown_by_classifier(),
            "by_truck": self.get_breakdown_by_truck(),
            "detailed_calls": [r.to_dict() for r in self._records],
        }
        
        # Write JSON
        json_path = output_dir / f"{output_name}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Write CSV (summary row)
        csv_path = output_dir / f"{output_name}.csv"
        
        # Calculate averages
        truck_breakdown = self.get_breakdown_by_truck()
        num_trucks = len([t for t in truck_breakdown if t != "unknown"])
        avg_cost_per_truck = stats["total_cost_usd"] / num_trucks if num_trucks > 0 else 0
        
        csv_fields = [
            "session_id", "timestamp", "model", "total_api_calls",
            "prompt_tokens", "completion_tokens", "image_tokens",
            "reasoning_tokens", "cached_tokens", "total_tokens",
            "total_cost_usd", "num_trucks", "avg_cost_per_truck"
        ]
        
        csv_row = {
            "session_id": report["session_id"],
            "timestamp": datetime.now().isoformat(),
            "model": report["model"],
            "total_api_calls": stats["total_api_calls"],
            "prompt_tokens": stats["total_prompt_tokens"],
            "completion_tokens": stats["total_completion_tokens"],
            "image_tokens": stats["total_image_tokens"],
            "reasoning_tokens": stats["total_reasoning_tokens"],
            "cached_tokens": stats["total_cached_tokens"],
            "total_tokens": stats["total_tokens"],
            "total_cost_usd": round(stats["total_cost_usd"], 6),
            "num_trucks": num_trucks,
            "avg_cost_per_truck": round(avg_cost_per_truck, 6),
        }
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerow(csv_row)
        
        return json_path, csv_path
    
    def print_summary(self) -> None:
        """Print a human-readable summary to console."""
        stats = self.get_session_stats()
        classifier_breakdown = self.get_breakdown_by_classifier()
        truck_breakdown = self.get_breakdown_by_truck()
        
        num_trucks = len([t for t in truck_breakdown if t != "unknown"])
        avg_cost = stats["total_cost_usd"] / num_trucks if num_trucks > 0 else 0
        
        pass
    
    def reset(self) -> None:
        """Reset the tracker for a new session."""
        with self._lock:
            self._records.clear()
            self._session_start = datetime.now()
            logger.info("CostTracker reset")


# Convenience function
def get_cost_tracker() -> CostTracker:
    """Get the singleton CostTracker instance."""
    return CostTracker.instance()
