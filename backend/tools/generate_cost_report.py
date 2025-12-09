#!/usr/bin/env python3
"""
Generate operating cost report from a classification session.

This tool creates human-readable cost reports showing:
- Total API calls and token usage
- Cost breakdown by classifier type
- Cost breakdown by truck
- Per-truck average cost

Usage:
    python tools/generate_cost_report.py [--output-dir PATH]
    
The report is generated from the CostTracker singleton which accumulates
all API calls during a classification session.

Note: This is typically called automatically at the end of main.py,
but can be run standalone to regenerate reports from the tracker state.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tracker.analytics import get_cost_tracker
from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_cost_report(output_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Generate cost report files in the specified directory.
    
    Creates:
    - operating_costs.json: Detailed JSON report with all call records
    - operating_costs.csv: Summary CSV for spreadsheet analysis
    
    Args:
        output_dir: Directory to write report files
        
    Returns:
        Tuple of (json_path, csv_path) or (None, None) if no data
    """
    cost_tracker = get_cost_tracker()
    
    # Export reports
    json_path, csv_path = cost_tracker.export_report(output_dir)
    
    return json_path, csv_path


def print_cost_summary():
    """Print a human-readable cost summary to console."""
    cost_tracker = get_cost_tracker()
    cost_tracker.print_summary()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate operating cost report from classification session"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory to write report files (default: outputs/reports)"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print summary to console, don't write files"
    )
    
    args = parser.parse_args()
    
    print_cost_summary()
    
    if not args.print_only:
        json_path, csv_path = generate_cost_report(args.output_dir)
        
        if json_path and csv_path:
            print(f"\nReports written:")
            print(f"  JSON: {json_path}")
            print(f"  CSV:  {csv_path}")
        else:
            print("\nNo API calls recorded - no reports generated.")


if __name__ == "__main__":
    main()
