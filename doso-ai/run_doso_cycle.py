#!/usr/bin/env python
"""
DOSO AI Self-Learning System Runner

This script runs the DOSO AI learning cycle that processes feedback data,
generates forecasts, and produces optimized vehicle configuration recommendations.

Usage:
    python run_doso_cycle.py --feedback FILE --sales FILE [--configs ID1 ID2 ...]
    python run_doso_cycle.py --help

Example:
    python run_doso_cycle.py --feedback sample_data/feedback_sample.csv --sales sample_data/sales_sample.csv --configs config1 config2
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure we can import from src modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the workflow runner
from src.workflow.doso_workflow import run_doso_cycle

# Ensure data directories exist
DATA_DIR = Path("data")
RUN_LOGS_DIR = DATA_DIR / "run_log"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RUN_LOGS_DIR, exist_ok=True)


def save_cycle_results(results):
    """Save cycle results to a timestamped JSON file"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_file = RUN_LOGS_DIR / f"cycle_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results_file


async def main():
    """Main entry point for the DOSO cycle runner"""
    parser = argparse.ArgumentParser(description="DOSO AI Self-Learning System Runner")
    
    parser.add_argument(
        "--feedback", 
        type=str,
        required=True,
        help="Path to the feedback CSV file"
    )
    
    parser.add_argument(
        "--sales", 
        type=str,
        required=True,
        help="Path to the sales history CSV file"
    )
    
    parser.add_argument(
        "--configs", 
        type=str,
        nargs="+",
        help="Configuration IDs to generate recommendations for (optional)"
    )
    
    parser.add_argument(
        "--save-results", 
        action="store_true",
        default=True,
        help="Save results to a JSON file (default: True)"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.feedback):
        print(f"Error: Feedback file not found: {args.feedback}")
        return 1
    
    if not os.path.exists(args.sales):
        print(f"Error: Sales file not found: {args.sales}")
        return 1
    
    print("=" * 80)
    print("DOSO AI Self-Learning System Runner")
    print("=" * 80)
    print(f"Feedback file: {args.feedback}")
    print(f"Sales file: {args.sales}")
    
    if args.configs:
        print(f"Generating recommendations for: {', '.join(args.configs)}")
    else:
        print("No configurations specified for recommendations")
    
    print("\nStarting DOSO cycle...")
    print("-" * 80)
    
    try:
        # Run the DOSO cycle
        results = await run_doso_cycle(
            feedback_file=args.feedback,
            sales_file=args.sales,
            config_ids=args.configs
        )
        
        # Save results if requested
        if args.save_results:
            results_file = save_cycle_results(results)
            print(f"\nResults saved to: {results_file}")
        
        # Print summary of results
        print("\nCycle Results:")
        print(f"Status: {results['status']}")
        print(f"Started: {results['started_at']}")
        print(f"Completed: {results['completed_at']}")
        
        for stage, result in results.get('stages', {}).items():
            status = result.get('status', 'unknown')
            status_symbol = "✓" if status == "success" else "!" if status == "warning" else "✗"
            print(f"{status_symbol} {stage.capitalize()}: {status}")
        
        # Return appropriate exit code
        return 0 if results['status'] == "success" else 1
    
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        return 130
    except Exception as e:
        print(f"\nError during cycle execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
