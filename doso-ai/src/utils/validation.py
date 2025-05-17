"""
Validation utilities for DOSO AI system.
"""

from datetime import datetime, timedelta
from typing import Any, List


def validate_data_freshness(data_points: List[Any], max_age_hours: int = 24) -> bool:
    """
    Validate that data points are fresh enough for analysis.

    Args:
        data_points: List of data points with timestamp attribute
        max_age_hours: Maximum allowed age in hours (default: 24)

    Returns:
        bool: True if data is fresh enough, False otherwise

    """
    if not data_points:
        return False

    current_time = datetime.now()
    max_age = timedelta(hours=max_age_hours)

    for point in data_points:
        if not hasattr(point, "timestamp"):
            return False

        age = current_time - point.timestamp
        if age > max_age:
            return False

    return True


def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
    """
    Validate that a numeric value falls within an acceptable range.

    Args:
        value: Value to validate
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value

    Returns:
        bool: True if value is within range, False otherwise

    """
    return min_val <= value <= max_val


def validate_market_data_completeness(
    market_data: dict,
    required_fields: List[str],
    allow_missing: float = 0.1,
) -> bool:
    """
    Validate that market data contains required fields.

    Args:
        market_data: Dictionary containing market data
        required_fields: List of required field names
        allow_missing: Maximum fraction of missing fields allowed (default: 0.1)

    Returns:
        bool: True if data is complete enough, False otherwise

    """
    if not market_data:
        return False

    missing_count = 0
    for field in required_fields:
        if field not in market_data or market_data[field] is None:
            missing_count += 1

    missing_fraction = missing_count / len(required_fields)
    return missing_fraction <= allow_missing
