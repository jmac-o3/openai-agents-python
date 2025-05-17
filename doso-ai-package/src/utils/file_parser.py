"""
File Parser utilities for DOSO AI

This module provides utilities for parsing various file formats,
including CSV, JSON, and text files. It handles validation and
preparation of data for use by the AI agents.
"""

import csv
import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Set

import numpy as np
import pandas as pd
from openai_agents import Agent, function_tool

from ..monitoring.tracing import trace_method, trace_async_method

logger = logging.getLogger(__name__)

class FileParser:
    """Utility class for parsing various file formats"""
    
    @staticmethod
    @trace_method("file_parser.parse_csv")
    def parse_csv(
        file_path: str, 
        expected_columns: Optional[List[str]] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Parse a CSV file into a pandas DataFrame
        
        Args:
            file_path: Path to the CSV file
            expected_columns: List of expected column names (for validation)
            validate: Whether to validate the columns
            
        Returns:
            Pandas DataFrame containing the parsed data
        
        Raises:
            ValueError: If the file doesn't exist or columns don't match expectations
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
            
        # Read the CSV file
        try:
            df = pd.read_csv(file_path)
            
            # Validate columns if requested
            if validate and expected_columns:
                missing_columns = set(expected_columns) - set(df.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                    
            return df
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {str(e)}")
            raise
    
    @staticmethod
    @trace_method("file_parser.parse_feedback_csv")
    def parse_feedback_csv(file_path: str) -> pd.DataFrame:
        """
        Parse a feedback CSV file with specific validations
        
        Expected columns:
        - config_id
        - sale_date
        - gross_profit
        - ddt (days to turn)
        - recommended_qty
        - actual_sold
        - constraint_hit
        - forecast_accuracy
        
        Args:
            file_path: Path to the feedback CSV file
            
        Returns:
            Pandas DataFrame with parsed and validated feedback data
        """
        expected_columns = [
            "config_id", "sale_date", "gross_profit", "ddt", 
            "recommended_qty", "actual_sold", "constraint_hit", "forecast_accuracy"
        ]
        
        df = FileParser.parse_csv(file_path, expected_columns)
        
        # Additional validations and preprocessing
        # Convert sale_date to datetime
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        
        # Ensure numeric columns are numeric
        numeric_columns = ["gross_profit", "ddt", "recommended_qty", "actual_sold", "forecast_accuracy"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # Replace NaN with appropriate values
        df["constraint_hit"] = df["constraint_hit"].fillna(False)
        
        # Drop rows with critical missing values
        df = df.dropna(subset=["config_id", "sale_date", "gross_profit"])
        
        return df
    
    @staticmethod
    @trace_method("file_parser.parse_sales_history_csv")
    def parse_sales_history_csv(file_path: str) -> pd.DataFrame:
        """
        Parse a sales history CSV file
        
        Args:
            file_path: Path to the sales history CSV file
            
        Returns:
            Pandas DataFrame with parsed sales history data
        """
        df = FileParser.parse_csv(file_path)
        
        # If we have 'date' or 'sale_date' column, convert to datetime
        date_columns = ['date', 'sale_date', 'order_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        return df
    
    @staticmethod
    @trace_method("file_parser.save_jsonl")
    def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save data as JSONL (one JSON object per line)
        
        Args:
            data: List of dictionaries to save
            file_path: Path to save the JSONL file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    
    @staticmethod
    @trace_method("file_parser.load_jsonl")
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a JSONL file
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries parsed from the JSONL file
        """
        if not os.path.exists(file_path):
            return []
            
        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
                    
        return data
    
    @staticmethod
    @trace_method("file_parser.save_json")
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        """
        Save data as a JSON file
        
        Args:
            data: Dictionary to save
            file_path: Path to save the JSON file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    @trace_method("file_parser.load_json")
    def load_json(file_path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load data from a JSON file
        
        Args:
            file_path: Path to the JSON file
            default: Default value to return if file doesn't exist
            
        Returns:
            Dictionary parsed from the JSON file
        """
        if not os.path.exists(file_path):
            return default if default is not None else {}
            
        with open(file_path, "r") as f:
            return json.load(f)
    
    @staticmethod
    @trace_method("file_parser.generate_run_log_path")
    def generate_run_log_path() -> str:
        """
        Generate a path for the run log file based on current date
        
        Returns:
            Path for the run log file
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_dir = os.path.join("doso-ai", "data", "run_log")
        os.makedirs(log_dir, exist_ok=True)
        
        return os.path.join(log_dir, f"{date_str}.json")
