"""
LearningAgent - Analyzes performance data and tunes scoring weights

This agent analyzes feedback from sales outcomes and forecasts to optimize
the weights used in recommendation scoring (profit, ddt, market, forecast).
It trains a regressor to learn optimal weights that improve performance over time.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from agents import Agent, RunContextWrapper, function_tool

# Ensure the data directory exists
DATA_DIR = Path("doso-ai/data")
PERFORMANCE_LOG = DATA_DIR / "performance_log.jsonl"
FORECAST_OUTPUT = DATA_DIR / "forecast_output.json"
DOSO_CONFIG = DATA_DIR / "doso_config.json"
LEARNING_LOGS = DATA_DIR / "run_log"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LEARNING_LOGS, exist_ok=True)


# Define data models
class DosoWeights(BaseModel):
    """Weights configuration for DOSO scoring algorithm"""
    profit_weight: float = 0.25
    ddt_weight: float = 0.25
    market_weight: float = 0.25
    forecast_weight: float = 0.25
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str = "elasticnet"
    
    def total(self) -> float:
        """Ensure weights sum to 1.0"""
        return self.profit_weight + self.ddt_weight + self.market_weight + self.forecast_weight
    
    def normalize(self) -> "DosoWeights":
        """Normalize weights to sum to 1.0"""
        total = self.total()
        if total == 0:
            # Default to equal weights if all weights are zero
            return DosoWeights()
            
        if abs(total - 1.0) < 0.001:  # Already normalized (accounting for floating point imprecision)
            return self
            
        self.profit_weight /= total
        self.ddt_weight /= total
        self.market_weight /= total
        self.forecast_weight /= total
        return self


class DosoConfig(BaseModel):
    """Configuration for the DOSO system"""
    weights: DosoWeights
    last_learning_cycle: str = Field(default_factory=lambda: datetime.now().isoformat())
    learning_cycles: int = 1
    

class LearningResult(BaseModel):
    """Result of a learning cycle"""
    old_weights: DosoWeights
    new_weights: DosoWeights
    samples_used: int
    model_type: str
    metrics: Dict[str, float]
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    cycle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class PerformanceData(BaseModel):
    """Combined performance data for learning"""
    config_id: str
    sale_date: str
    profit_score: float  # Normalized score from gross_profit
    ddt_score: float     # Normalized score from ddt (inverse, lower is better)
    forecast_score: float  # Normalized score from forecast accuracy
    market_score: float  # Placeholder, to be derived from external data
    outcome_rating: float  # Target variable for learning
    

# Utility functions
def load_feedback_data() -> List[Dict]:
    """Load feedback data from the JSONL store"""
    records = []
    
    if not PERFORMANCE_LOG.exists():
        return records
    
    with open(PERFORMANCE_LOG, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except Exception as e:
                print(f"Error loading feedback record: {e}")
    
    return records


def load_forecast_data() -> Dict[str, Any]:
    """Load forecast data from the forecast output file"""
    if not FORECAST_OUTPUT.exists():
        return {}
        
    try:
        with open(FORECAST_OUTPUT, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading forecast output: {e}")
        return {}


def load_current_config() -> DosoConfig:
    """Load current DOSO configuration"""
    if not DOSO_CONFIG.exists():
        # Return default configuration if file doesn't exist
        return DosoConfig(weights=DosoWeights())
    
    try:
        with open(DOSO_CONFIG, "r") as f:
            config_data = json.load(f)
            return DosoConfig.model_validate(config_data)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        # Return default configuration on error
        return DosoConfig(weights=DosoWeights())


def save_config(config: DosoConfig) -> bool:
    """Save DOSO configuration to file"""
    try:
        with open(DOSO_CONFIG, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
            
        # Also save a timestamped copy to the learning logs
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = LEARNING_LOGS / f"config_{timestamp}.json"
        
        with open(log_file, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


def save_learning_result(result: LearningResult) -> bool:
    """Save learning result to log file"""
    try:
        # Save to learning logs
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = LEARNING_LOGS / f"learning_{timestamp}.json"
        
        with open(log_file, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving learning result: {e}")
        return False


def prepare_performance_data(feedback_data: List[Dict], forecast_data: Dict[str, Any]) -> List[PerformanceData]:
    """Prepare combined performance data for learning"""
    if not feedback_data:
        return []
        
    # Extract forecast accuracy information
    forecast_accuracy = {}
    for forecast in forecast_data:
        if not isinstance(forecast, dict):
            continue
            
        config_id = forecast.get('config_id')
        mape = forecast.get('mape')
        
        if config_id and mape is not None:
            # Convert MAPE to accuracy score (higher is better)
            accuracy = max(0, min(1, 1 - (mape / 100)))
            forecast_accuracy[config_id] = accuracy
    
    # Process feedback data
    performance_data = []
    
    # Get min/max values for normalization
    gross_profits = [record.get('gross_profit', 0) for record in feedback_data]
    min_profit = min(gross_profits) if gross_profits else 0
    max_profit = max(gross_profits) if gross_profits else 1
    profit_range = max_profit - min_profit
    
    ddts = [record.get('ddt', 0) for record in feedback_data]
    min_ddt = min(ddts) if ddts else 0
    max_ddt = max(ddts) if ddts else 1
    ddt_range = max_ddt - min_ddt
    
    for record in feedback_data:
        try:
            config_id = record.get('config_id')
            outcome_rating = record.get('outcome_rating')
            
            # Skip records without outcome rating
            if outcome_rating is None:
                if 'actual_sold' in record and 'recommended_qty' in record and record['recommended_qty'] > 0:
                    # Calculate a ratio-based outcome rating
                    actual = record['actual_sold']
                    recommended = record['recommended_qty']
                    outcome_rating = min(1.0, actual / recommended) if recommended > 0 else 0.5
                else:
                    continue
            
            # Normalize profit score (higher profit is better)
            profit = record.get('gross_profit', 0)
            profit_score = (profit - min_profit) / max(1, profit_range)
            
            # Normalize DDT score (lower DDT is better, so invert)
            ddt = record.get('ddt', 0)
            ddt_score = 1 - ((ddt - min_ddt) / max(1, ddt_range))
            
            # Get forecast score from forecast accuracy data
            forecast_score = forecast_accuracy.get(config_id, 0.5)  # Default to 0.5 if not found
            
            # Placeholder for market score (future enhancement)
            market_score = 0.5  # Default market score
            
            performance_data.append(PerformanceData(
                config_id=config_id,
                sale_date=record.get('sale_date', ''),
                profit_score=profit_score,
                ddt_score=ddt_score,
                forecast_score=forecast_score,
                market_score=market_score,
                outcome_rating=outcome_rating
            ))
            
        except Exception as e:
            print(f"Error processing performance record: {e}")
    
    return performance_data


def train_linear_model(data: List[PerformanceData]) -> Tuple[DosoWeights, Dict[str, float]]:
    """Train a simple linear regression model to determine optimal weights"""
    if len(data) < 5:  # Need at least 5 samples for meaningful training
        # Return default weights
        return DosoWeights(), {}
    
    # Prepare data
    X = []
    y = []
    
    for item in data:
        X.append([
            item.profit_score,
            item.ddt_score,
            item.market_score,
            item.forecast_score
        ])
        y.append(item.outcome_rating)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Extract weights
    weights = [max(0, w) for w in model.coef_]  # Ensure non-negative weights
    total = sum(weights)
    
    if total > 0:
        normalized_weights = [w / total for w in weights]
    else:
        normalized_weights = [0.25, 0.25, 0.25, 0.25]  # Default to equal weights
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create weights object
    doso_weights = DosoWeights(
        profit_weight=normalized_weights[0],
        ddt_weight=normalized_weights[1],
        market_weight=normalized_weights[2],
        forecast_weight=normalized_weights[3],
        model_type="linear"
    )
    
    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "samples": len(data)
    }
    
    return doso_weights, metrics


def train_elastic_net_model(data: List[PerformanceData], alpha: float = 0.1, l1_ratio: float = 0.5) -> Tuple[DosoWeights, Dict[str, float]]:
    """Train an ElasticNet model to determine optimal weights with regularization"""
    if len(data) < 5:  # Need at least 5 samples for meaningful training
        # Return default weights
        return DosoWeights(), {}
    
    # Prepare data
    X = []
    y = []
    
    for item in data:
        X.append([
            item.profit_score,
            item.ddt_score,
            item.market_score,
            item.forecast_score
        ])
        y.append(item.outcome_rating)
    
    X = np.array(X)
    y = np.array(y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train ElasticNet model with regularization
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)
    
    # Extract weights
    weights = [max(0, w) for w in model.coef_]  # Ensure non-negative weights
    total = sum(weights)
    
    if total > 0:
        normalized_weights = [w / total for w in weights]
    else:
        normalized_weights = [0.25, 0.25, 0.25, 0.25]  # Default to equal weights
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create weights object
    doso_weights = DosoWeights(
        profit_weight=normalized_weights[0],
        ddt_weight=normalized_weights[1],
        market_weight=normalized_weights[2],
        forecast_weight=normalized_weights[3],
        model_type="elasticnet"
    )
    
    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "samples": len(data)
    }
    
    return doso_weights, metrics


def train_random_forest_model(data: List[PerformanceData]) -> Tuple[DosoWeights, Dict[str, float]]:
    """Train a Random Forest model and extract feature importances as weights"""
    if len(data) < 10:  # Need more samples for a meaningful random forest
        # Return default weights
        return DosoWeights(), {}
    
    # Prepare data
    X = []
    y = []
    
    for item in data:
        X.append([
            item.profit_score,
            item.ddt_score,
            item.market_score,
            item.forecast_score
        ])
        y.append(item.outcome_rating)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Extract feature importances as weights
    weights = model.feature_importances_
    total = sum(weights)
    
    if total > 0:
        normalized_weights = [w / total for w in weights]
    else:
        normalized_weights = [0.25, 0.25, 0.25, 0.25]  # Default to equal weights
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create weights object
    doso_weights = DosoWeights(
        profit_weight=normalized_weights[0],
        ddt_weight=normalized_weights[1],
        market_weight=normalized_weights[2],
        forecast_weight=normalized_weights[3],
        model_type="randomforest"
    )
    
    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "samples": len(data)
    }
    
    return doso_weights, metrics


def train_best_model(data: List[PerformanceData]) -> Tuple[DosoWeights, Dict[str, float], str]:
    """Train multiple models and select the best one based on cross-validation"""
    if len(data) < 5:  # Not enough data for meaningful training
        return DosoWeights(), {}, "default"
    
    # Prepare data
    X = []
    y = []
    
    for item in data:
        X.append([
            item.profit_score,
            item.ddt_score,
            item.market_score,
            item.forecast_score
        ])
        y.append(item.outcome_rating)
    
    X = np.array(X)
    y = np.array(y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use cross-validation to evaluate models
    kf = KFold(n_splits=min(5, len(data)), shuffle=True, random_state=42)
    
    models = {
        "linear": LinearRegression(),
        "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }
    
    # Add Random Forest only if we have enough data
    if len(data) >= 10:
        models["randomforest"] = RandomForestRegressor(n_estimators=100, random_state=42)
    
    best_model_name = None
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        scores = []
        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)  # R² score
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_model_name = name
            best_model = model
    
    # Train the best model on the full dataset
    if best_model_name == "linear":
        weights, metrics = train_linear_model(data)
    elif best_model_name == "elasticnet":
        weights, metrics = train_elastic_net_model(data)
    elif best_model_name == "randomforest":
        weights, metrics = train_random_forest_model(data)
    else:
        # Default to ElasticNet if something went wrong
        weights, metrics = train_elastic_net_model(data)
        best_model_name = "elasticnet"
    
    return weights, metrics, best_model_name


# Tool functions for the agent
@function_tool
async def run_learning_cycle(ctx: RunContextWrapper[Any], model_type: str = "auto", optimization_target: str = "balanced") -> Dict[str, Any]:
    """
    Run a learning cycle to optimize recommendation weights based on historical performance.
    
    Args:
        model_type: Type of model to use for learning ('linear', 'elasticnet', 'randomforest', or 'auto')
        optimization_target: Strategy for optimization ('balanced', 'profit', 'turnover', 'forecast')
    """
    try:
        # Load current configuration
        current_config = load_current_config()
        old_weights = current_config.weights
        
        # Load feedback and forecast data
        feedback_data = load_feedback_data()
        forecast_data = load_forecast_data()
        
        if not feedback_data:
            return {
                "status": "error",
                "message": "No feedback data available. Run the feedback collector first."
            }
        
        # Prepare combined performance data
        performance_data = prepare_performance_data(feedback_data, forecast_data)
        
        if not performance_data:
            return {
                "status": "error",
                "message": "Could not prepare performance data from available sources."
            }
        
        # Apply optimization target bias
        if optimization_target != "balanced":
            # Apply a bias towards the specified optimization target
            # This will influence the learning process to focus more on specific factors
            
            for item in performance_data:
                if optimization_target == "profit":
                    # Emphasize profit correlation with outcome
                    item.profit_score = (item.profit_score + item.outcome_rating) / 2
                elif optimization_target == "turnover":
                    # Emphasize DDT (days to turn) correlation with outcome
                    item.ddt_score = (item.ddt_score + item.outcome_rating) / 2
                elif optimization_target == "forecast":
                    # Emphasize forecast accuracy correlation with outcome
                    item.forecast_score = (item.forecast_score + item.outcome_rating) / 2
        
        # Train the model based on the selected type
        if model_type == "auto":
            new_weights, metrics, selected_model = train_best_model(performance_data)
        elif model_type == "linear":
            new_weights, metrics = train_linear_model(performance_data)
            selected_model = "linear"
        elif model_type == "elasticnet":
            new_weights, metrics = train_elastic_net_model(performance_data)
            selected_model = "elasticnet"
        elif model_type == "randomforest":
            new_weights, metrics = train_random_forest_model(performance_data)
            selected_model = "randomforest"
        else:
            return {
                "status": "error",
                "message": f"Invalid model type: {model_type}"
            }
        
        # Ensure the new weights are valid
        new_weights.normalize()
        
        # Check if we have meaningful metrics
        if not metrics:
            # Not enough data for meaningful training, keep the current weights
            return {
                "status": "warning",
                "message": f"Insufficient data for training ({len(performance_data)} samples). Keeping current weights.",
                "weights": old_weights.model_dump()
            }
        
        # Calculate weight changes
        weight_changes = {
            "profit": new_weights.profit_weight - old_weights.profit_weight,
            "ddt": new_weights.ddt_weight - old_weights.ddt_weight,
            "market": new_weights.market_weight - old_weights.market_weight,
            "forecast": new_weights.forecast_weight - old_weights.forecast_weight
        }
        
        # Create learning result
        learning_result = LearningResult(
            old_weights=old_weights,
            new_weights=new_weights,
            samples_used=len(performance_data),
            model_type=selected_model,
            metrics=metrics
        )
        
        # Save learning result
        save_learning_result(learning_result)
        
        # Update and save configuration
        current_config.weights = new_weights
        current_config.last_learning_cycle = datetime.now().isoformat()
        current_config.learning_cycles += 1
        save_config(current_config)
        
        return {
            "status": "success",
            "message": f"Learning cycle completed with {selected_model} model using {len(performance_data)} samples",
            "old_weights": old_weights.model_dump(),
            "new_weights": new_weights.model_dump(),
            "weight_changes": weight_changes,
            "metrics": metrics,
            "model_type": selected_model,
            "optimization_target": optimization_target
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running learning cycle: {str(e)}"
        }


@function_tool
async def get_current_weights(ctx: RunContextWrapper[Any]) -> Dict[str, Any]:
    """
    Get the current recommendation weights from the configuration.
    """
    try:
        # Load current configuration
        current_config = load_current_config()
        weights = current_config.weights
        
        return {
            "status": "success",
            "weights": weights.model_dump(),
            "last_updated": current_config.last_learning_cycle,
            "learning_cycles": current_config.learning_cycles
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving current weights: {str(e)}"
        }


@function_tool
async def analyze_learning_history(ctx: RunContextWrapper[Any]) -> Dict[str, Any]:
    """
    Analyze the history of learning cycles and weight changes over time.
    """
    try:
        # Check for learning log files
        log_files = list(LEARNING_LOGS.glob("learning_*.json"))
        
        if not log_files:
            return {
                "status": "warning",
                "message": "No learning history found",
                "cycles": 0
            }
        
        # Load and analyze learning history
        history = []
        weight_trends = {
            "profit": [],
            "ddt": [],
            "market": [],
            "forecast": []
        }
        
        for log_file in sorted(log_files):
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)
                    
                    # Extract key information
                    cycle_info = {
                        "timestamp": data.get("created_at", ""),
                        "cycle_id": data.get("cycle_id", ""),
                        "model_type": data.get("model_type", ""),
                        "samples_used": data.get("samples_used", 0),
                        "old_weights": data.get("old_weights", {}),
                        "new_weights": data.get("new_weights", {}),
                        "metrics": data.get("metrics", {})
                    }
                    
                    history.append(cycle_info)
                    
                    # Track weight trends
                    if "new_weights" in data:
                        weights = data["new_weights"]
                        weight_trends["profit"].append(weights.get("profit_weight", 0.25))
                        weight_trends["ddt"].append(weights.get("ddt_weight", 0.25))
                        weight_trends["market"].append(weights.get("market_weight", 0.25))
                        weight_trends["forecast"].append(weights.get("forecast_weight", 0.25))
                    
            except Exception as e:
                print(f"Error processing log file {log_file}: {e}")
        
        # Calculate trends and statistics
        cycles = len(history)
        
        if cycles > 0:
            # Get first and last cycle for comparison
            first_cycle = history[0]
            last_cycle = history[-1]
            
            # Calculate weight changes from first to last
            first_weights = first_cycle.get("old_weights", {})
            last_weights = last_cycle.get("new_weights", {})
            
            overall_changes = {}
            for key in ["profit_weight", "ddt_weight", "market_weight", "forecast_weight"]:
                first_value = first_weights.get(key, 0.25)
                last_value = last_weights.get(key, 0.25)
                overall_changes[key] = last_value - first_value
            
            # Calculate metrics trends if available
            metrics_trends = {}
            if all("metrics" in cycle and "r2" in cycle["metrics"] for cycle in history):
                metrics_trends["r2"] = [cycle["metrics"]["r2"] for cycle in history]
                metrics_trends["mse"] = [cycle["metrics"]["mse"] for cycle in history]
            
            return {
                "status": "success",
                "message": f"Found {cycles} learning cycles",
                "cycles": cycles,
                "weight_trends": weight_trends,
                "overall_changes": overall_changes,
                "metrics_trends": metrics_trends,
                "latest_weights": last_weights,
                "history_summary": history
            }
        
        return {
            "status": "warning",
            "message": "Learning history is incomplete or corrupted",
            "cycles": cycles
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error analyzing learning history: {str(e)}"
        }


@function_tool
async def set_manual_weights(ctx: RunContextWrapper[Any], profit_weight: Optional[float] = None, ddt_weight: Optional[float] = None, 
                             market_weight: Optional[float] = None, forecast_weight: Optional[float] = None) -> Dict[str, Any]:
    """
    Manually set recommendation weights without running a learning cycle.
    
    Args:
        profit_weight: Weight for profit factor (0-1)
        ddt_weight: Weight for days-to-turn factor (0-1)
        market_weight: Weight for market factor (0-1)
        forecast_weight: Weight for forecast factor (0-1)
    """
    try:
        # Load current configuration
        current_config = load_current_config()
        old_weights = current_config.weights
        
        # Create new weights, starting with current weights
        new_weights = DosoWeights(
            profit_weight=profit_weight if profit_weight is not None else old_weights.profit_weight,
            ddt_weight=ddt_weight if ddt_weight is not None else old_weights.ddt_weight,
            market_weight=market_weight if market_weight is not None else old_weights.market_weight,
            forecast_weight=forecast_weight if forecast_weight is not None else old_weights.forecast_weight,
            model_type="manual"
        )
        
        # Normalize weights
        new_weights.normalize()
        
        # Calculate weight changes
        weight_changes = {
            "profit": new_weights.profit_weight - old_weights.profit_weight,
            "ddt": new_weights.ddt_weight - old_weights.ddt_weight,
            "market": new_weights.market_weight - old_weights.market_weight,
            "forecast": new_weights.forecast_weight - old_weights.forecast_weight
        }
        
        # Update configuration
        current_config.weights = new_weights
        current_config.last_learning_cycle = datetime.now().isoformat()
        save_config(current_config)
        
        # Create and save learning result for tracking
        learning_result = LearningResult(
            old_weights=old_weights,
            new_weights=new_weights,
            samples_used=0,  # Manual setting, not based on samples
            model_type="manual",
            metrics={}
        )
        save_learning_result(learning_result)
        
        return {
            "status": "success",
            "message": "Weights updated manually",
            "old_weights": old_weights.model_dump(),
            "new_weights": new_weights.model_dump(),
            "weight_changes": weight_changes
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error setting manual weights: {str(e)}"
        }


# Define the learning agent
learning_agent = Agent(
    name="Learning Agent",
    description="Analyzes performance data and tunes scoring weights",
    instructions="""
    You are an agent that optimizes recommendation weights based on historical performance data.
    Your responsibilities include:
    
    1. Analyzing feedback data to identify patterns and correlations
    2. Training models to determine optimal weights for different factors
    3. Providing insights on how weights affect recommendation performance
    4. Supporting manual weight adjustments when needed
    
    When analyzing data, consider both profitability and turnover speed metrics.
    Document the reasoning behind weight adjustments and their expected impact.
    """,
    tools=[
        run_learning_cycle,
        get_current_weights,
        analyze_learning_history,
        set_manual_weights
    ],
    model="gpt-4o",
)
