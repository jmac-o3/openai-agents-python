"""
ForecastingAgent - Generates demand forecasts for vehicle configurations

This agent generates 8-week rolling demand forecasts for vehicle configurations
using historical sales data. It utilizes Prophet or similar time series models
to create predictions that other agents can use.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dateutil.relativedelta import relativedelta

from agents import Agent, RunContextWrapper, function_tool

# Ensure the data directory exists
DATA_DIR = Path("doso-ai/data")
FORECAST_OUTPUT = DATA_DIR / "forecast_output.json"
FORECAST_LOGS = DATA_DIR / "run_log"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FORECAST_LOGS, exist_ok=True)


class ForecastPeriod(TypedDict):
    """A period within a forecast"""
    date: str  # ISO format date
    quantity: float
    lower_bound: float
    upper_bound: float
    model: str


class ConfigForecast(BaseModel):
    """Forecast for a specific vehicle configuration"""
    config_id: str
    periods: List[ForecastPeriod]
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_type: str
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    mae: Optional[float] = None   # Mean Absolute Error
    forecast_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SalesData(BaseModel):
    """Sales history data for a configuration"""
    config_id: str
    dates: List[str]
    quantities: List[int]


class TrainingResult(BaseModel):
    """Result of training a forecast model"""
    mape: Optional[float] = None
    mae: Optional[float] = None
    model_type: str
    success: bool = True
    error_message: Optional[str] = None


class ForecastSummary(BaseModel):
    """Summary of forecast generation"""
    total_configs: int
    successful_forecasts: int
    failed_forecasts: int = 0
    model_types: List[str]
    average_mape: Optional[float] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "success"
    message: Optional[str] = None


def load_sales_history(file_path: str) -> List[SalesData]:
    """Load sales history data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Validate that required columns exist
        required_columns = ['date', 'config_id', 'quantity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Group by config_id
        configs = []
        for config_id, group in df.groupby('config_id'):
            group = group.sort_values('date')
            
            sales_data = SalesData(
                config_id=str(config_id),
                dates=group['date'].tolist(),
                quantities=group['quantity'].tolist()
            )
            configs.append(sales_data)
            
        return configs
        
    except Exception as e:
        print(f"Error loading sales history: {e}")
        return []


def save_forecast_output(forecasts: List[ConfigForecast]) -> bool:
    """Save forecasts to the forecast output file"""
    try:
        with open(FORECAST_OUTPUT, "w") as f:
            json.dump(
                [forecast.model_dump() for forecast in forecasts], 
                f, 
                indent=2
            )
        
        # Also save a timestamped copy to the run log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = FORECAST_LOGS / f"forecast_{timestamp}.json"
        
        with open(log_file, "w") as f:
            json.dump(
                [forecast.model_dump() for forecast in forecasts], 
                f, 
                indent=2
            )
            
        return True
    except Exception as e:
        print(f"Error saving forecast output: {e}")
        return False


def load_forecast_output() -> List[ConfigForecast]:
    """Load forecasts from the forecast output file"""
    if not FORECAST_OUTPUT.exists():
        return []
        
    try:
        with open(FORECAST_OUTPUT, "r") as f:
            forecast_data = json.load(f)
            return [ConfigForecast.model_validate(forecast) for forecast in forecast_data]
    except Exception as e:
        print(f"Error loading forecast output: {e}")
        return []


def train_prophet_model(sales_data: SalesData, periods: int = 8) -> Tuple[List[ForecastPeriod], TrainingResult]:
    """Train a Prophet model for time series forecasting"""
    try:
        # Try to import Prophet - this might fail if it's not installed
        try:
            from prophet import Prophet
        except ImportError:
            # If Prophet is not installed, use a fallback model
            print("Prophet not installed. Using statsmodels ETS model instead.")
            return train_ets_model(sales_data, periods)
            
        # Create a dataframe for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(sales_data.dates),
            'y': sales_data.quantities
        })
        
        # Train Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(df)
        
        # Generate future dates
        last_date = pd.to_datetime(sales_data.dates[-1])
        future = model.make_future_dataframe(periods=periods, freq='W')
        
        # Make predictions
        forecast = model.predict(future)
        
        # Extract the forecast periods
        forecast_periods = []
        
        # Calculate MAPE and MAE on the test set (last 20% of data)
        test_size = max(1, int(len(df) * 0.2))
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]
        
        if not test_df.empty:
            model_test = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model_test.fit(train_df)
            
            future_test = model_test.make_future_dataframe(periods=test_size, freq='W')
            forecast_test = model_test.predict(future_test)
            
            # Calculate errors
            y_true = test_df['y'].values
            y_pred = forecast_test.tail(test_size)['yhat'].values
            
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1, y_true))) * 100
            mae = np.mean(np.abs(y_true - y_pred))
        else:
            mape = None
            mae = None
            
        # Get only future dates
        future_forecast = forecast[forecast['ds'] > last_date]
        
        for _, row in future_forecast.iterrows():
            forecast_period: ForecastPeriod = {
                'date': row['ds'].strftime('%Y-%m-%d'),
                'quantity': float(max(0, row['yhat'])),
                'lower_bound': float(max(0, row['yhat_lower'])),
                'upper_bound': float(max(0, row['yhat_upper'])),
                'model': 'prophet'
            }
            forecast_periods.append(forecast_period)
            
        training_result = TrainingResult(
            mape=mape,
            mae=mae,
            model_type='prophet',
            success=True
        )
            
        return forecast_periods, training_result
            
    except Exception as e:
        # If Prophet training fails, fall back to ETS model
        print(f"Error training Prophet model: {e}")
        print("Falling back to ETS model")
        return train_ets_model(sales_data, periods)


def train_arima_model(sales_data: SalesData, periods: int = 8) -> Tuple[List[ForecastPeriod], TrainingResult]:
    """Train an ARIMA model for time series forecasting"""
    try:
        # Create a time series from the sales data
        ts = pd.Series(
            data=sales_data.quantities,
            index=pd.to_datetime(sales_data.dates)
        )
        
        # Fit ARIMA model
        model = ARIMA(ts, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=periods)
        forecast_index = pd.date_range(
            start=pd.to_datetime(sales_data.dates[-1]) + pd.Timedelta(days=7),
            periods=periods,
            freq='W'
        )
        
        # Calculate confidence intervals
        pred_interval = model_fit.get_forecast(steps=periods).conf_int()
        lower_bounds = pred_interval.iloc[:, 0]
        upper_bounds = pred_interval.iloc[:, 1]
        
        # Create forecast periods
        forecast_periods = []
        for i, date in enumerate(forecast_index):
            forecast_period: ForecastPeriod = {
                'date': date.strftime('%Y-%m-%d'),
                'quantity': float(max(0, forecast[i])),
                'lower_bound': float(max(0, lower_bounds[i])),
                'upper_bound': float(max(0, upper_bounds[i])),
                'model': 'arima'
            }
            forecast_periods.append(forecast_period)
        
        # Calculate MAPE and MAE on test set
        test_size = max(1, int(len(ts) * 0.2))
        if test_size > 0:
            train, test = ts[:-test_size], ts[-test_size:]
            model_test = ARIMA(train, order=(1, 1, 1))
            model_test_fit = model_test.fit()
            
            # Generate forecast for test period
            test_forecast = model_test_fit.forecast(steps=test_size)
            
            # Calculate errors
            y_true = test.values
            y_pred = test_forecast.values
            
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1, y_true))) * 100
            mae = np.mean(np.abs(y_true - y_pred))
        else:
            mape = None
            mae = None
            
        training_result = TrainingResult(
            mape=mape,
            mae=mae,
            model_type='arima',
            success=True
        )
            
        return forecast_periods, training_result
            
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        error_result = TrainingResult(
            model_type='arima',
            success=False,
            error_message=str(e)
        )
        return [], error_result


def train_ets_model(sales_data: SalesData, periods: int = 8) -> Tuple[List[ForecastPeriod], TrainingResult]:
    """Train an ETS (Exponential Smoothing) model for time series forecasting"""
    try:
        # Create a time series from the sales data
        ts = pd.Series(
            data=sales_data.quantities,
            index=pd.to_datetime(sales_data.dates)
        )
        
        # Need at least 2 seasons of data for multiplicative seasonal
        if len(ts) >= 104:  # 2 years of weekly data
            model = ExponentialSmoothing(
                ts, 
                seasonal_periods=52, 
                trend='add',
                seasonal='mul',
                damped_trend=True
            )
        elif len(ts) >= 52:  # 1 year of weekly data
            model = ExponentialSmoothing(
                ts, 
                seasonal_periods=52, 
                trend='add',
                seasonal='add',
                damped_trend=True
            )
        else:
            # Not enough data for seasonal model
            model = ExponentialSmoothing(
                ts,
                trend='add',
                damped_trend=True
            )
            
        model_fit = model.fit()
        
        # Generate forecast
        forecast_index = pd.date_range(
            start=pd.to_datetime(sales_data.dates[-1]) + pd.Timedelta(days=7),
            periods=periods,
            freq='W'
        )
        
        forecast = model_fit.forecast(periods)
        
        # Simple approximation for confidence intervals (±20%)
        lower_bounds = forecast * 0.8
        upper_bounds = forecast * 1.2
        
        # Create forecast periods
        forecast_periods = []
        for i, date in enumerate(forecast_index):
            forecast_period: ForecastPeriod = {
                'date': date.strftime('%Y-%m-%d'),
                'quantity': float(max(0, forecast[i])),
                'lower_bound': float(max(0, lower_bounds[i])),
                'upper_bound': float(max(0, upper_bounds[i])),
                'model': 'ets'
            }
            forecast_periods.append(forecast_period)
        
        # Calculate MAPE and MAE on test set
        test_size = max(1, int(len(ts) * 0.2))
        if test_size > 0:
            train, test = ts[:-test_size], ts[-test_size:]
            
            if len(train) >= 104:
                model_test = ExponentialSmoothing(
                    train, 
                    seasonal_periods=52, 
                    trend='add',
                    seasonal='mul',
                    damped_trend=True
                )
            elif len(train) >= 52:
                model_test = ExponentialSmoothing(
                    train, 
                    seasonal_periods=52, 
                    trend='add',
                    seasonal='add',
                    damped_trend=True
                )
            else:
                model_test = ExponentialSmoothing(
                    train,
                    trend='add',
                    damped_trend=True
                )
                
            model_test_fit = model_test.fit()
            
            # Generate forecast for test period
            test_forecast = model_test_fit.forecast(test_size)
            
            # Calculate errors
            y_true = test.values
            y_pred = test_forecast.values
            
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1, y_true))) * 100
            mae = np.mean(np.abs(y_true - y_pred))
        else:
            mape = None
            mae = None
            
        training_result = TrainingResult(
            mape=mape,
            mae=mae,
            model_type='ets',
            success=True
        )
            
        return forecast_periods, training_result
            
    except Exception as e:
        print(f"Error training ETS model: {e}")
        error_result = TrainingResult(
            model_type='ets',
            success=False,
            error_message=str(e)
        )
        return [], error_result


def train_best_model(sales_data: SalesData, periods: int = 8) -> Tuple[List[ForecastPeriod], str]:
    """Train multiple models and select the best one based on error metrics"""
    # Try each model type
    models_to_try = [
        ('prophet', train_prophet_model),
        ('arima', train_arima_model),
        ('ets', train_ets_model)
    ]
    
    best_mape = float('inf')
    best_model_forecast = []
    best_model_type = ""
    
    for model_name, model_trainer in models_to_try:
        try:
            forecast_periods, training_result = model_trainer(sales_data, periods)
            
            if not training_result.success or not forecast_periods:
                continue
                
            # Use MAPE as the selection criterion if available
            if training_result.mape is not None and training_result.mape < best_mape:
                best_mape = training_result.mape
                best_model_forecast = forecast_periods
                best_model_type = model_name
                
        except Exception as e:
            print(f"Error training {model_name} model: {e}")
    
    # If no model worked well, default to ETS as a fallback
    if not best_model_forecast:
        fallback_forecast, _ = train_ets_model(sales_data, periods)
        return fallback_forecast, 'ets'
        
    return best_model_forecast, best_model_type


@function_tool
async def generate_forecasts(ctx: RunContextWrapper[Any], file_path: str, periods: int = 8, model_type: str = "auto") -> Dict[str, Any]:
    """
    Generate demand forecasts for vehicle configurations using historical sales data.
    
    Args:
        file_path: Path to the CSV file containing sales history data
        periods: Number of future periods (weeks) to forecast
        model_type: The type of model to use ('prophet', 'arima', 'ets', or 'auto')
    """
    try:
        # Load sales history data
        sales_data_list = load_sales_history(file_path)
        
        if not sales_data_list:
            return {
                "status": "error",
                "message": "No valid sales data found in the file"
            }
        
        # Generate forecasts for each configuration
        forecasts = []
        successful_forecasts = 0
        failed_forecasts = 0
        model_types_used = set()
        total_mape = 0.0
        mape_count = 0
        
        for sales_data in sales_data_list:
            try:
                # Train the model and generate forecast
                if model_type == "auto":
                    # Try different models and select the best one
                    forecast_periods, selected_model = train_best_model(sales_data, periods)
                    model_types_used.add(selected_model)
                    
                    # Create ConfigForecast object
                    forecast = ConfigForecast(
                        config_id=sales_data.config_id,
                        periods=forecast_periods,
                        model_type=selected_model
                    )
                    
                    # Calculate accuracy metrics if possible
                    if len(sales_data.quantities) > 5:  # Need enough data for test/train split
                        test_size = max(1, len(sales_data.quantities) // 5)
                        test_actual = sales_data.quantities[-test_size:]
                        test_dates = sales_data.dates[-test_size:]
                        
                        # Train on history except test period
                        train_data = SalesData(
                            config_id=sales_data.config_id,
                            dates=sales_data.dates[:-test_size],
                            quantities=sales_data.quantities[:-test_size]
                        )
                        
                        # Generate forecast for test period
                        test_forecast, _ = train_best_model(train_data, test_size)
                        
                        if test_forecast:
                            # Calculate metrics
                            test_pred = [p['quantity'] for p in test_forecast]
                            
                            # Mean Absolute Error
                            mae = np.mean([abs(p - a) for p, a in zip(test_pred, test_actual)])
                            
                            # Mean Absolute Percentage Error
                            mape_values = [abs((p - a) / max(1, a)) for p, a in zip(test_pred, test_actual)]
                            mape = np.mean(mape_values) * 100
                            
                            forecast.mae = mae
                            forecast.mape = mape
                            
                            total_mape += mape
                            mape_count += 1
                else:
                    # Use specified model type
                    if model_type == "prophet":
                        forecast_periods, training_result = train_prophet_model(sales_data, periods)
                    elif model_type == "arima":
                        forecast_periods, training_result = train_arima_model(sales_data, periods)
                    elif model_type == "ets":
                        forecast_periods, training_result = train_ets_model(sales_data, periods)
                    else:
                        return {
                            "status": "error",
                            "message": f"Invalid model type: {model_type}"
                        }
                    
                    if not training_result.success:
                        failed_forecasts += 1
                        continue
                        
                    model_types_used.add(model_type)
                    
                    # Create ConfigForecast object
                    forecast = ConfigForecast(
                        config_id=sales_data.config_id,
                        periods=forecast_periods,
                        model_type=model_type,
                        mape=training_result.mape,
                        mae=training_result.mae
                    )
                    
                    if training_result.mape is not None:
                        total_mape += training_result.mape
                        mape_count += 1
                
                forecasts.append(forecast)
                successful_forecasts += 1
                
            except Exception as e:
                print(f"Error generating forecast for config {sales_data.config_id}: {e}")
                failed_forecasts += 1
        
        # Save forecasts to output file
        save_forecast_output(forecasts)
        
        # Calculate average MAPE
        avg_mape = total_mape / mape_count if mape_count > 0 else None
        
        # Create summary
        summary = ForecastSummary(
            total_configs=len(sales_data_list),
            successful_forecasts=successful_forecasts,
            failed_forecasts=failed_forecasts,
            model_types=list(model_types_used),
            average_mape=avg_mape
        )
        
        return {
            "status": "success",
            "message": f"Generated forecasts for {successful_forecasts} configurations with {failed_forecasts} failures",
            "forecasts_generated": successful_forecasts,
            "models_used": list(model_types_used),
            "average_mape": avg_mape,
            "summary": summary.model_dump()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating forecasts: {str(e)}"
        }


@function_tool
async def forecast_demand(ctx: RunContextWrapper[Any], config_id: str) -> Dict[str, Any]:
    """
    Get demand forecast for a specific vehicle configuration.
    
    Args:
        config_id: The configuration ID to get forecast for
    """
    try:
        # Load forecasts from output file
        forecasts = load_forecast_output()
        
        if not forecasts:
            return {
                "status": "error",
                "message": "No forecasts available. Run generate_forecasts first."
            }
        
        # Find forecast for the specified config_id
        config_forecast = next((f for f in forecasts if f.config_id == config_id), None)
        
        if not config_forecast:
            return {
                "status": "warning",
                "message": f"No forecast found for config_id {config_id}",
                "forecast": None
            }
        
        # Extract data from forecast
        forecast_data = {
            "config_id": config_forecast.config_id,
            "model_type": config_forecast.model_type,
            "created_at": config_forecast.created_at,
            "mape": config_forecast.mape,
            "mae": config_forecast.mae,
            "periods": config_forecast.periods
        }
        
        return {
            "status": "success",
            "message": f"Found forecast for config_id {config_id}",
            "forecast": forecast_data
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving forecast: {str(e)}"
        }


@function_tool
async def list_available_forecasts(ctx: RunContextWrapper[Any]) -> Dict[str, Any]:
    """
    List all available forecasts with summary information.
    """
    try:
        # Load forecasts from output file
        forecasts = load_forecast_output()
        
        if not forecasts:
            return {
                "status": "warning",
                "message": "No forecasts available. Run generate_forecasts first.",
                "forecasts": []
            }
        
        # Create summary list
        forecast_summaries = []
        for f in forecasts:
            # Calculate average predicted quantity
            avg_qty = sum(p['quantity'] for p in f.periods) / len(f.periods) if f.periods else 0
            
            summary = {
                "config_id": f.config_id,
                "model_type": f.model_type,
                "created_at": f.created_at,
                "periods": len(f.periods),
                "avg_predicted_qty": avg_qty,
                "mape": f.mape,
                "forecast_id": f.forecast_id
            }
            forecast_summaries.append(summary)
        
        return {
            "status": "success",
            "message": f"Found {len(forecasts)} forecasts",
            "forecasts": forecast_summaries
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing forecasts: {str(e)}"
        }


# Define the forecasting agent
forecasting_agent = Agent(
    name="Forecasting Agent",
    description="Generates demand forecasts for vehicle configurations",
    instructions="""
    You are an agent that generates 8-week rolling demand forecasts for vehicle configurations.
    Your responsibilities include:
    
    1. Processing historical sales data from CSV files
    2. Training and evaluating time series forecasting models
    3. Generating and storing demand forecasts for each configuration
    4. Providing forecast data to other agents
    
    You can use different forecasting models (Prophet, ARIMA, ETS) depending on data characteristics.
    Forecast accuracy is evaluated using metrics like MAPE (Mean Absolute Percentage Error).
    """,
    tools=[
        generate_forecasts,
        forecast_demand,
        list_available_forecasts
    ],
    model="gpt-4o",
)
