"""
DOSO AI - Complete Streamlit Application

This application provides a comprehensive interface for the DOSO AI system,
including file ingestion, agent orchestration, forecast generation,
scoring optimization, and feedback evaluation.

Usage:
    streamlit run doso-ai/doso_streamlit_app.py
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import DOSO agents and workflow
from doso_ai.src.agents.feedback_collector_agent import feedback_collector
from doso_ai.src.agents.forecasting_agent import forecasting_agent
from doso_ai.src.agents.recommendation_agent import recommendation_agent
from doso_ai.src.agents.learning_agent import learning_agent, DosoWeights
from doso_ai.src.workflow.doso_workflow import (
    run_doso_cycle, 
    run_feedback_collection, 
    run_forecasting, 
    run_learning_cycle, 
    run_recommendation, 
    run_batch_recommendations
)

# Configure page
st.set_page_config(
    page_title="DOSO AI | Complete System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define constants
DATA_DIR = Path("doso-ai/data")
UPLOADED_DIR = DATA_DIR / "uploaded"
FORECASTS_DIR = DATA_DIR / "forecasts"
RUN_LOG_DIR = DATA_DIR / "run_log"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
PERFORMANCE_LOG = DATA_DIR / "performance_log.jsonl"
FORECAST_OUTPUT = DATA_DIR / "forecast_output.json"
DOSO_CONFIG = DATA_DIR / "doso_config.json"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(FORECASTS_DIR, exist_ok=True)
os.makedirs(RUN_LOG_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize session state
if "dealer_id" not in st.session_state:
    st.session_state.dealer_id = "DEALER001"

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "cycle_results" not in st.session_state:
    st.session_state.cycle_results = None

if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None

if "recommendation_data" not in st.session_state:
    st.session_state.recommendation_data = None

if "learning_data" not in st.session_state:
    st.session_state.learning_data = None

if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"

if "processing_log" not in st.session_state:
    st.session_state.processing_log = []

# Add logging function
def add_to_log(message: str, level: str = "info") -> None:
    """
    Add a message to the processing log
    
    Args:
        message: Log message
        level: Log level (info, warning, error)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append({
        "timestamp": timestamp,
        "message": message,
        "level": level
    })


# File handling functions
def save_uploaded_file(file, file_type: str) -> str:
    """Save an uploaded file to the data directory and track it"""
    # Create timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.name}"
    file_path = UPLOADED_DIR / filename
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Store in session state by file type
    if file_type not in st.session_state.uploaded_files:
        st.session_state.uploaded_files[file_type] = []
    
    file_info = {
        "file_name": file.name,
        "path": str(file_path),
        "uploaded_at": datetime.now().isoformat()
    }
    
    st.session_state.uploaded_files[file_type].append(file_info)
    add_to_log(f"Uploaded {file_type} file: {file.name}")
    return str(file_path)


def load_doso_config() -> Dict:
    """Load DOSO configuration from file or create default"""
    if DOSO_CONFIG.exists():
        with open(DOSO_CONFIG, 'r') as f:
            return json.load(f)
    else:
        # Create default config
        default_config = {
            "weights": {
                "profit_margin": 0.3,
                "days_to_turn": 0.3,
                "market_demand": 0.25,
                "seasonal_factors": 0.15
            },
            "confidence_threshold": 0.7,
            "max_recommendations": 10,
            "learning_rate": 0.05
        }
        with open(DOSO_CONFIG, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config


def save_doso_config(config: Dict) -> None:
    """Save DOSO configuration to file"""
    with open(DOSO_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)


def update_performance_log(results: Dict) -> None:
    """Append performance metrics to the performance log"""
    timestamp = datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "dealer_id": st.session_state.dealer_id,
        "results": results
    }
    
    # Append to log file
    with open(PERFORMANCE_LOG, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def load_performance_log() -> List[Dict]:
    """Load performance log entries"""
    entries = []
    if PERFORMANCE_LOG.exists():
        with open(PERFORMANCE_LOG, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    return entries


# Async helper function
def run_async(coro):
    """Run an async coroutine in Streamlit"""
    return asyncio.run(coro)


# Agent workflow functions
def process_sales_history(file_path: str, output_path: Optional[str] = None) -> Dict:
    """Process sales history file with forecasting agent"""
    if not file_path:
        st.error("No sales history file provided")
        add_to_log("No sales history file provided", level="error")
        return {}
    
    if not output_path:
        output_path = str(FORECAST_OUTPUT)
        
    add_to_log(f"Processing sales history file: {file_path}")
    with st.spinner("Processing sales history with forecasting agent..."):
        # Run forecasting agent on the sales data
        start_time = time.time()
        result = run_async(run_forecasting(
            sales_file=file_path,
            output_path=output_path
        ))
        processing_time = time.time() - start_time
        
        if result and "status" in result and result["status"] == "success":
            st.session_state.forecast_data = result
            add_to_log(f"Sales forecasting completed successfully in {processing_time:.2f}s")
            st.success("Sales forecasting completed successfully!")
        else:
            add_to_log(f"Sales forecasting failed after {processing_time:.2f}s", level="error")
            st.error("Sales forecasting failed")
        
        # Add processing time to result
        result["processing_time"] = processing_time
        return result


def generate_recommendations(inventory_path: str, order_bank_path: str, forecast_data: Dict, market_path: Optional[str] = None) -> Dict:
    """Generate recommendations based on inventory, order bank, and forecast data"""
    if not inventory_path or not order_bank_path:
        st.error("Missing required files for recommendation generation")
        add_to_log("Missing required files for recommendation generation", level="error")
        return {}
        
    if not forecast_data:
        st.error("Forecast data is required for recommendation generation")
        add_to_log("Forecast data is required for recommendation generation", level="error")
        return {}
    
    add_to_log("Generating recommendations...")
    with st.spinner("Generating recommendations..."):
        # Load DOSO config for weights
        config = load_doso_config()
        weights = DosoWeights(
            profit_margin=config["weights"]["profit_margin"],
            days_to_turn=config["weights"]["days_to_turn"],
            market_demand=config["weights"]["market_demand"],
            seasonal_factors=config["weights"]["seasonal_factors"]
        )
        
        start_time = time.time()
        # Run recommendation agent
        result = run_async(run_recommendation(
            inventory_file=inventory_path,
            order_bank_file=order_bank_path,
            forecast_data=forecast_data,
            weights=weights,
            dealer_id=st.session_state.dealer_id,
            max_recommendations=config["max_recommendations"],
            confidence_threshold=config["confidence_threshold"],
            market_file=market_path
        ))
        processing_time = time.time() - start_time
        
        if result and "status" in result and result["status"] == "success":
            st.session_state.recommendation_data = result
            add_to_log(f"Recommendations generated successfully in {processing_time:.2f}s")
            st.success("Recommendations generated successfully!")
        else:
            add_to_log(f"Recommendation generation failed after {processing_time:.2f}s", level="error")
            st.error("Recommendation generation failed")
        
        # Add processing time to result
        result["processing_time"] = processing_time    
        return result


def collect_feedback(feedback_path: str, recommendation_data: Dict) -> Dict:
    """Collect and process feedback data for learning"""
    if not feedback_path:
        st.error("No feedback file provided")
        add_to_log("No feedback file provided", level="error")
        return {}
        
    if not recommendation_data:
        st.error("Recommendation data is required for feedback collection")
        add_to_log("Recommendation data is required for feedback collection", level="error")
        return {}
    
    add_to_log(f"Processing feedback file: {feedback_path}")
    with st.spinner("Processing feedback data..."):
        # Run feedback collection agent
        start_time = time.time()
        result = run_async(run_feedback_collection(
            feedback_file=feedback_path,
            recommendation_data=recommendation_data,
            dealer_id=st.session_state.dealer_id
        ))
        processing_time = time.time() - start_time
        
        if result and "status" in result and result["status"] == "success":
            st.session_state.feedback_data = result
            add_to_log(f"Feedback collection completed successfully in {processing_time:.2f}s")
            st.success("Feedback collection completed successfully!")
        else:
            add_to_log(f"Feedback collection failed after {processing_time:.2f}s", level="error")
            st.error("Feedback collection failed")
        
        # Add processing time to result
        result["processing_time"] = processing_time
        return result


def optimize_weights(feedback_data: Dict) -> Dict:
    """Optimize recommendation weights based on feedback data"""
    if not feedback_data:
        st.error("Feedback data is required for weight optimization")
        add_to_log("Feedback data is required for weight optimization", level="error")
        return {}
    
    add_to_log("Optimizing recommendation weights...")
    with st.spinner("Optimizing recommendation weights..."):
        # Load current config
        config = load_doso_config()
        current_weights = DosoWeights(
            profit_margin=config["weights"]["profit_margin"],
            days_to_turn=config["weights"]["days_to_turn"],
            market_demand=config["weights"]["market_demand"],
            seasonal_factors=config["weights"]["seasonal_factors"]
        )
        
        # Run learning agent
        start_time = time.time()
        result = run_async(run_learning_cycle(
            feedback_data=feedback_data,
            current_weights=current_weights,
            learning_rate=config["learning_rate"]
        ))
        processing_time = time.time() - start_time
        
        if result and "status" in result and result["status"] == "success":
            # Update config with new weights
            optimized_weights = result.get("optimized_weights", {})
            config["weights"]["profit_margin"] = optimized_weights.get("profit_margin", current_weights.profit_margin)
            config["weights"]["days_to_turn"] = optimized_weights.get("days_to_turn", current_weights.days_to_turn)
            config["weights"]["market_demand"] = optimized_weights.get("market_demand", current_weights.market_demand)
            config["weights"]["seasonal_factors"] = optimized_weights.get("seasonal_factors", current_weights.seasonal_factors)
            
            # Save updated config
            save_doso_config(config)
            
            st.session_state.learning_data = result
            add_to_log(f"Weight optimization completed successfully in {processing_time:.2f}s")
            st.success("Weight optimization completed successfully!")
        else:
            add_to_log(f"Weight optimization failed after {processing_time:.2f}s", level="error")
            st.error("Weight optimization failed")
        
        # Add processing time to result
        result["processing_time"] = processing_time
        return result


def run_full_doso_cycle() -> Dict:
    """Run the complete DOSO cycle"""
    # Check for required files
    if "sales_history" not in st.session_state.uploaded_files or not st.session_state.uploaded_files["sales_history"]:
        st.error("Sales history file is required")
        add_to_log("Sales history file is required", level="error")
        return {}
        
    if "inventory" not in st.session_state.uploaded_files or not st.session_state.uploaded_files["inventory"]:
        st.error("Inventory file is required")
        add_to_log("Inventory file is required", level="error")
        return {}
        
    if "order_bank" not in st.session_state.uploaded_files or not st.session_state.uploaded_files["order_bank"]:
        st.error("Order bank file is required")
        add_to_log("Order bank file is required", level="error")
        return {}
        
    if "feedback" not in st.session_state.uploaded_files or not st.session_state.uploaded_files["feedback"]:
        st.error("Feedback file is required")
        add_to_log("Feedback file is required", level="error")
        return {}
    
    # Get most recent files
    sales_file = st.session_state.uploaded_files["sales_history"][-1]["path"]
    inventory_file = st.session_state.uploaded_files["inventory"][-1]["path"]
    order_bank_file = st.session_state.uploaded_files["order_bank"][-1]["path"]
    feedback_file = st.session_state.uploaded_files["feedback"][-1]["path"]
    
    # Optional market file
    market_file = None
    if "market" in st.session_state.uploaded_files and st.session_state.uploaded_files["market"]:
        market_file = st.session_state.uploaded_files["market"][-1]["path"]
    
    add_to_log("Starting complete DOSO cycle...")
    with st.spinner("Running complete DOSO cycle..."):
        # Run the full cycle
        start_time = time.time()
        
        # In a production application, we would use the full cycle function
        # but for more control and visibility, we'll run each step separately
        
        # Step 1: Forecasting
        forecast_result = process_sales_history(sales_file)
        
        # Step 2: Recommendations (only if forecasting succeeded)
        recommendation_result = {}
        if forecast_result and "status" in forecast_result and forecast_result["status"] == "success":
            recommendation_result = generate_recommendations(
                inventory_path=inventory_file,
                order_bank_path=order_bank_file,
                forecast_data=forecast_result,
                market_path=market_file
            )
        
        # Step 3: Feedback Collection (only if recommendations succeeded)
        feedback_result = {}
        if recommendation_result and "status" in recommendation_result and recommendation_result["status"] == "success":
            feedback_result = collect_feedback(
                feedback_path=feedback_file,
                recommendation_data=recommendation_result
            )
        
        # Step 4: Learning (only if feedback collection succeeded)
        learning_result = {}
        if feedback_result and "status" in feedback_result and feedback_result["status"] == "success":
            learning_result = optimize_weights(feedback_result)
        
        # Combine results from all stages
        results = {
            "cycle_id": f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started_at": datetime.now().isoformat(),
            "dealer_id": st.session_state.dealer_id,
            "stages": {
                "forecasting": forecast_result,
                "recommendation": recommendation_result,
                "feedback": feedback_result,
                "learning": learning_result
            },
            "completed_at": datetime.now().isoformat(),
            "processing_time": time.time() - start_time
        }
        
        # Determine overall status
        all_statuses = [
            forecast_result.get("status", "error"),
            recommendation_result.get("status", "error"),
            feedback_result.get("status", "error"),
            learning_result.get("status", "error")
        ]
        
        if all(status == "success" for status in all_statuses):
            results["status"] = "success"
            add_to_log(f"Full DOSO cycle completed successfully in {results['processing_time']:.2f}s")
            st.success("Full DOSO cycle completed successfully!")
        elif any(status == "success" for status in all_statuses):
            results["status"] = "partial"
            add_to_log(f"Full DOSO cycle partially completed in {results['processing_time']:.2f}s", level="warning")
            st.warning("DOSO cycle partially completed - some steps were successful")
        else:
            results["status"] = "error"
            add_to_log(f"Full DOSO cycle failed after {results['processing_time']:.2f}s", level="error")
            st.error("DOSO cycle execution failed")
            
        # Store result in session state
        st.session_state.cycle_results = results
            
        # Update performance log
        update_performance_log(results)
            
        return results


# Visualization functions
def display_uploaded_files():
    """Display a table of uploaded files by type"""
    if not st.session_state.uploaded_files:
        st.info("No files have been uploaded yet.")
        return
        
    # Group files by type
    for file_type, files in st.session_state.uploaded_files.items():
        if files:
            st.subheader(f"{file_type.replace('_', ' ').title()} Files")
            
            # Create a table of files
            file_data = []
            for file in files:
                # Convert timestamp to readable format
                if "uploaded_at" in file:
                    try:
                        upload_time = datetime.fromisoformat(file["uploaded_at"])
                        timestamp = upload_time.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        timestamp = "Unknown"
                else:
                    timestamp = "Unknown"
                    
                file_data.append({
                    "File Name": file["file_name"],
                    "Path": file["path"],
                    "Uploaded At": timestamp
                })
            
            # Display as dataframe
            df = pd.DataFrame(file_data)
            st.dataframe(df, use_container_width=True)


def display_forecast_results(forecast_data: Dict):
    """Display forecast results with visualizations"""
    if not forecast_data or "forecasts" not in forecast_data:
        st.info("No forecast data available. Run the forecasting process first.")
        return
        
    forecasts = forecast_data.get("forecasts", {})
    
    # Extract forecast data for visualization
    configs = []
    sales_data = []
    lower_bounds = []
    upper_bounds = []
    months = []
    
    for config, forecast in forecasts.items():
        if "monthly_forecast" in forecast:
            for month, data in forecast["monthly_forecast"].items():
                configs.append(config)
                months.append(month)
                sales_data.append(data["projected_sales"])
                lower_bounds.append(data["lower_bound"])
                upper_bounds.append(data["upper_bound"])
    
    if not configs:
        st.info("No forecast data to display.")
        return
        
    # Create dataframe for plotting
    df = pd.DataFrame({
        "Configuration": configs,
        "Month": months,
        "Projected Sales": sales_data,
        "Lower Bound": lower_bounds,
        "Upper Bound": upper_bounds
    })
    
    # Display summary metrics
    st.subheader("Forecast Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Configurations", len(set(configs)))
        
    with col2:
        st.metric("Forecast Horizon", f"{len(set(months))} months")
        
    with col3:
        confidence = forecast_data.get("confidence_level", 0.9) * 100
        st.metric("Confidence Level", f"{confidence:.0f}%")
    
    # Plot forecasts by configuration
    st.subheader("Sales Forecasts by Configuration")
    
    fig = px.line(
        df, 
        x="Month", 
        y="Projected Sales",
        color="Configuration",
        line_group="Configuration",
        hover_data=["Lower Bound", "Upper Bound"],
        title="Monthly Sales Forecast by Configuration"
    )
    
    # Add confidence intervals
    for config in set(configs):
        config_df = df[df["Configuration"] == config]
        fig.add_traces(
            go.Scatter(
                x=config_df["Month"].tolist() + config_df["Month"].tolist()[::-1],
                y=config_df["Upper Bound"].tolist() + config_df["Lower Bound"].tolist()[::-1],
                fill="toself",
                fillcolor=f"rgba(128, 128, 128, 0.2)",
                line=dict(color="rgba(255, 255, 255, 0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{config} Confidence Interval"
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast data in table format
    st.subheader("Detailed Forecast Data")
    st.dataframe(df, use_container_width=True)


def display_recommendations(recommendation_data: Dict):
    """Display recommendation results with visualizations"""
    if not recommendation_data or "recommendations" not in recommendation_data:
        st.info("No recommendation data available. Run the recommendation process first.")
        return
        
    recommendations = recommendation_data.get("recommendations", [])
    
    if not recommendations:
        st.info("No recommendations to display.")
        return
    
    # Extract data for visualization
    configs = []
    scores = []
    justifications = []
    
    for rec in recommendations:
        configs.append(rec.get("configuration", "Unknown"))
        scores.append(rec.get("doso_score", 0))
        justifications.append(rec.get("justification", ""))
    
    # Create dataframe for display
    df = pd.DataFrame({
        "Configuration": configs,
        "DOSO Score": scores,
        "Justification": justifications
    })
    
    # Sort by score descending
    df = df.sort_values(by="DOSO Score", ascending=False)
    
    # Display summary metrics
    st.subheader("Recommendation Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Recommendations", len(recommendations))
        
    with col2:
        if scores:
            st.metric("Average DOSO Score", f"{sum(scores)/len(scores):.2f}")
        
    with col3:
        if scores:
            st.metric("Highest DOSO Score", f"{max(scores):.2f}")
    
    # Plot recommendations by score
    st.subheader("Recommended Configurations by DOSO Score")
    
    fig = px.bar(
        df,
        x="Configuration",
        y="DOSO Score",
        color="DOSO Score",
        color_continuous_scale="viridis",
        title="Configuration DOSO Scores"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display weights used for scoring
    if "weights" in recommendation_data:
        weights = recommendation_data["weights"]
        st.subheader("Scoring Weights")
        
        weights_df = pd.DataFrame({
            "Factor": list(weights.keys()),
            "Weight": list(weights.values())
        })
        
        fig = px.pie(
            weights_df,
            names="Factor",
            values="Weight",
            title="DOSO Score Weighting Factors"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display recommendation details in table
    st.subheader("Detailed Recommendations")
    st.dataframe(df, use_container_width=True)
    
    # Display top recommendation details
    if recommendations:
        st.subheader("Top Recommendation Details")
        top_rec = recommendations[0]
        
        st.markdown(f"**Configuration:** {top_rec.get('configuration', 'Unknown')}")
        st.markdown(f"**DOSO Score:** {top_rec.get('doso_score', 0):.2f}")
        st.markdown(f"**Justification:** {top_rec.get('justification', '')}")
        
        # Display detailed factors if available
        if "score_factors" in top_rec:
            factors = top_rec["score_factors"]
            
            factors_df = pd.DataFrame({
                "Factor": list(factors.keys()),
                "Value": list(factors.values())
            })
            
            st.subheader("Score Factors")
            st.dataframe(factors_df, use_container_width=True)


def display_learning_results(learning_data: Dict):
    """Display learning results with visualizations"""
    if not learning_data or "optimized_weights" not in learning_data:
        st.info("No learning data available. Run the learning process first.")
        return
    
    # Get current and optimized weights
    current_weights = learning_data.get("current_weights", {})
    optimized_weights = learning_data.get("optimized_weights", {})
    
    if not current_weights or not optimized_weights:
        st.info("Weight data is incomplete.")
        return
    
    # Create comparison dataframe
    factors = list(current_weights.keys())
    current_values = [current_weights[f] for f in factors]
    optimized_values = [optimized_weights[f] for f in factors]
    
    df = pd.DataFrame({
        "Factor": factors,
        "Before Learning": current_values,
        "After Learning": optimized_values,
        "Change": [optimized_values[i] - current_values[i] for i in range(len(factors))]
    })
    
    # Display metrics
    st.subheader("Learning Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        training_samples = learning_data.get("training_samples", 0)
        st.metric("Training Samples", training_samples)
        
    with col2:
        learning_rate = learning_data.get("learning_rate", 0)
        st.metric("Learning Rate", f"{learning_rate:.2f}")
        
    with col3:
        iterations = learning_data.get("iterations", 0)
        st.metric("Iterations", iterations)
    
    # Plot weight comparison
    st.subheader("Weight Optimization Results")
    
    # Melt dataframe for bar chart comparison
    melted_df = pd.melt(
        df,
        id_vars=["Factor"],
        value_vars=["Before Learning", "After Learning"],
        var_name="Stage",
        value_name="Weight"
    )
    
    fig = px.bar(
        melted_df,
        x="Factor",
        y="Weight",
        color="Stage",
        barmode="group",
        title="Weight Comparison Before and After Learning"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display weight changes
    st.subheader("Weight Changes")
    
    change_df = df[["Factor", "Change"]].copy()
    
    fig = px.bar(
        change_df,
        x="Factor",
        y="Change",
        color="Change",
        color_continuous_scale="RdBu",
        title="Weight Changes from Learning Process"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display performance metrics if available
    if "performance_metrics" in learning_data:
        metrics = learning_data["performance_metrics"]
        
        st.subheader("Performance Metrics")
        
        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })
        
        st.dataframe(metrics_df, use_container_width=True)
    
    # Display weight data in table
    st.subheader("Detailed Weight Data")
    st.dataframe(df, use_container_width=True)


def display_feedback_analysis(feedback_data: Dict):
    """Display feedback analysis with visualizations"""
    if not feedback_data or "feedback_summary" not in feedback_data:
        st.info("No feedback data available. Run the feedback collection process first
