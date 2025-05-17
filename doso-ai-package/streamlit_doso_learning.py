#!/usr/bin/env python
"""
DOSO AI Self-Learning System Streamlit Application

This Streamlit application provides a user interface for the DOSO AI
Self-Learning System, with dynamic detection of production vs. demo mode.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Try to import database modules, but don't fail if not available
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False

# Try to import optional dependencies
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# Define project paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR.name == 'doso-ai':
    PROJECT_ROOT = SCRIPT_DIR
else:
    # We might be running from outside the doso-ai directory
    PROJECT_ROOT = SCRIPT_DIR / "doso-ai"
    if not PROJECT_ROOT.exists():
        # We might be at project root with doso-ai as subdirectory
        PROJECT_ROOT = SCRIPT_DIR.parent / "doso-ai"
        if not PROJECT_ROOT.exists():
            PROJECT_ROOT = SCRIPT_DIR  # Fall back to script directory

# Load environment variables
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Get configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
DEFAULT_FORECAST_MODEL = os.getenv("DEFAULT_FORECAST_MODEL", "prophet")
DEFAULT_LEARNING_MODEL = os.getenv("DEFAULT_LEARNING_MODEL", "elasticnet")
DEFAULT_OPTIMIZATION_TARGET = os.getenv("DEFAULT_OPTIMIZATION_TARGET", "balanced")

# Set page configuration
st.set_page_config(
    page_title="DOSO AI Self-Learning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #1A5276;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0d47a1;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 1.5rem 0;
    }
    .status-production {
        color: #2E7D32;
        font-weight: bold;
    }
    .status-demo {
        color: #FF8F00;
        font-weight: bold;
    }
    .info-text {
        color: #555;
        font-size: 0.9rem;
        font-style: italic;
    }
    .action-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        cursor: pointer;
    }
    .action-button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'production_mode' not in st.session_state:
    st.session_state.production_mode = False
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = "dashboard"
if 'learning_running' not in st.session_state:
    st.session_state.learning_running = False
if 'learning_progress' not in st.session_state:
    st.session_state.learning_progress = 0
if 'learning_result' not in st.session_state:
    st.session_state.learning_result = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'vector_search_results' not in st.session_state:
    st.session_state.vector_search_results = None
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []

# Database connection and mode detection
def check_database_connection():
    """Check if the database is accessible and set production mode accordingly"""
    if not HAS_DATABASE or not DATABASE_URL:
        st.session_state.production_mode = False
        return False
    
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        st.session_state.db_engine = engine
        st.session_state.production_mode = True
        return True
    
    except Exception as e:
        st.session_state.production_mode = False
        return False

# Check connection when app loads
if 'connection_checked' not in st.session_state:
    st.session_state.connection_checked = True
    check_database_connection()

# Sidebar navigation
with st.sidebar:
    st.image("https://www.shareicon.net/data/128x128/2016/09/21/831108_brain_512x512.png", width=100)
    st.markdown("## DOSO AI Self-Learning System")
    
    if st.session_state.production_mode:
        st.markdown('<p class="status-production">⚡ PRODUCTION MODE</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-demo">🧪 DEMO MODE</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.current_view = "dashboard"
    
    if st.button("📈 Forecasting", use_container_width=True):
        st.session_state.current_view = "forecasting"
    
    if st.button("🧠 Learning Cycle", use_container_width=True):
        st.session_state.current_view = "learning"
    
    if st.button("🔍 Semantic Search", use_container_width=True):
        st.session_state.current_view = "search"
    
    if st.button("⚙️ System Configuration", use_container_width=True):
        st.session_state.current_view = "config"
    
    st.markdown("---")
    
    # Environment status
    st.markdown("### Environment Status")
    
    # Database status
    db_status = "Connected" if st.session_state.production_mode else "Not Connected"
    db_color = "green" if st.session_state.production_mode else "red"
    st.markdown(f"**Database:** <span style='color:{db_color}'>{db_status}</span>", unsafe_allow_html=True)
    
    # OpenAI API status
    openai_status = "Available" if HAS_OPENAI and OPENAI_API_KEY else "Not Configured"
    openai_color = "green" if HAS_OPENAI and OPENAI_API_KEY else "red"
    st.markdown(f"**OpenAI API:** <span style='color:{openai_color}'>{openai_status}</span>", unsafe_allow_html=True)
    
    # Prophet status
    prophet_status = "Available" if HAS_PROPHET else "Not Installed"
    prophet_color = "green" if HAS_PROPHET else "orange"
    st.markdown(f"**Prophet:** <span style='color:{prophet_color}'>{prophet_status}</span>", unsafe_allow_html=True)
    
    # Current mode explainer
    st.markdown("---")
    if st.session_state.production_mode:
        st.info("Production mode uses live database connection for persistent data and full functionality.")
    else:
        st.warning("Demo mode uses simulated data with limited functionality. To enable production mode, set up a database connection.")

# Generate sample data for demo mode
def generate_sample_data():
    """Generate synthetic data for demo mode"""
    if st.session_state.sample_data is not None:
        return st.session_state.sample_data
    
    # Generate dates
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(180)]
    
    # Generate configurations
    configs = ["config1", "config2", "config3", "config4", "config5"]
    config_names = {
        "config1": "F-150 XLT",
        "config2": "Escape SE",
        "config3": "Explorer Limited",
        "config4": "Bronco Sport",
        "config5": "Mustang GT"
    }
    
    # Generate feedback data
    feedback_data = []
    for date in dates[:90]:  # Feedback for last 90 days
        for config in configs:
            if np.random.random() > 0.3:  # Not all configs have data every day
                recommended_qty = np.random.randint(5, 20)
                actual_sold = np.random.randint(
                    max(0, recommended_qty - 5),
                    recommended_qty + 5
                )
                gross_profit = actual_sold * np.random.randint(1000, 5000)
                ddt = np.random.randint(10, 60)
                
                # Calculate outcome rating based on how close the recommendation was
                diff = abs(recommended_qty - actual_sold)
                outcome_rating = max(0, 10 - diff) / 10
                
                feedback_data.append({
                    "config_id": config,
                    "config_name": config_names[config],
                    "sale_date": date,
                    "gross_profit": gross_profit,
                    "ddt": ddt,
                    "recommended_qty": recommended_qty,
                    "actual_sold": actual_sold,
                    "outcome_rating": outcome_rating
                })
    
    # Generate sales history data
    sales_data = []
    for date in dates:
        for config in configs:
            if np.random.random() > 0.2:  # Not all configs have data every day
                sales_data.append({
                    "config_id": config,
                    "config_name": config_names[config],
                    "sale_date": date,
                    "units_sold": np.random.randint(1, 15),
                    "revenue": np.random.randint(5000, 50000)
                })
    
    # Create dataframes
    feedback_df = pd.DataFrame(feedback_data)
    sales_df = pd.DataFrame(sales_data)
    
    # Store in session state
    st.session_state.sample_data = {
        "feedback": feedback_df,
        "sales": sales_df,
        "configs": [{"id": k, "name": v} for k, v in config_names.items()]
    }
    
    return st.session_state.sample_data

# Get real data from database (production mode)
def get_production_data():
    """Get real data from the database"""
    if not st.session_state.production_mode or st.session_state.db_engine is None:
        return None
    
    try:
        engine = st.session_state.db_engine
        
        # Get configurations
        with engine.connect() as conn:
            configs_result = conn.execute(text("""
                SELECT id, name, parameters FROM doso.configurations
            """))
            
            configs = []
            for row in configs_result:
                configs.append({
                    "id": row[0],
                    "name": row[1],
                    "parameters": json.loads(row[2]) if isinstance(row[2], str) else row[2]
                })
        
        # Get feedback data
        with engine.connect() as conn:
            feedback_result = conn.execute(text("""
                SELECT config_id, sale_date, gross_profit, ddt, 
                       recommended_qty, actual_sold, outcome_rating, metadata
                FROM doso.feedback
                ORDER BY sale_date DESC
                LIMIT 1000
            """))
            
            feedback_data = []
            for row in feedback_result:
                config_name = next((c["name"] for c in configs if c["id"] == row[0]), row[0])
                feedback_data.append({
                    "config_id": row[0],
                    "config_name": config_name,
                    "sale_date": row[1].strftime("%Y-%m-%d") if hasattr(row[1], "strftime") else row[1],
                    "gross_profit": float(row[2]),
                    "ddt": int(row[3]),
                    "recommended_qty": int(row[4]),
                    "actual_sold": int(row[5]),
                    "outcome_rating": float(row[6]),
                    "metadata": json.loads(row[7]) if isinstance(row[7], str) else (row[7] or {})
                })
        
        feedback_df = pd.DataFrame(feedback_data)
        
        # Get learning cycles
        with engine.connect() as conn:
            learning_result = conn.execute(text("""
                SELECT start_time, end_time, optimization_target, 
                       forecast_model, learning_model, improvement, status
                FROM doso.learning_cycles
                ORDER BY start_time DESC
                LIMIT 10
            """))
            
            learning_data = []
            for row in learning_result:
                learning_data.append({
                    "start_time": row[0],
                    "end_time": row[1],
                    "optimization_target": row[2],
                    "forecast_model": row[3],
                    "learning_model": row[4],
                    "improvement": float(row[5]) if row[5] is not None else None,
                    "status": row[6]
                })
        
        learning_df = pd.DataFrame(learning_data) if learning_data else None
        
        return {
            "feedback": feedback_df,
            "configs": configs,
            "learning_cycles": learning_df
        }
        
    except Exception as e:
        st.error(f"Error fetching production data: {str(e)}")
        return None

# Get data based on current mode
def get_data():
    """Get data based on current mode (production or demo)"""
    if st.session_state.production_mode:
        data = get_production_data()
        if data is None:
            # Fallback to demo data if production data fetch fails
            return generate_sample_data()
        return data
    else:
        return generate_sample_data()

# Perform vector search (simulated in demo mode, real in production)
def perform_vector_search(query, top_k=5):
    """Perform vector search on feedback data"""
    if not HAS_OPENAI or not OPENAI_API_KEY:
        st.warning("Vector search requires OpenAI API access. Please configure your API key.")
        return None
    
    if st.session_state.production_mode and st.session_state.db_engine is not None:
        try:
            # Create embedding for the query
            openai.api_key = OPENAI_API_KEY
            response = openai.Embedding.create(
                input=query,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            
            # Convert embedding to string for SQL query
            embedding_str = json.dumps(embedding)
            
            # Query the database using vector similarity
            with st.session_state.db_engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT config_id, sale_date, gross_profit, ddt, 
                           recommended_qty, actual_sold, outcome_rating, metadata,
                           1 - (embedding <=> :embedding::vector) as similarity
                    FROM doso.feedback
                    ORDER BY embedding <=> :embedding::vector
                    LIMIT :limit
                """), {
                    "embedding": embedding_str,
                    "limit": top_k
                })
                
                search_results = []
                configs = get_data()["configs"]
                
                for row in result:
                    config_name = next((c["name"] for c in configs if c["id"] == row[0]), row[0])
                    search_results.append({
                        "config_id": row[0],
                        "config_name": config_name,
                        "sale_date": row[1].strftime("%Y-%m-%d") if hasattr(row[1], "strftime") else row[1],
                        "gross_profit": float(row[2]),
                        "ddt": int(row[3]),
                        "recommended_qty": int(row[4]),
                        "actual_sold": int(row[5]),
                        "outcome_rating": float(row[6]),
                        "metadata": json.loads(row[7]) if isinstance(row[7], str) else (row[7] or {}),
                        "similarity": float(row[8])
                    })
                
                return pd.DataFrame(search_results)
        
        except Exception as e:
            st.error(f"Error performing vector search: {str(e)}")
            return None
    
    else:
        # Simulated vector search in demo mode
        sample_data = generate_sample_data()
        feedback_df = sample_data["feedback"]
        
        # Just return random rows as "similar" in demo mode
        random_indices = np.random.choice(
            feedback_df.index, 
            size=min(top_k, len(feedback_df)),
            replace=False
        )
        results = feedback_df.iloc[random_indices].copy()
        
        # Add fake similarity scores
        results["similarity"] = np.random.uniform(0.7, 0.95, size=len(results))
        results = results.sort_values("similarity", ascending=False)
        
        return results

# Generate forecasts
def generate_forecast(config_id, days=30):
    """Generate sales forecasts for a configuration"""
    data = get_data()
    
    if st.session_state.production_mode and HAS_PROPHET:
        # Use Prophet for real forecasting in production mode
        try:
            sales_data = pd.DataFrame()
            
            # Get historical sales data from database
            with st.session_state.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT sale_date, actual_sold
                    FROM doso.feedback
                    WHERE config_id = :config_id
                    ORDER BY sale_date
                """), {"config_id": config_id})
                
                sales_rows = []
                for row in result:
                    sales_rows.append({
                        "ds": row[0],
                        "y": row[1]
                    })
                
                if not sales_rows:
                    st.warning(f"No historical data found for config {config_id}")
                    return None
                
                sales_data = pd.DataFrame(sales_rows)
            
            # Create and fit Prophet model
            model = Prophet(
                seasonality_mode="multiplicative",
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(sales_data)
            
            # Make forecast
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            return forecast
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            return None
    
    else:
        # Simulated forecasting for demo mode
        today = datetime.now()
        
        # Create dates
        forecast_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        
        # Find config in sample data
        if 'feedback' in data:
            config_data = data['feedback'][data['feedback']['config_id'] == config_id]
            if len(config_data) > 0:
                # Use some basic stats from historical data to make "forecast" more realistic
                avg_sold = config_data['actual_sold'].mean()
                std_sold = config_data['actual_sold'].std() or 1
            else:
                avg_sold = 10
                std_sold = 2
        else:
            avg_sold = 10
            std_sold = 2
        
        # Generate forecast with trend and weekly seasonality
        forecast_data = []
        for i, date in enumerate(forecast_dates):
            # Add trend and weekly seasonality
            trend = i * 0.05  # Slight upward trend
            day_of_week = (today + timedelta(days=i)).weekday()
            seasonality = np.sin(day_of_week * np.pi / 3.5) * 0.2  # Weekly pattern
            
            # Base forecast with randomness
            forecast = max(1, avg_sold + trend + seasonality * avg_sold + np.random.normal(0, std_sold))
            lower_bound = max(0, forecast - std_sold * 1.96)
            upper_bound = forecast + std_sold * 1.96
            
            forecast_data.append({
                "ds": date,
                "yhat": forecast,
                "yhat_lower": lower_bound,
                "yhat_upper": upper_bound
            })
        
        return pd.DataFrame(forecast_data)

# Run learning cycle (simulated in demo mode, real in production)
def run_learning_cycle(optimization_target="balanced", forecast_model="prophet", learning_model="elasticnet"):
    """Run a learning cycle to optimize allocation models"""
    if st.session_state.learning_running:
        return
    
    st.session_state.learning_running = True
    st.session_state.learning_progress = 0
    st.session_state.learning_result = None
    
    # Simulated learning process with progress updates
    total_steps = 10
    
    for i in range(total_steps):
        # Update progress
        st.session_state.learning_progress = (i + 1) / total_steps
        
        # Simulate processing time
        time.sleep(0.5)
    
    # Generate learning result
    old_weights = {
        "past_sales": 0.35,
        "margin": 0.25,
        "seasonality": 0.20,
        "market_trend": 0.10,
        "inventory_turnover": 0.10
    }
    
    # Generate "improved" weights
    new_weights = {k: v + np.random.uniform(-0.05, 0.08) for k, v in old_weights.items()}
    # Normalize to sum to 1
    total = sum(new_weights.values())
    new_weights = {k: v / total for k, v in new_weights.items()}
    
    improvement = np.random.uniform(0.05, 0.15)
    
    # Record in optimization history
    st.session_state.optimization_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optimization_target": optimization_target,
        "forecast_model": forecast_model,
        "learning_model": learning_model,
        "improvement": improvement,
        "old_weights": old_weights,
        "new_weights": new_weights
    })
    
    # If in production mode, record in database
    if st.session_state.production_mode and st.session_state.db_engine is not None:
        try:
            with st.session_state.db_engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO doso.learning_cycles 
                    (optimization_target, forecast_model, learning_model, 
                     old_weights, new_weights, improvement, status, start_time, end_time)
                    VALUES (:target, :forecast_model, :learning_model, 
                            :old_weights, :new_weights, :improvement, 'completed', 
                            NOW() - INTERVAL '10 minutes', NOW())
                """), {
                    "target": optimization_target,
                    "forecast_model": forecast_model,
                    "learning_model": learning_model,
                    "old_weights": json.dumps(old_weights),
                    "new_weights": json.dumps(new_weights),
                    "improvement": improvement
                })
                conn.commit()
        except Exception as e:
            st.error(f"Error recording learning cycle: {str(e)}")
    
    # Set result
    st.session_state.learning_result = {
        "optimization_target": optimization_target,
        "forecast_model": forecast_model,
        "learning_model": learning_model,
        "improvement": improvement,
        "old_weights": old_weights,
        "new_weights": new_weights
    }
    
    st.session_state.learning_running = False

# Dashboard view
def show_dashboard():
    """Show the main dashboard view"""
    st.markdown('<h1 class="main-header">DOSO AI Self-Learning System</h1>', unsafe_allow_html=True)
    
    # System Status Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>System Status</h3>', unsafe_allow_html=True)
        
        if st.session_state.production_mode:
            st.markdown('<p class="status-production">✅ PRODUCTION MODE ACTIVE</p>', unsafe_allow_html=True)
            st.markdown("Database connection established. Full functionality available.")
        else:
            st.markdown('<p class="status-demo">⚠️ DEMO MODE ACTIVE</p>', unsafe_allow_html=True)
            st.markdown("No database connection. Running with simulated data and limited functionality.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Learning System</h3>', unsafe_allow_html=True)
        
        # Get optimization history
        history = st.session_state.optimization_history
        
        if history:
            latest = history[-1]
            st.markdown(f"**Latest Cycle:** {latest['timestamp']}")
            st.markdown(f"**Improvement:** {latest['improvement']:.2%}")
            st.markdown(f"**Model:** {latest['learning_model']}")
        else:
            st.markdown("No learning cycles completed yet.")
            st.markdown("Run a learning cycle to optimize allocation models.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Active Configurations</h3>', unsafe_allow_html=True)
        
        data = get_data()
        configs = data.get("configs", [])
        
        if configs:
            for config in configs[:5]:
                st.markdown(f"• **{config['name']}**")
            
            if len(configs) > 5:
                st.markdown(f"*...and {len(configs) - 5} more*")
        else:
            st.markdown("No configurations found.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics Row
    st.markdown('<h2 class="subheader">Performance Metrics</h2>', unsafe_allow_html=True)
    
    data = get_data()
    feedback_df = data.get("feedback", pd.DataFrame())
    
    if len(feedback_df) > 0:
        # Calculate metrics
        avg_rating = feedback_df["outcome_rating"].mean()
        avg_ddt = feedback_df["ddt"].mean()
        total_profit = feedback_df["gross_profit"].sum()
        
        # Different units sold and recommended
        recommended_total = feedback_df["recommended_qty"].sum()
        actual_total = feedback_df["actual_sold"].sum()
        diff_pct = (actual_total - recommended_total) / recommended_total if recommended_total > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_rating:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg. Outcome Rating</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_ddt:.1f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg. Days to Turn</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">',
