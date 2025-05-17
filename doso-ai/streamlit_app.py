"""
DOSO AI Streamlit Interface

This Streamlit application provides a user interface for the Dealer Inventory Optimization
System AI, allowing dealerships to upload data, run analysis, and view recommendations.
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame
from streamlit.runtime.uploaded_file_manager import UploadedFile

from doso_ai.src.agents.triage_agent import TriageAgent
from doso_ai.src.db.redis import cache_key, get_cached_data, set_cached_data
from doso_ai.src.db.vector_store import VectorStore
from doso_ai.src.schemas import DealerRequest, WorkflowType
from doso_ai.src.utils.telemetry import log_info
from doso_ai.src.workflow.orchestration import orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doso-ai-streamlit")

# Initialize vector store for semantic search
vector_store = VectorStore()

# Set page configuration
st.set_page_config(
    page_title="DOSO AI - Dealer Inventory Optimization System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .recommendation {
        padding: 0.5rem;
        border-left: 3px solid #1E3A8A;
        background-color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4b5563;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .info-text {
        color: #4b5563;
        font-size: 0.9rem;
    }
    .status-running {
        color: #2563eb;
        font-weight: bold;
    }
    .status-success {
        color: #059669;
        font-weight: bold;
    }
    .status-error {
        color: #dc2626;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6b7280;
        font-size: 0.8rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'dealer_id' not in st.session_state:
    st.session_state.dealer_id = None
if 'request_id' not in st.session_state:
    st.session_state.request_id = None
if 'workflow_results' not in st.session_state:
    st.session_state.workflow_results = None
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = {}
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []


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


async def parse_uploaded_file(file: UploadedFile, file_type: str) -> Dict[str, Any]:
    """
    Parse uploaded dealer data file
    
    Args:
        file: Uploaded file object
        file_type: Type of data file (inventory, sales, etc.)
        
    Returns:
        Parsed data dictionary
    """
    add_to_log(f"Processing {file_type} file: {file.name}")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Parse based on file type
        if file_type == "inventory":
            df = pd.read_csv(tmp_path)
            add_to_log(f"Inventory data loaded: {len(df)} records")
            
            # Process data
            parsed_data = {
                "total_units": len(df),
                "total_value": df["value"].sum() if "value" in df.columns else 0,
                "models": df["model"].value_counts().to_dict() if "model" in df.columns else {},
                "aging": {
                    "0-30": int(df[df["days_on_lot"] <= 30].shape[0]) if "days_on_lot" in df.columns else 0,
                    "31-60": int(df[(df["days_on_lot"] > 30) & (df["days_on_lot"] <= 60)].shape[0]) if "days_on_lot" in df.columns else 0,
                    "61-90": int(df[(df["days_on_lot"] > 60) & (df["days_on_lot"] <= 90)].shape[0]) if "days_on_lot" in df.columns else 0,
                    "90+": int(df[df["days_on_lot"] > 90].shape[0]) if "days_on_lot" in df.columns else 0,
                }
            }
            
            # Store dataframe in session for visualization
            st.session_state.parsed_data[file_type] = df
            
        elif file_type == "sales":
            df = pd.read_csv(tmp_path)
            add_to_log(f"Sales data loaded: {len(df)} records")
            
            # Process data
            parsed_data = {
                "total_sales": len(df),
                "total_revenue": df["price"].sum() if "price" in df.columns else 0,
                "models": df["model"].value_counts().to_dict() if "model" in df.columns else {},
                "monthly_sales": df.groupby(df["date"].str[:7]).size().to_dict() if "date" in df.columns else {}
            }
            
            # Store dataframe in session for visualization
            st.session_state.parsed_data[file_type] = df
            
        elif file_type == "market":
            df = pd.read_csv(tmp_path)
            add_to_log(f"Market data loaded: {len(df)} records")
            
            # Process data
            parsed_data = {
                "market_share": df["market_share"].mean() if "market_share" in df.columns else 0,
                "competitor_data": {
                    comp: df[df["competitor"] == comp]["market_share"].mean() 
                    for comp in df["competitor"].unique() 
                } if "competitor" in df.columns and "market_share" in df.columns else {},
                "trends": {
                    model: df[df["model"] == model]["trend"].mean() 
                    for model in df["model"].unique()
                } if "model" in df.columns and "trend" in df.columns else {}
            }
            
            # Store dataframe in session for visualization
            st.session_state.parsed_data[file_type] = df
            
        else:
            # Handle other file types
            with open(tmp_path, 'r') as f:
                content = f.read()
            
            # Basic parsing for text files
            parsed_data = {
                "content": content,
                "lines": len(content.split('\n')),
                "type": file_type
            }
        
        add_to_log(f"Successfully parsed {file_type} data")
        return parsed_data
    
    except Exception as e:
        add_to_log(f"Error parsing {file_type} file: {str(e)}", level="error")
        return {"error": str(e)}
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def run_agent_workflow(data: Dict[str, Any], request_type: str, dealer_id: str) -> Dict[str, Any]:
    """
    Run agent workflow for the given data
    
    Args:
        data: Processed data to analyze
        request_type: Type of analysis to perform
        dealer_id: Dealer identifier
        
    Returns:
        Analysis results and recommendations
    """
    # Create unique request ID
    request_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    st.session_state.request_id = request_id
    
    add_to_log(f"Starting workflow for request: {request_id}")
    
    # Create dealer request
    dealer_request = DealerRequest(
        request_id=request_id,
        dealer_id=dealer_id,
        request_type=request_type,
        data=data,
        timestamp=datetime.now(),
    )
    
    try:
        # Run workflow through orchestrator
        result = await orchestrator.process_request(request=dealer_request)
        
        # Store result in session state
        st.session_state.workflow_results = result
        
        # Add to history
        st.session_state.history.append({
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "dealer_id": dealer_id,
            "request_type": request_type,
            "result": result
        })
        
        add_to_log("Workflow completed successfully", level="info")
        return result
        
    except Exception as e:
        add_to_log(f"Error in workflow execution: {str(e)}", level="error")
        return {"error": str(e)}


async def search_similar_analyses(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar previous analyses
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of similar analyses
    """
    try:
        results = await vector_store.search_similar(
            query=query,
            limit=limit,
            metadata_filter={"dealer_id": st.session_state.dealer_id} if st.session_state.dealer_id else None
        )
        
        add_to_log(f"Found {len(results)} similar analyses")
        return results
        
    except Exception as e:
        add_to_log(f"Error searching similar analyses: {str(e)}", level="error")
        return []


def render_sidebar() -> None:
    """Render the sidebar with dealer login and history"""
    with st.sidebar:
        st.image("https://www.ford.com/content/dam/brand_ford/en_us/brand/logo/ford-logo-3d-extruded-1920x1080.png", width=120)
        st.header("DOSO AI")
        
        # Dealer login
        dealer_id = st.text_input("Dealer ID", value=st.session_state.dealer_id or "", 
                                help="Enter your Ford dealer ID")
                                
        if dealer_id:
            if st.session_state.dealer_id != dealer_id:
                st.session_state.dealer_id = dealer_id
                add_to_log(f"Logged in as dealer: {dealer_id}")
        
        st.markdown("---")
        
        # Processing log
        st.subheader("Processing Log")
        log_container = st.container(height=300)
        
        with log_container:
            for log in reversed(st.session_state.processing_log):
                level_color = {
                    "info": "blue",
                    "warning": "orange",
                    "error": "red"
                }.get(log["level"], "black")
                
                st.markdown(f"<small>{log['timestamp']} - <span style='color: {level_color};'>{log['message']}</span></small>", 
                          unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analysis history
        if st.session_state.history:
            st.subheader("Recent Analyses")
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                with st.expander(f"{item['request_type']} - {item['timestamp'][:10]}"):
                    st.write(f"Request ID: {item['request_id']}")
                    st.write(f"Type: {item['request_type']}")
                    
                    if st.button("Load Results", key=f"load_{i}"):
                        st.session_state.workflow_results = item["result"]
                        add_to_log(f"Loaded previous analysis: {item['request_id']}")
                        st.rerun()
        
        st.markdown("---")
        
        # About section
        with st.expander("About DOSO AI"):
            st.write("""
            **Dealer Inventory Optimization System AI**
            
            DOSO AI helps Ford dealerships optimize inventory, track allocations, analyze market trends, 
            and plan orders effectively using AI-powered recommendations.
            
            Version: 0.2.0
            """)
            
        st.markdown("<div class='footer'>© 2025 Ford Motor Company</div>", unsafe_allow_html=True)


def render_upload_section() -> None:
    """Render the data upload section"""
    st.markdown("<h1 class='main-header'>Dealer Inventory Optimization System</h1>", unsafe_allow_html=True)
    
    if not st.session_state.dealer_id:
        st.warning("Please enter your Dealer ID in the sidebar to continue.")
        return
    
    st.markdown("<h2 class='sub-header'>Data Upload</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Upload your dealership data files to start the analysis process.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("<p class='step-header'>Step 1: Inventory Data</p>", unsafe_allow_html=True)
            inventory_file = st.file_uploader(
                "Upload current inventory CSV", 
                type=["csv"], 
                key="inventory_upload",
                help="CSV with columns: vin, model, trim, color, msrp, value, days_on_lot"
            )
    
    with col2:
        with st.container(border=True):
            st.markdown("<p class='step-header'>Step 2: Sales Data</p>", unsafe_allow_html=True)
            sales_file = st.file_uploader(
                "Upload sales history CSV", 
                type=["csv"], 
                key="sales_upload",
                help="CSV with columns: date, vin, model, trim, price, customer_zip"
            )
    
    with col3:
        with st.container(border=True):
            st.markdown("<p class='step-header'>Step 3: Market Data</p>", unsafe_allow_html=True)
            market_file = st.file_uploader(
                "Upload market data CSV", 
                type=["csv"], 
                key="market_upload",
                help="CSV with columns: model, market_share, competitor, trend"
            )
    
    # Optional allocation data
    with st.expander("Additional Data (Optional)"):
        col1, col2 = st.columns(2)
        
        with col1:
            allocation_file = st.file_uploader(
                "Allocation Data", 
                type=["csv", "txt"], 
                key="allocation_upload",
                help="Current allocation data from Ford"
            )
        
        with col2:
            constraints_file = st.file_uploader(
                "Order Constraints", 
                type=["csv", "txt"], 
                key="constraints_upload",
                help="Current ordering constraints"
            )
    
    # Analysis type selection
    st.markdown("<h2 class='sub-header'>Analysis Type</h2>", unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "Select the type of analysis to perform",
        options=[
            "Inventory Optimization",
            "Order Planning",
            "Market Analysis",
            "Gap Analysis",
            "Allocation Tracking",
            "Sales Velocity Analysis",
            "Constraint Check"
        ],
        index=0,
        help="Choose the analysis type that best fits your current needs"
    )
    
    # Workflow mapping
    workflow_mapping = {
        "Inventory Optimization": WorkflowType.INVENTORY_OPTIMIZATION,
        "Order Planning": WorkflowType.ORDER_PLANNING,
        "Market Analysis": WorkflowType.MARKET_ANALYSIS,
        "Gap Analysis": WorkflowType.GAP_ANALYSIS,
        "Allocation Tracking": WorkflowType.ALLOCATION_TRACKING,
        "Constraint Check": WorkflowType.CONSTRAINT_CHECK,
        "Sales Velocity Analysis": WorkflowType.GENERAL_ANALYSIS,
    }
    
    # Submit button
    if st.button("Run Analysis", type="primary", disabled=not st.session_state.dealer_id):
        with st.spinner("Processing data and running analysis..."):
            asyncio.run(_process_uploaded_files(
                inventory_file=inventory_file,
                sales_file=sales_file,
                market_file=market_file,
                allocation_file=allocation_file,
                constraints_file=constraints_file,
                analysis_type=analysis_type,
                workflow_type=workflow_mapping.get(analysis_type, WorkflowType.GENERAL_ANALYSIS)
            ))
        
        # Force app to rerun to show results
        st.rerun()


async def _process_uploaded_files(
    inventory_file: Optional[UploadedFile],
    sales_file: Optional[UploadedFile],
    market_file: Optional[UploadedFile],
    allocation_file: Optional[UploadedFile],
    constraints_file: Optional[UploadedFile],
    analysis_type: str,
    workflow_type: WorkflowType
) -> None:
    """
    Process uploaded files and run workflow
    
    Args:
        inventory_file: Inventory data file
        sales_file: Sales history file
        market_file: Market data file
        allocation_file: Allocation data file
        constraints_file: Constraints file
        analysis_type: Type of analysis
        workflow_type: Workflow type enum value
    """
    add_to_log(f"Starting {analysis_type} analysis")
    combined_data = {"workflow_type": workflow_type.value}
    
    # Process inventory file
    if inventory_file:
        inventory_data = await parse_uploaded_file(inventory_file, "inventory")
        combined_data["inventory"] = inventory_data
    
    # Process sales file
    if sales_file:
        sales_data = await parse_uploaded_file(sales_file, "sales")
        combined_data["sales"] = sales_data
    
    # Process market file
    if market_file:
        market_data = await parse_uploaded_file(market_file, "market")
        combined_data["market"] = market_data
        
    # Process allocation file
    if allocation_file:
        allocation_data = await parse_uploaded_file(allocation_file, "allocation")
        combined_data["allocation"] = allocation_data
    
    # Process constraints file
    if constraints_file:
        constraints_data = await parse_uploaded_file(constraints_file, "constraints")
        combined_data["constraints"] = constraints_data
    
    # Run analysis workflow
    add_to_log(f"Running {analysis_type} workflow")
    result = await run_agent_workflow(
        data=combined_data,
        request_type=analysis_type,
        dealer_id=st.session_state.dealer_id
    )


def render_results_section() -> None:
    """Render the analysis results section"""
    if not st.session_state.workflow_results:
        return
    
    results = st.session_state.workflow_results
    
    st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
    
    # Display execution stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-value'>{results.workflow_type.value.replace('_', ' ').title()}</div>
                <div class='metric-label'>Analysis Type</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-value'>{len(results.recommendations)}</div>
                <div class='metric-label'>Recommendations</div>
            </div>""",
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-value'>{results.execution_time:.2f}s</div>
                <div class='metric-label'>Processing Time</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    # Key recommendations
    st.markdown("<h3 class='sub-header'>Key Recommendations</h3>", unsafe_allow_html=True)
    
    for i, rec in enumerate(results.recommendations):
        st.markdown(
            f"""<div class='recommendation'>
                <strong>#{i+1}:</strong> {rec}
            </div>""",
            unsafe_allow_html=True
        )
    
    # Next steps
    if results.next_steps:
        st.markdown("<h3 class='sub-header'>Suggested Next Steps</h3>", unsafe_allow_html=True)
        
        for i, step in enumerate(results.next_steps):
            st.markdown(
                f"""<div class='card'>
                    <strong>Step {i+1}:</strong> {step}
                </div>""",
                unsafe_allow_html=True
            )
    
    # Detailed results tabs
    st.markdown("<h3 class='sub-header'>Detailed Analysis</h3>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Inventory Analysis", "Market Insights", "Gap Analysis", "Visualizations"])
    
    with tabs[0]:
        _render_inventory_tab(results)
    
    with tabs[1]:
        _render_market_tab(results)
    
    with tabs[2]:
        _render_gap_tab(results)
    
    with tabs[3]:
        _render_visualizations_tab(results)


def _render_inventory_tab(results: Any) -> None:
    """Render inventory analysis tab"""
    if "inventory_analysis" in results.results:
        inventory_results = results.results["inventory_analysis"]
        
        if isinstance(inventory_results, dict) and "metrics" in inventory_results:
            metrics = inventory_results["metrics"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Units", metrics.get("total_units", 0))
            
            with col2:
                st.metric("Avg Days Supply", f"{metrics.get('average_days_supply', 0):.1f}")
            
            with col3:
                st.metric("Turnover Rate", f"{metrics.get('turnover_rate', 0):.2f}x")
            
            with col4:
                st.metric("Total Value", f"${metrics.get('total_value', 0):,.0f}")
            
            # Aging Distribution
            if "aging_distribution" in metrics:
                st.subheader("Aging Distribution")
                
                aging = metrics["aging_distribution"]
                
                # Create aging chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=list(aging.keys()),
                    y=list(aging.values()),
                    marker_color=['#22c55e', '#facc15', '#f97316', '#ef4444'],
                    text=list(aging.values()),
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Vehicle Aging (Days on Lot)",
                    xaxis_title="Days on Lot",
                    yaxis_title="Number of Vehicles",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display insights
            if "insights" in inventory_results:
                st.subheader("Inventory Insights")
                for insight in inventory_results["insights"]:
                    st.info(insight)
            
            # Risk factors
            if "risk_factors" in inventory_results:
                st.subheader("Risk Factors")
                for risk in inventory_results["risk_factors"]:
                    st.warning(risk)
    else:
        st.info("No inventory analysis data available.")


def _render_market_tab(results: Any) -> None:
    """Render market analysis tab"""
    if "market_analysis" in results.results:
        market_results = results.results["market_analysis"]
        
        if isinstance(market_results, dict):
            # Market trends
            if "trends" in market_results:
                st.subheader("Market Trends")
                
                trends = market_results["trends"]
                if isinstance(trends, dict):
                    trends_df = pd.DataFrame(trends.items(), columns=["Model", "Trend Score"])
                    trends_df["Trend Direction"] = trends_df["Trend Score"].apply(
                        lambda x: "Increasing" if x > 0 else "Decreasing" if x < 0 else "Stable"
                    )
                    
                    # Show table
                    st.dataframe(trends_df, use_container_width=True, hide_index=True)
                    
                    # Create trend chart
                    fig = px.bar(
                        trends_df,
                        x="Model", 
                        y="Trend Score",
                        color="Trend Direction",
                        color_discrete_map={
                            "Increasing": "#22c55e",
                            "Stable": "#facc15",
                            "Decreasing": "#ef4444"
                        },
                        height=300
                    )
                    
                    fig.update_layout(
                        title="Market Trend by Model",
                        xaxis_title="Model",
                        yaxis_title="Trend Score",
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Market share
            if "market_share" in market_results:
                st.subheader("Market Share Analysis")
                
                market_share = market_results["market_share"]
                if isinstance(market_share, dict):
                    share_df = pd.DataFrame(market_share.items(), columns=["Segment", "Share"])
                    
                    # Create pie chart
                    fig = px.pie(
                        share_df,
                        values="Share",
                        names="Segment",
                        height=300
                    )
                    
                    fig.update_layout(
                        title="Market Share by Segment",
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on market
            if "recommendations" in market_results:
                st.subheader("Market-Based Recommendations")
                for rec in market_results["recommendations"]:
                    st.info(rec)
    else:
        st.info("No market analysis data available.")


def _render_gap_tab(results: Any) -> None:
    """Render gap analysis tab"""
    if "gap_analysis" in results.results:
        gap_results = results.results["gap_analysis"]
        
        if isinstance(gap_results, dict):
            # Inventory gaps
            if "inventory_gaps" in gap_results:
                st.subheader("Inventory Gaps")
                
                gaps = gap_results["inventory_gaps"]
                if isinstance(gaps, dict):
                    gaps_df = pd.DataFrame([
                        {"Model": model, "Current": data.get("current", 0), 
                         "Optimal": data.get("optimal", 0), 
                         "Gap": data.get("optimal", 0) - data.get("current", 0)}
                        for model, data in gaps.items()
                    ])
                    
                    # Show table
                    st.dataframe(gaps_df, use_container_width=True, hide_index=True)
                    
                    # Create gap chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=gaps_df["Model"],
                        y=gaps_df["Current"],
                        name="Current Inventory",
                        marker_color="#3b82f6"
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=gaps_df["Model"],
                        y=gaps_df["Optimal"],
                        name="Optimal Inventory",
                        marker_color="#22c55e"
                    ))
                    
                    fig.update_layout(
                        title="Current vs. Optimal Inventory",
                        xaxis_title="Model",
                        yaxis_title="Units",
                        barmode="group",
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Opportunities
            if "opportunities" in gap_results:
                st.subheader("Identified Opportunities")
                
                for i, opportunity in enumerate(gap_results["opportunities"]):
                    st.success(f"Opportunity {i+1}: {opportunity}")
            
            # Recommendations
            if "recommendations" in gap_results:
                st.subheader("Gap-Closing Recommendations")
                
                for i, rec in enumerate(gap_results["recommendations"]):
                    st.info(rec)
    else:
        st.info("No gap analysis data available.")


def _render_visualizations_tab(results: Any) -> None:
    """Render visualizations tab"""
    # Check if we have parsed dataframes
    has_inventory = "inventory" in st.session_state.parsed_data
    has_sales = "sales" in st.session_state.parsed_data
    has_market = "market" in st.session_state.parsed_data
    
    if not (has_inventory or has_sales or has_market):
        st.info("No data available for visualization.")
        return
    
    # Visualization selection
    viz_type = st.selectbox(
        "Select Visualization",
        options=[
            "Inventory Mix",
            "Sales Trends",
            "Market Comparison",
            "Aging Distribution",
            "Turnover Analysis",
            "Profit Analysis"
        ]
    )
    
    if viz_type == "Inventory Mix" and has_inventory:
        _render_inventory_mix_viz()
    elif viz_type == "Sales Trends" and has_sales:
        _render_sales_trends_viz()
    elif viz_type == "Market Comparison" and has_market:
        _render_market_comparison_viz()
    elif viz_type == "Aging Distribution" and has_inventory:
        _render_aging_viz()
    elif viz_type == "Turnover Analysis" and (has_inventory and has_sales):
        _render_turnover_viz()
    elif viz_type == "Profit Analysis" and has_sales:
        _render_profit_viz()
    else:
        st.warning("Required data for this visualization is not available.")


def _render_inventory_mix_viz() -> None:
    """Render inventory mix visualization"""
    df = st.session_state.parsed_data["inventory"]
    
    # Get the dimensions to analyze
    dimension = st.radio(
        "Group By",
        options=["model", "color", "trim"],
        horizontal=True
    )
    
    if dimension in df.columns:
        # Create grouped dataframe
        grouped = df[dimension].value_counts().reset_index()
        grouped.columns = [dimension, "count"]
        
        # Create bar chart
        fig = px.bar(
            grouped, 
            x=dimension, 
            y="count",
            title=f"Inventory Mix by {dimension.title()}",
            labels={"count": "Number of Vehicles", dimension: dimension.title()},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.dataframe(grouped, hide_index=True, use_container_width=True)
    else:
        st.warning(f"Column '{dimension}' not found in inventory data.")


def _render_sales_trends_viz() -> None:
    """Render sales trends visualization"""
    df = st.session_state.parsed_data["sales"]
    
    # Ensure we have a date column
    if "date" not in df.columns:
        st.warning("Sales data does not contain date information.")
        return
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            st.warning("Could not parse date column.")
            return
    
    # Group by month and model
    df["month"] = df["date"].dt.to_period("M").astype(str)
    
    if "model" in df.columns:
        # Sales by model and month
        pivot_df = df.pivot_table(
            index="month", 
            columns="model", 
            values="vin", 
            aggfunc="count", 
            fill_value=0
        ).reset_index()
        
        # Melt for plotting
        melted = pd.melt(
            pivot_df, 
            id_vars=["month"], 
            var_name="model", 
            value_name="sales"
        )
        
        # Create line chart
        fig = px.line(
            melted, 
            x="month", 
            y="sales", 
            color="model",
            markers=True,
            title="Monthly Sales by Model",
            labels={"month": "Month", "sales": "Units Sold", "model": "Model"},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Just total sales by month
        monthly_sales = df.groupby("month")["vin"].count().reset_index()
        monthly_sales.columns = ["month", "sales"]
        
        # Create bar chart
        fig = px.bar(
            monthly_sales, 
            x="month", 
            y="sales",
            title="Monthly Sales",
            labels={"month": "Month", "sales": "Units Sold"},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def _render_market_comparison_viz() -> None:
    """Render market comparison visualization"""
    df = st.session_state.parsed_data["market"]
    
    # Check for necessary columns
    if "market_share" not in df.columns or "competitor" not in df.columns:
        st.warning("Market data does not contain required market_share or competitor columns.")
        return
    
    # Create comparison chart
    fig = px.bar(
        df, 
        x="competitor", 
        y="market_share",
        color="competitor",
        title="Market Share by Competitor",
        labels={"competitor": "Competitor", "market_share": "Market Share (%)"},
        height=500
    )
    
    # Add horizontal line for average
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=df["market_share"].mean(),
        x1=len(df["competitor"].unique()) - 0.5,
        y1=df["market_share"].mean(),
        line=dict(color="red", width=2, dash="dash"),
        name="Average"
    )
    
    # Add annotation for the average line
    fig.add_annotation(
        x=len(df["competitor"].unique()) - 1,
        y=df["market_share"].mean(),
        text=f"Average: {df['market_share'].mean():.2f}%",
        showarrow=False,
        yshift=10
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # If we have trend data
    if "trend" in df.columns and "model" in df.columns:
        st.subheader("Market Trends by Model")
        
        trend_df = df.groupby("model")["trend"].mean().reset_index()
        
        # Create horizontal bar chart
        fig = px.bar(
            trend_df,
            y="model",
            x="trend",
            orientation="h",
            title="Market Trend Score by Model",
            labels={"model": "Model", "trend": "Trend Score"},
            height=400,
            color="trend",
            color_continuous_scale=["red", "yellow", "green"]
        )
        
        st.plotly_chart(fig, use_container_width=True)


def _render_aging_viz() -> None:
    """Render aging distribution visualization"""
    df = st.session_state.parsed_data["inventory"]
    
    if "days_on_lot" not in df.columns:
        st.warning("Inventory data does not contain days_on_lot information.")
        return
    
    # Create aging buckets
    df["age_bucket"] = pd.cut(
        df["days_on_lot"],
        bins=[0, 30, 60, 90, float("inf")],
        labels=["0-30 days", "31-60 days", "61-90 days", "90+ days"]
    )
    
    # Count by age bucket
    age_counts = df["age_bucket"].value_counts().reset_index()
    age_counts.columns = ["Age", "Count"]
    
    # Sort by age
    age_order = ["0-30 days", "31-60 days", "61-90 days", "90+ days"]
    age_counts["Age"] = pd.Categorical(age_counts["Age"], categories=age_order, ordered=True)
    age_counts = age_counts.sort_values("Age")
    
    # Create bar chart with custom colors
    fig = px.bar(
        age_counts, 
        x="Age", 
        y="Count",
        title="Inventory Aging Distribution",
        labels={"Age": "Days on Lot", "Count": "Number of Vehicles"},
        height=400,
        color="Age",
        color_discrete_map={
            "0-30 days": "#4ade80",
            "31-60 days": "#facc15",
            "61-90 days": "#f97316",
            "90+ days": "#ef4444"
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Age by model
    if "model" in df.columns:
        st.subheader("Aging by Model")
        
        # Get average days on lot by model
        model_age = df.groupby("model")["days_on_lot"].mean().reset_index()
        model_age.columns = ["Model", "Average Days on Lot"]
        model_age = model_age.sort_values("Average Days on Lot", ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            model_age,
            y="Model",
            x="Average Days on Lot",
            orientation="h",
            title="Average Age by Model",
            height=400,
            color="Average Days on Lot",
            color_continuous_scale=["green", "yellow", "red"]
        )
        
        st.plotly_chart(fig, use_container_width=True)


def _render_turnover_viz() -> None:
    """Render inventory turnover visualization"""
    inventory_df = st.session_state.parsed_data["inventory"]
    sales_df = st.session_state.parsed_data["sales"]
    
    # Check for model column in both dataframes
    if "model" not in inventory_df.columns or "model" not in sales_df.columns:
        st.warning("Both inventory and sales data must contain model information.")
        return
    
    # Get inventory counts by model
    inventory_counts = inventory_df["model"].value_counts().reset_index()
    inventory_counts.columns = ["model", "inventory_count"]
    
    # Get sales counts by model
    sales_counts = sales_df["model"].value_counts().reset_index()
    sales_counts.columns = ["model", "sales_count"]
    
    # Merge the two
    turnover_df = pd.merge(inventory_counts, sales_counts, on="model", how="outer").fillna(0)
    
    # Calculate turnover ratio (sales / inventory)
    turnover_df["turnover_ratio"] = turnover_df["sales_count"] / turnover_df["inventory_count"]
    turnover_df["turnover_ratio"] = turnover_df["turnover_ratio"].replace([float('inf')], 0)
    
    # Sort by turnover ratio
    turnover_df = turnover_df.sort_values("turnover_ratio", ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=turnover_df["model"],
        y=turnover_df["turnover_ratio"],
        name="Turnover Ratio",
        marker_color="#3b82f6"
    ))
    
    # Add a reference line for the ideal turnover ratio (example: 1.5)
    ideal_turnover = 1.5
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=ideal_turnover,
        x1=len(turnover_df) - 0.5,
        y1=ideal_turnover,
        line=dict(color="green", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=len(turnover_df) - 1,
        y=ideal_turnover,
        text=f"Ideal: {ideal_turnover}",
        showarrow=False,
        yshift=10
    )
    
    fig.update_layout(
        title="Inventory Turnover by Model",
        xaxis_title="Model",
        yaxis_title="Turnover Ratio (Sales/Inventory)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data table
    st.dataframe(
        turnover_df[["model", "inventory_count", "sales_count", "turnover_ratio"]],
        hide_index=True,
        use_container_width=True
    )


def _render_profit_viz() -> None:
    """Render profit analysis visualization"""
    sales_df = st.session_state.parsed_data["sales"]
    
    # Check for required columns
    if "price" not in sales_df.columns:
        st.warning("Sales data does not contain price information.")
        return
    
    # Summary statistics
    total_revenue = sales_df["price"].sum()
    avg_price = sales_df["price"].mean()
    min_price = sales_df["price"].min()
    max_price = sales_df["price"].max()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col2:
        st.metric("Average Price", f"${avg_price:,.0f}")
    
    with col3:
        st.metric("Minimum Price", f"${min_price:,.0f}")
    
    with col4:
        st.metric("Maximum Price", f"${max_price:,.0f}")
    
    # Price distribution histogram
    fig = px.histogram(
        sales_df,
        x="price",
        title="Sale Price Distribution",
        labels={"price": "Sale Price ($)"},
        height=400,
        nbins=20
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # If we have model data, show revenue by model
    if "model" in sales_df.columns:
        model_revenue = sales_df.groupby("model")["price"].agg(["sum", "mean", "count"]).reset_index()
        model_revenue.columns = ["Model", "Total Revenue", "Average Price", "Sales Count"]
        model_revenue = model_revenue.sort_values("Total Revenue", ascending=False)
        
        # Create bar chart
        fig = px.bar(
            model_revenue,
            x="Model",
            y="Total Revenue",
            title="Revenue by Model",
            labels={"Model": "Model", "Total Revenue": "Total Revenue ($)"},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.dataframe(model_revenue, hide_index=True, use_container_width=True)


def render_search_section() -> None:
    """Render the semantic search section for finding similar analyses"""
    st.markdown("<h2 class='sub-header'>Search Previous Analyses</h2>", unsafe_allow_html=True)
    
    query = st.text_input(
        "Search previous analyses and recommendations", 
        placeholder="e.g., F-150 inventory optimization, market share, aging vehicles"
    )
    
    if query and st.button("Search", key="search_button"):
        with st.spinner("Searching similar analyses..."):
            results = asyncio.run(search_similar_analyses(query))
            st.session_state.search_results = results
    
    # Display search results
    if st.session_state.search_results:
        st.subheader(f"Found {len(st.session_state.search_results)} Similar Analyses")
        
        for i, result in enumerate(st.session_state.search_results):
            with st.expander(f"Result {i+1} - Similarity: {result['similarity']:.2f}"):
                st.write(f"Request ID: {result['id']}")
                st.write(f"Dealer ID: {result['metadata'].get('dealer_id', 'N/A')}")
                st.write(f"Request Type: {result['metadata'].get('request_type', 'N/A')}")
                st.write(f"Timestamp: {result['metadata'].get('timestamp', 'N/A')}")
                
                st.subheader("Recommendations")
                for rec in result['content'].get('recommendations', []):
                    st.info(rec)


def render_actions_section() -> None:
    """Render the actions section for executing next steps"""
    if not st.session_state.workflow_results:
        return
    
    st.markdown("<h2 class='sub-header'>Actions</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Take action on the recommendations provided by DOSO AI.</p>", unsafe_allow_html=True)
    
    # Create different action buttons based on workflow type
    workflow_type = st.session_state.workflow_results.workflow_type
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "Download Analysis Report",
            data=json.dumps(st.session_state.workflow_results.dict(), default=str, indent=2),
            file_name=f"doso_analysis_{st.session_state.request_id}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )
    
    with col2:
        if workflow_type == WorkflowType.INVENTORY_OPTIMIZATION:
            if st.button("Generate Order Recommendations", key="order_rec_button"):
                with st.spinner("Generating order recommendations..."):
                    # This would typically call a function, but we'll simulate it
                    time.sleep(2)
                    st.success("Order recommendations generated! Check the Recommendations tab.")
        elif workflow_type == WorkflowType.ORDER_PLANNING:
            if st.button("Submit Orders to Ford", key="submit_orders_button"):
                with st.spinner("Preparing order submission..."):
                    # This would typically call a function, but we'll simulate it
                    time.sleep(2)
                    st.success("Orders prepared for submission. Review in the Orders tab.")
    
    with col3:
        if st.button("Schedule Dealership Consultation", key="schedule_consult_button"):
            with st.spinner("Checking available appointments..."):
                # This would typically call a scheduling API, but we'll simulate it
                time.sleep(1)
                st.success("Consultation request submitted. A Ford representative will contact you within 24 hours.")


def main() -> None:
    """Main application entry point"""
    # Render sidebar
    render_sidebar()
    
    # Create tabs for different sections
    tabs = st.tabs(["Data Upload & Analysis", "Results & Insights", "Search & History"])
    
    with tabs[0]:
        render_upload_section()
    
    with tabs[1]:
        render_results_section()
        render_actions_section()
    
    with tabs[2]:
        render_search_section()


if __name__ == "__main__":
    main()
