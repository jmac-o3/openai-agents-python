"""
Modified Streamlit demo for DOSO AI that bypasses database requirements

This simplified script allows you to view the UI without needing Redis and PostgreSQL
"""

import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import uuid
import os
import tempfile
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Any, Dict, List, Optional, Union

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
    """Add a message to the processing log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append({
        "timestamp": timestamp,
        "message": message,
        "level": level
    })


def parse_uploaded_file(file: UploadedFile, file_type: str) -> Dict[str, Any]:
    """Parse uploaded dealer data file"""
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
        
        # About section
        with st.expander("About DOSO AI"):
            st.write("""
            **Dealer Inventory Optimization System AI**
            
            DOSO AI helps Ford dealerships optimize inventory, track allocations, analyze market trends, 
            and plan orders effectively using AI-powered recommendations.
            
            Version: 0.2.0 (Demo Mode)
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
    
    # Submit button
    if st.button("Run Analysis", type="primary", disabled=not st.session_state.dealer_id):
        with st.spinner("Processing data and running analysis..."):
            # Show that this is a demo
            st.info("⚠️ This is running in demo mode without backend services. In a production environment, this would connect to Redis and PostgreSQL databases and run AI agent workflows.")
            time.sleep(2)
            
            # Process files if uploaded
            combined_data = {}
            
            if inventory_file:
                combined_data["inventory"] = parse_uploaded_file(inventory_file, "inventory")
            
            if sales_file:
                combined_data["sales"] = parse_uploaded_file(sales_file, "sales")
            
            if market_file:
                combined_data["market"] = parse_uploaded_file(market_file, "market")
                
            if allocation_file:
                combined_data["allocation"] = parse_uploaded_file(allocation_file, "allocation")
            
            if constraints_file:
                combined_data["constraints"] = parse_uploaded_file(constraints_file, "constraints")
                
            # Create simulated results
            st.session_state.request_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
            add_to_log(f"Demo mode: Simulating {analysis_type} workflow")
            
            # Here we'd load a mock result corresponding to the analysis type
            st.session_state.workflow_results = {
                "request_id": st.session_state.request_id,
                "dealer_id": st.session_state.dealer_id,
                "workflow_type": analysis_type,
                "execution_time": 3.5,
                "is_complete": True,
                "recommendations": [
                    "Consider reducing F-150 inventory by 15% to improve turn rate",
                    "Increase Escape hybrid models by 8 units based on growing demand",
                    "Aging units over 90 days should be prioritized for retail promotions",
                    "Adjust color mix to increase Velocity Blue and decrease Oxford White",
                    "Optimize trim level mix based on regional sales performance"
                ],
                "next_steps": [
                    "Review current allocation and adjust order mix",
                    "Implement targeted marketing for aging inventory",
                    "Schedule dealer consultation to review complete findings"
                ],
                "results": {
                    "inventory_analysis": {
                        "metrics": {
                            "total_units": 120,
                            "total_value": 4500000,
                            "average_days_supply": 75.5,
                            "turnover_rate": 2.3,
                            "aging_distribution": {
                                "0-30 days": 45,
                                "31-60 days": 35,
                                "61-90 days": 25,
                                "90+ days": 15
                            }
                        },
                        "insights": [
                            "Current inventory levels are 15% above optimal",
                            "F-150 models are showing slower turn rates than the segment average",
                            "SUV inventory mix aligns well with current market demand",
                            "Premium trims are selling 30% faster than base models"
                        ],
                        "risk_factors": [
                            "15 units are aging beyond 90 days, creating potential carrying cost concerns",
                            "Current color mix is misaligned with regional preferences"
                        ]
                    },
                    "market_analysis": {
                        "trends": {
                            "F-150": -0.2,
                            "Escape": 1.5,
                            "Explorer": 0.3,
                            "Bronco": 2.1,
                            "Mustang": -0.5
                        },
                        "market_share": {
                            "SUV": 45,
                            "Truck": 35,
                            "Crossover": 15,
                            "Sports": 5
                        },
                        "recommendations": [
                            "Increase focus on Bronco and Escape models showing positive market trends",
                            "Adjust F-150 and Mustang inventory levels to align with market trends"
                        ]
                    },
                    "gap_analysis": {
                        "inventory_gaps": {
                            "F-150": {"current": 40, "optimal": 34},
                            "Escape": {"current": 25, "optimal": 33},
                            "Explorer": {"current": 20, "optimal": 22},
                            "Bronco": {"current": 15, "optimal": 22},
                            "Mustang": {"current": 20, "optimal": 15}
                        },
                        "opportunities": [
                            "Increase Escape inventory to capitalize on market growth",
                            "Increase Bronco allocation to match growing demand",
                            "Reduce Mustang inventory to improve turn rate"
                        ],
                        "recommendations": [
                            "Request 8 additional Escape units in next allocation",
                            "Request 7 additional Bronco units in next allocation",
                            "Reduce Mustang order bank by 5 units"
                        ]
                    }
                }
            }
            
            # Add to history
            st.session_state.history.append({
                "request_id": st.session_state.request_id,
                "timestamp": datetime.now().isoformat(),
                "dealer_id": st.session_state.dealer_id,
                "request_type": analysis_type,
                "result": st.session_state.workflow_results
            })
            
            add_to_log("Demo workflow completed successfully", level="info")
            
        # Force app to rerun to show results
        st.rerun()


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
                <div class='metric-value'>{results.get("workflow_type", "Analysis").replace('_', ' ').title()}</div>
                <div class='metric-label'>Analysis Type</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-value'>{len(results.get("recommendations", []))}</div>
                <div class='metric-label'>Recommendations</div>
            </div>""",
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-value'>{results.get("execution_time", 0):.2f}s</div>
                <div class='metric-label'>Processing Time</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    # Key recommendations
    st.markdown("<h3 class='sub-header'>Key Recommendations</h3>", unsafe_allow_html=True)
    
    for i, rec in enumerate(results.get("recommendations", [])):
        st.markdown(
            f"""<div class='recommendation'>
                <strong>#{i+1}:</strong> {rec}
            </div>""",
            unsafe_allow_html=True
        )
    
    # Next steps
    if results.get("next_steps"):
        st.markdown("<h3 class='sub-header'>Suggested Next Steps</h3>", unsafe_allow_html=True)
        
        for i, step in enumerate(results.get("next_steps", [])):
            st.markdown(
                f"""<div class='card'>
                    <strong>Step {i+1}:</strong> {step}
                </div>""",
                unsafe_allow_html=True
            )
    
    # Detailed results tabs
    st.markdown("<h3 class='sub-header'>Detailed Analysis</h3>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Inventory Analysis", "Market Insights", "Gap Analysis", "Visualizations"])
    
    all_results = results.get("results", {})
    
    with tabs[0]:
        # Inventory analysis tab
        inventory_results = all_results.get("inventory_analysis", {})
        
        if inventory_results:
            metrics = inventory_results.get("metrics", {})
            
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
    
    with tabs[1]:
        # Market analysis tab
        market_results = all_results.get("market_analysis", {})
        
        if market_results:
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
    
    with tabs[2]:
        # Gap analysis tab
        gap_results = all_results.get("gap_analysis", {})
        
        if gap_results:
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
    
    with tabs[3]:
        st.info("This tab would show visualizations of your uploaded data. In demo mode, please upload CSV files to see visualizations.")
        
        # Show the visualization tab interface but with a notice that it requires actual data
        if not st.session_state.parsed_data:
            st.warning("Upload inventory, sales, or market data files to enable visualizations")
        

def render_actions_section() -> None:
    """Render the actions section for executing next steps"""
    if not st.session_state.workflow_results:
        return
    
    st.markdown("<h2 class='sub-header'>Actions</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Take action on the recommendations provided by DOSO AI.</p>", unsafe_allow_html=True)
    
    # Create different action buttons 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Analysis Report"):
            st.info("In a production environment, this would generate a downloadable report.")
    
    with col2:
        if st.button("Generate Order Recommendations"):
            with st.spinner("Generating order recommendations..."):
                time.sleep(1)
                st.success("Demo: Order recommendations would be generated in production.")
    
    with col3:
        if st.button("Schedule Consultation"):
            with st.spinner("Checking available appointments..."):
                time.sleep(1)
                st.success("Demo: Consultation request would be submitted in production.")


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
        st.info("In the full version, this tab would allow you to search previous analyses.")


if __name__ == "__main__":
    main()
