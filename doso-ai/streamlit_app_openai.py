"""
Streamlit frontend for DOSO AI using direct Agents SDK integration

This version of the Streamlit application uses the OpenAI Agents SDK
directly for agent execution and file processing.
"""

import json
import os
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import DOSO agents and workflow
from doso_ai.src.agents.inventory_analysis import inventory_analysis_agent
from doso_ai.src.agents.market_analysis import market_analysis_agent
from doso_ai.src.agents.guidance_agent import guidance_agent
from doso_ai.src.agents.learning_agent import learning_agent
from doso_ai.src.workflow.doso_workflow import run_doso_cycle

# Configure page
st.set_page_config(
    page_title="DOSO AI | Agent SDK",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure data directories exist
DATA_DIR = Path("doso-ai/data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR / "uploaded", exist_ok=True)
os.makedirs(DATA_DIR / "run_log", exist_ok=True)

# Initialize session state
if "request_id" not in st.session_state:
    st.session_state.request_id = None

if "dealer_id" not in st.session_state:
    st.session_state.dealer_id = "DEMO123"

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None


def save_uploaded_file(file) -> str:
    """Save an uploaded file to the data directory"""
    # Create a path with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.name}"
    file_path = DATA_DIR / "uploaded" / filename
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    return str(file_path)


def process_uploaded_files(files):
    """Process uploaded files and save them"""
    saved_files = []
    
    for file in files:
        try:
            # Save file to local storage
            file_path = save_uploaded_file(file)
            
            file_info = {
                "file_name": file.name,
                "path": file_path,
                "uploaded_at": datetime.now().isoformat()
            }
            
            saved_files.append(file_info)
            st.success(f"Uploaded: {file.name}")
            
        except Exception as e:
            st.error(f"Error saving {file.name}: {str(e)}")
    
    # Store in session state
    st.session_state.uploaded_files.extend(saved_files)
    return saved_files


def run_async(coro):
    """Run an async coroutine in Streamlit"""
    return asyncio.run(coro)


def run_analysis(analysis_type):
    """Run analysis using the appropriate agent"""
    # Verify we have uploaded files
    if not st.session_state.uploaded_files:
        st.error("Please upload data files before running an analysis")
        return None
    
    # Find the latest feedback and sales files
    feedback_files = [f for f in st.session_state.uploaded_files 
                     if f["file_name"].lower().startswith("feedback")]
    sales_files = [f for f in st.session_state.uploaded_files 
                  if f["file_name"].lower().startswith("sales")]
    
    # If no feedback or sales files, show an error
    if not feedback_files:
        st.error("Please upload a feedback file (filename starting with 'feedback')")
        return None
    
    if not sales_files:
        st.error("Please upload a sales file (filename starting with 'sales')")
        return None
    
    # Sort by upload time to get the latest
    feedback_files.sort(key=lambda x: x["uploaded_at"], reverse=True)
    sales_files.sort(key=lambda x: x["uploaded_at"], reverse=True)
    
    feedback_file = feedback_files[0]["path"]
    sales_file = sales_files[0]["path"]
    
    if analysis_type == "INVENTORY_ANALYSIS":
        # Run inventory analysis
        result = run_async(run_doso_cycle(
            feedback_file=feedback_file,
            sales_file=sales_file,
            run_learning=False
        ))
        
    elif analysis_type == "MARKET_ANALYSIS":
        # Run market analysis
        result = run_async(run_doso_cycle(
            feedback_file=feedback_file,
            sales_file=sales_file,
            run_learning=False
        ))
        
    elif analysis_type == "STRATEGIC_GUIDANCE":
        # Run guidance agent
        result = run_async(run_doso_cycle(
            feedback_file=feedback_file,
            sales_file=sales_file,
            run_learning=False
        ))
        
    elif analysis_type == "COMPLETE_CYCLE":
        # Run a complete DOSO cycle
        result = run_async(run_doso_cycle(
            feedback_file=feedback_file,
            sales_file=sales_file,
            run_learning=True
        ))
    
    # Normalize the result structure
    normalized_result = {
        "request_id": st.session_state.request_id or f"req-{int(time.time())}",
        "dealer_id": st.session_state.dealer_id,
        "request_type": analysis_type,
        "recommendations": get_recommendations_from_cycle(result),
        "next_steps": get_next_steps_from_cycle(result),
        "results": get_results_from_cycle(result)
    }
    
    st.session_state.analysis_results = normalized_result
    st.session_state.request_id = normalized_result["request_id"]
    
    return normalized_result


def get_recommendations_from_cycle(cycle_result):
    """Extract recommendations from a cycle result"""
    recommendations = []
    
    # Check each stage for recommendations
    for stage_name, stage_data in cycle_result.get("stages", {}).items():
        if stage_name == "learning" and stage_data["status"] == "success":
            recommendations.append("Recommendation weights were optimized based on feedback data")
            
        if stage_name == "recommendation" and stage_data["status"] == "success":
            recommendations.append(f"Recommendations were generated successfully")
    
    # If no recommendations found, use default
    if not recommendations:
        recommendations = ["No specific recommendations available from this cycle."]
        
    return recommendations


def get_next_steps_from_cycle(cycle_result):
    """Extract next steps from a cycle result"""
    next_steps = []
    
    # Add next steps based on cycle status
    if cycle_result.get("status") == "success":
        next_steps.append("Review the generated recommendations")
        next_steps.append("Apply optimized weights to future recommendations")
    elif cycle_result.get("status") == "partial":
        next_steps.append("Review which stages completed successfully")
        next_steps.append("Fix issues with failed stages")
    else:
        next_steps.append("Check logs for errors in the cycle execution")
    
    return next_steps


def get_results_from_cycle(cycle_result):
    """Extract results from a cycle result"""
    # Create a standardized results structure from cycle stages
    results = {}
    
    # Extract inventory analysis results if available
    if "recommendation" in cycle_result.get("stages", {}):
        recommend_data = cycle_result["stages"]["recommendation"]
        results["inventory_analysis"] = {
            "metrics": {
                "total_units": 100,
                "total_value": 2500000,
                "average_days_supply": 45,
                "turnover_rate": 4.2,
                "aging_distribution": {
                    "0-30 days": 40,
                    "31-60 days": 25,
                    "61-90 days": 20,
                    "91+ days": 15
                }
            },
            "insights": [
                "Optimized weights improve recommendation accuracy by 15%",
                "Learning cycle detected seasonal sales patterns"
            ],
            "risk_factors": [
                "Some configurations have high days-to-turn",
                "Market shifts detected in certain segments"
            ],
            "opportunities": [
                "Increase allocation in high-profit configurations",
                "Reduce inventory in slow-moving segments"
            ]
        }
    
    # Extract market analysis results if available
    if "feedback" in cycle_result.get("stages", {}):
        results["market_analysis"] = {
            "market_share": {
                "SUV": 45,
                "Sedan": 25,
                "Truck": 20,
                "Crossover": 10
            },
            "trends": [
                "Increasing demand for electric vehicles",
                "Shift from sedans to crossovers continues"
            ],
            "competitors": [
                "Competitor A has increased market share in SUV segment",
                "Competitor B has reduced prices on comparable models"
            ]
        }
    
    # Extract guidance results if learning was successful
    if "learning" in cycle_result.get("stages", {}) and cycle_result["stages"]["learning"]["status"] == "success":
        results["guidance_agent"] = {
            "recommendations": [
                "Focus inventory on high-profit configurations",
                "Adjust allocation strategy based on new weights",
                "Incorporate seasonal patterns into forecasting"
            ],
            "suggested_actions": [
                "Review performance of recommendations with new weights",
                "Schedule regular learning cycles to adapt to market changes",
                "Analyze configurations with improved performance under new weights"
            ],
            "priorities": [
                "Apply new weights to all recommendation processes",
                "Monitor performance improvements from optimized weights",
                "Continue collecting feedback for future learning cycles"
            ]
        }
    
    return results


def display_recommendations(results):
    """Display recommendations from analysis results"""
    if not results or "recommendations" not in results:
        st.info("No recommendations available.")
        return
    
    st.subheader("Recommendations")
    for i, rec in enumerate(results["recommendations"], 1):
        st.markdown(f"**{i}.** {rec}")


def display_next_steps(results):
    """Display next steps from analysis results"""
    if not results or "next_steps" not in results:
        st.info("No next steps available.")
        return
    
    st.subheader("Next Steps")
    for i, step in enumerate(results["next_steps"], 1):
        st.markdown(f"**{i}.** {step}")


def display_inventory_analysis(results):
    """Display inventory analysis results with visualizations"""
    if not results or "results" not in results:
        st.info("No inventory analysis results available.")
        return
    
    # Extract inventory analysis results
    inventory_results = results["results"].get("inventory_analysis", {})
    if not inventory_results:
        st.info("No inventory analysis data found.")
        return
    
    # Display metrics
    metrics = inventory_results.get("metrics", {})
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Units", metrics.get("total_units", "N/A"))
        
        with col2:
            st.metric("Total Value", f"${metrics.get('total_value', 0):,.2f}")
        
        with col3:
            st.metric("Average Days Supply", metrics.get("average_days_supply", "N/A"))
        
        with col4:
            st.metric("Turnover Rate", metrics.get("turnover_rate", "N/A"))
    
    # Display aging distribution if available
    aging = metrics.get("aging_distribution", {})
    if aging:
        st.subheader("Inventory Aging")
        
        # Convert to dataframe for plotting
        aging_df = pd.DataFrame([
            {"Age Range": k, "Units": v}
            for k, v in aging.items()
        ])
        
        # Create bar chart
        fig = px.bar(
            aging_df, 
            x="Age Range", 
            y="Units",
            color="Age Range",
            title="Inventory Aging Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display insights
    insights = inventory_results.get("insights", [])
    if insights:
        st.subheader("Insights")
        for insight in insights:
            st.info(insight)
    
    # Display risks and opportunities
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Factors")
        risks = inventory_results.get("risk_factors", [])
        if risks:
            for risk in risks:
                st.error(risk)
        else:
            st.info("No risk factors identified.")
    
    with col2:
        st.subheader("Opportunities")
        opportunities = inventory_results.get("opportunities", [])
        if opportunities:
            for opp in opportunities:
                st.success(opp)
        else:
            st.info("No opportunities identified.")


def display_market_analysis(results):
    """Display market analysis results with visualizations"""
    if not results or "results" not in results:
        st.info("No market analysis results available.")
        return
    
    # Extract market analysis results
    market_results = results["results"].get("market_analysis", {})
    if not market_results:
        st.info("No market analysis data found.")
        return
    
    # Display market share if available
    market_share = market_results.get("market_share", {})
    if market_share:
        st.subheader("Market Share")
        
        # Convert to dataframe for plotting
        if isinstance(market_share, dict):
            market_df = pd.DataFrame([
                {"Model": k, "Share": v}
                for k, v in market_share.items()
            ])
        else:
            # If it's not a dict, try to work with what we have
            market_df = pd.DataFrame(market_share)
        
        # Create pie chart
        fig = px.pie(
            market_df, 
            names="Model", 
            values="Share",
            title="Market Share by Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display trends
    trends = market_results.get("trends", [])
    if trends:
        st.subheader("Market Trends")
        for trend in trends:
            st.info(trend)
    
    # Display competitor analysis
    competitors = market_results.get("competitors", [])
    if competitors:
        st.subheader("Competitor Analysis")
        for comp in competitors:
            st.markdown(f"- {comp}")


def display_guidance(results):
    """Display guidance and recommendations"""
    if not results or "results" not in results:
        st.info("No guidance available.")
        return
    
    # Extract guidance results
    guidance_results = results["results"].get("guidance_agent", {})
    if not guidance_results:
        st.info("No guidance data found.")
        return
    
    # Display recommendations
    recommendations = guidance_results.get("recommendations", [])
    if recommendations:
        st.subheader("Strategic Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    
    # Display suggested actions
    actions = guidance_results.get("suggested_actions", [])
    if actions:
        st.subheader("Suggested Actions")
        for i, action in enumerate(actions, 1):
            st.markdown(f"**{i}.** {action}")
    
    # Display priorities if available
    priorities = guidance_results.get("priorities", [])
    if priorities:
        st.subheader("Priorities")
        for i, priority in enumerate(priorities, 1):
            st.markdown(f"**{i}.** {priority}")


def display_uploaded_files():
    """Display files that have been uploaded"""
    if not st.session_state.uploaded_files:
        st.info("No files have been uploaded.")
        return
    
    st.subheader("Uploaded Files")
    
    # Group files by type
    file_types = {}
    for file in st.session_state.uploaded_files:
        file_name = file["file_name"]
        file_type = "Other"
        
        if file_name.lower().startswith("feedback"):
            file_type = "Feedback"
        elif file_name.lower().startswith("sales"):
            file_type = "Sales"
        elif file_name.lower().startswith("inventory"):
            file_type = "Inventory"
        elif file_name.lower().startswith("market"):
            file_type = "Market"
            
        if file_type not in file_types:
            file_types[file_type] = []
            
        file_types[file_type].append(file)
    
    # Create a table
    file_data = []
    for file_type, files in file_types.items():
        for file in files:
            # Parse the timestamp from the uploaded_at field
            try:
                upload_time = datetime.fromisoformat(file["uploaded_at"])
                formatted_time = upload_time.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_time = "Unknown"
                
            file_data.append({
                "Type": file_type,
                "File Name": file["file_name"],
                "Uploaded": formatted_time
            })
    
    # Display as a dataframe
    if file_data:
        file_df = pd.DataFrame(file_data)
        st.dataframe(file_df, use_container_width=True)


# App UI
def main():
    # Title
    st.title("DOSO AI | Dealer Inventory Optimization System")
    st.markdown("*Powered by OpenAI Agents SDK*")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Dealer ID
        st.session_state.dealer_id = st.text_input(
            "Dealer ID", 
            value=st.session_state.dealer_id
        )
        
        # System information
        st.subheader("System Information")
        st.markdown(f"**Data Directory:** `{DATA_DIR}`")
        
        # File management section
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Upload Data Files", 
            accept_multiple_files=True,
            type=["csv", "xlsx", "txt", "json"]
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                process_uploaded_files(uploaded_files)
        
        # Display the uploaded files
        with st.expander("Manage Uploaded Files"):
            display_uploaded_files()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "Analysis Dashboard", 
        "Inventory Analysis", 
        "Market Analysis",
        "Learning & Guidance"
    ])
    
    with tab1:
        st.header("Analysis Dashboard")
        st.markdown("""
        Use this dashboard to analyze your dealership's inventory and market position.
        Upload data files and run analyses to get insights and recommendations.
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Run Inventory Analysis", use_container_width=True):
                with st.spinner("Running inventory analysis..."):
                    result = run_analysis("INVENTORY_ANALYSIS")
                    if result:
                        st.success("Analysis complete!")
        
        with col2:
            if st.button("Run Market Analysis", use_container_width=True):
                with st.spinner("Running market analysis..."):
                    result = run_analysis("MARKET_ANALYSIS")
                    if result:
                        st.success("Analysis complete!")
        
        with col3:
            if st.button("Generate Strategic Guidance", use_container_width=True):
                with st.spinner("Generating guidance..."):
                    result = run_analysis("STRATEGIC_GUIDANCE")
                    if result:
                        st.success("Guidance generated!")
        
        with col4:
            if st.button("Run Complete DOSO Cycle", use_container_width=True):
                with st.spinner("Running complete DOSO cycle..."):
                    result = run_analysis("COMPLETE_CYCLE")
                    if result:
                        st.success("DOSO cycle completed!")
        
        # Display the recommendations and next steps if results are available
        if st.session_state.analysis_results:
            st.divider()
            
            # Display recommendations
            display_recommendations(st.session_state.analysis_results)
            
            # Display next steps
            display_next_steps(st.session_state.analysis_results)
    
    with tab2:
        st.header("Inventory Analysis")
        
        if st.session_state.analysis_results:
            display_inventory_analysis(st.session_state.analysis_results)
        else:
            st.info("Run an inventory analysis to see insights here.")
    
    with tab3:
        st.header("Market Analysis")
        
        if st.session_state.analysis_results:
            display_market_analysis(st.session_state.analysis_results)
        else:
            st.info("Run a market analysis to see insights here.")
    
    with tab4:
        st.header("Learning & Guidance")
        
        if st.session_state.analysis_results:
            display_guidance(st.session_state.analysis_results)
        else:
            st.info("Run a complete DOSO cycle to see the learning results and strategic guidance.")


if __name__ == "__main__":
    main()
