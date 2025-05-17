"""
DosoWorkflow - Orchestrates the DOSO AI self-learning system

This module orchestrates the workflow between the different agents in the
DOSO AI system:
1. FeedbackCollector - Processes sales outcomes and builds vector search index
2. Forecasting - Generates demand forecasts for configurations
3. Recommendation - Provides optimized configuration recommendations
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from agents import Runner, trace

# Import agent modules with correct paths
from doso_ai.src.agents.feedback_collector_agent import feedback_collector
from doso_ai.src.agents.forecasting_agent import forecasting_agent
from doso_ai.src.agents.recommendation_agent import recommendation_agent
from doso_ai.src.agents.learning_agent import learning_agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("doso-ai/data/run_log/doso_workflow.log", mode='a')
    ]
)
logger = logging.getLogger("doso_workflow")

# Ensure data directories exist
DATA_DIR = Path("doso-ai/data")
RUN_LOGS_DIR = DATA_DIR / "run_log"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RUN_LOGS_DIR, exist_ok=True)


async def run_feedback_collection(feedback_file: str) -> Dict[str, Any]:
    """
    Run the feedback collection process on a feedback file
    
    Args:
        feedback_file: Path to the feedback CSV file
        
    Returns:
        Dictionary with status and summary information
    """
    logger.info(f"Running feedback collection on {feedback_file}")
    
    try:
        # Generate a run ID for tracing
        run_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with trace(workflow_name="DOSO Feedback Collection", trace_id=run_id):
            result = await Runner.run(
                feedback_collector,
                f"Process the feedback data from {feedback_file} and build the vector index",
            )
            
            # Extract useful information from the result
            final_output = result.final_output
            
            # Check for success indications in the output
            if "processed" in final_output and "successfully" in final_output:
                status = "success"
            else:
                status = "warning"
                
            return {
                "status": status,
                "file": feedback_file,
                "run_id": run_id,
                "result": final_output
            }
    
    except Exception as e:
        logger.error(f"Error in feedback collection: {e}")
        return {
            "status": "error",
            "file": feedback_file,
            "error": str(e)
        }


async def run_forecasting(sales_file: str) -> Dict[str, Any]:
    """
    Run the forecasting process on a sales history file
    
    Args:
        sales_file: Path to the sales history CSV file
        
    Returns:
        Dictionary with status and forecast information
    """
    logger.info(f"Running forecasting on {sales_file}")
    
    try:
        # Generate a run ID for tracing
        run_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with trace(workflow_name="DOSO Forecasting", trace_id=run_id):
            result = await Runner.run(
                forecasting_agent,
                f"Generate 8-week demand forecasts from the sales history in {sales_file}",
            )
            
            # Extract useful information from the result
            final_output = result.final_output
            
            # Check for success indications in the output
            if "forecast" in final_output and "generated" in final_output:
                status = "success"
            else:
                status = "warning"
                
            return {
                "status": status,
                "file": sales_file,
                "run_id": run_id,
                "result": final_output
            }
    
    except Exception as e:
        logger.error(f"Error in forecasting: {e}")
        return {
            "status": "error",
            "file": sales_file,
            "error": str(e)
        }


async def run_recommendation(config_id: str) -> Dict[str, Any]:
    """
    Generate a recommendation for a specific configuration
    
    Args:
        config_id: Configuration ID to generate recommendation for
        
    Returns:
        Dictionary with status and recommendation information
    """
    logger.info(f"Generating recommendation for {config_id}")
    
    try:
        # Generate a run ID for tracing
        run_id = f"recommend_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with trace(workflow_name="DOSO Recommendation", trace_id=run_id):
            result = await Runner.run(
                recommendation_agent,
                f"Generate an optimized recommendation for configuration {config_id}",
            )
            
            # Extract useful information from the result
            final_output = result.final_output
            
            # Check for success indications in the output
            if "recommend" in final_output and config_id in final_output:
                status = "success"
            else:
                status = "warning"
                
            return {
                "status": status,
                "config_id": config_id,
                "run_id": run_id,
                "result": final_output
            }
    
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        return {
            "status": "error",
            "config_id": config_id,
            "error": str(e)
        }


async def run_learning_cycle(model_type: str = "auto") -> Dict[str, Any]:
    """
    Run a learning cycle to optimize recommendation weights based on feedback
    
    Args:
        model_type: Type of model to use for learning ('linear', 'elasticnet', 'randomforest', or 'auto')
        
    Returns:
        Dictionary with status and learning information
    """
    logger.info(f"Running learning cycle with model type {model_type}")
    
    try:
        # Generate a run ID for tracing
        run_id = f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with trace(workflow_name="DOSO Learning", trace_id=run_id):
            result = await Runner.run(
                learning_agent,
                f"Run a learning cycle with model_type={model_type} to optimize recommendation weights",
            )
            
            # Extract useful information from the result
            final_output = result.final_output
            
            # Check for success indications in the output
            if "success" in final_output and "weight" in final_output:
                status = "success"
            else:
                status = "warning"
                
            return {
                "status": status,
                "run_id": run_id,
                "model_type": model_type,
                "result": final_output
            }
    
    except Exception as e:
        logger.error(f"Error in learning cycle: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def run_batch_recommendations(config_ids: List[str]) -> Dict[str, Any]:
    """
    Generate recommendations for multiple configurations
    
    Args:
        config_ids: List of configuration IDs
        
    Returns:
        Dictionary with status and recommendations information
    """
    logger.info(f"Generating batch recommendations for {len(config_ids)} configurations")
    
    try:
        # Generate a run ID for tracing
        run_id = f"batch_recommend_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with trace(workflow_name="DOSO Batch Recommendations", trace_id=run_id):
            # Convert the list to a comma-separated string
            config_ids_str = ", ".join(config_ids)
            
            result = await Runner.run(
                recommendation_agent,
                f"Generate batch recommendations for the following configurations: {config_ids_str}",
            )
            
            # Extract useful information from the result
            final_output = result.final_output
            
            # Check for success indications in the output
            if "recommendations" in final_output and "generated" in final_output:
                status = "success"
            else:
                status = "warning"
                
            return {
                "status": status,
                "config_count": len(config_ids),
                "run_id": run_id,
                "result": final_output
            }
    
    except Exception as e:
        logger.error(f"Error in batch recommendations: {e}")
        return {
            "status": "error",
            "config_count": len(config_ids),
            "error": str(e)
        }


async def run_doso_cycle(
    feedback_file: str,
    sales_file: str,
    config_ids: Optional[List[str]] = None,
    run_learning: bool = True
) -> Dict[str, Any]:
    """
    Run a complete DOSO cycle with feedback collection, forecasting, learning, and recommendations
    
    Args:
        feedback_file: Path to the feedback CSV file
        sales_file: Path to the sales history CSV file
        config_ids: Optional list of configuration IDs for recommendations
        run_learning: Whether to run the learning cycle (default: True)
        
    Returns:
        Dictionary with status and results from each stage
    """
    logger.info(f"Running DOSO cycle with feedback={feedback_file}, sales={sales_file}")
    
    results = {
        "cycle_id": f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "started_at": datetime.now().isoformat(),
        "stages": {}
    }
    
    try:
        # Stage 1: Feedback Collection
        feedback_result = await run_feedback_collection(feedback_file)
        results["stages"]["feedback"] = feedback_result
        
        if feedback_result["status"] == "error":
            logger.warning("Feedback collection failed, but continuing with cycle")
        
        # Stage 2: Forecasting
        forecast_result = await run_forecasting(sales_file)
        results["stages"]["forecasting"] = forecast_result
        
        if forecast_result["status"] == "error":
            logger.warning("Forecasting failed, but continuing with cycle")
        
        # Stage 3: Learning (optimize weights based on feedback)
        if run_learning and feedback_result["status"] != "error":
            learning_result = await run_learning_cycle("auto")
            results["stages"]["learning"] = learning_result
            
            if learning_result["status"] == "error":
                logger.warning("Learning cycle failed, but continuing with cycle")
        
        # Stage 4: Recommendations
        if config_ids:
            if len(config_ids) == 1:
                # Single recommendation
                recommend_result = await run_recommendation(config_ids[0])
                results["stages"]["recommendation"] = recommend_result
            else:
                # Batch recommendations
                recommend_result = await run_batch_recommendations(config_ids)
                results["stages"]["recommendations"] = recommend_result
        
        # Determine overall cycle status
        all_stages = [
            feedback_result["status"], 
            forecast_result["status"]
        ]
        
        # Add learning status if it was run
        if run_learning and "learning" in results["stages"]:
            all_stages.append(results["stages"]["learning"]["status"])
            
        if all(status == "error" for status in all_stages):
            results["status"] = "failed"
        elif any(status in ["error", "warning"] for status in all_stages):
            results["status"] = "partial"
        else:
            results["status"] = "success"
        
        results["completed_at"] = datetime.now().isoformat()
        
        # Log the cycle completion
        logger.info(f"DOSO cycle completed with status: {results['status']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in DOSO cycle: {e}")
        results["status"] = "error"
        results["error"] = str(e)
        results["completed_at"] = datetime.now().isoformat()
        return results


# Example usage:
# async def main():
#     await run_doso_cycle(
#         feedback_file="doso-ai/sample_data/feedback_sample.csv",
#         sales_file="doso-ai/sample_data/sales_sample.csv",
#         config_ids=["config1", "config2", "config3"]
#     )
