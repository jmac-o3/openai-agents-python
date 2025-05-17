"""
Tests for the DOSO AI Self-Learning System components.

These tests verify that all components of the self-learning system function correctly:
- Feedback collection and embedding creation
- Forecasting with various models
- Learning and weight optimization
- Vector search for pattern discovery
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import pytest
from datetime import datetime

# Add project root to path to make imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use environment variables if available, otherwise use test values
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///:memory:")


class TestFeedbackCollector(unittest.TestCase):
    """Tests for the feedback collection and embedding process"""
    
    @pytest.mark.asyncio
    async def test_process_feedback_file(self):
        """Test processing a feedback file"""
        # Create a mock FeedbackCollectorAgent
        with patch('src.agents.feedback_collector_agent.FeedbackCollectorAgent') as MockAgent:
            # Configure the mock
            mock_instance = MockAgent.return_value
            mock_instance.process_feedback_file.return_value = {
                "status": "success",
                "records_processed": 5,
                "vectors_created": 5
            }
            
            # Create a temporary CSV file for testing
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                f.write(b"config_id,sale_date,gross_profit,ddt,recommended_qty,actual_sold,outcome_rating\n")
                f.write(b"config1,2025-03-01,1850,35,10,12,0.85\n")
                f.write(b"config2,2025-03-15,1200,25,8,8,0.95\n")
                temp_path = f.name
            
            try:
                # Call the method
                from src.agents.feedback_collector_agent import feedback_collector
                result = await feedback_collector.process_feedback_file(temp_path)
                
                # Check the result
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["records_processed"], 5)
                self.assertEqual(result["vectors_created"], 5)
            finally:
                # Clean up the temporary file
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_create_embedding(self):
        """Test creating an embedding from text"""
        with patch('openai.Embedding.create') as mock_create:
            # Configure the mock
            mock_create.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]
            }
            
            # Import the function
            from src.utils.vector_store import create_embedding
            
            # Call the function
            text = "Test text for embedding"
            embedding = await create_embedding(text)
            
            # Check the result
            self.assertEqual(len(embedding), 5)
            self.assertEqual(embedding[0], 0.1)
            
            # Verify the mock was called correctly
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            self.assertEqual(kwargs["input"], text)
            self.assertEqual(kwargs["model"], "text-embedding-ada-002")


class TestForecasting(unittest.TestCase):
    """Tests for the forecasting agent"""
    
    @pytest.mark.asyncio
    async def test_forecast_demand(self):
        """Test forecasting demand for a configuration"""
        # Create a mock ForecastingAgent
        with patch('src.agents.forecasting_agent.ForecastingAgent') as MockAgent:
            # Configure the mock
            mock_instance = MockAgent.return_value
            mock_instance.forecast_demand.return_value = {
                "status": "success",
                "config_id": "config1",
                "forecast": [10, 12, 15, 11, 14, 16, 13]
            }
            
            # Call the method
            from src.agents.forecasting_agent import forecasting_agent
            result = await forecasting_agent.forecast_demand("config1")
            
            # Check the result
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["config_id"], "config1")
            self.assertEqual(len(result["forecast"]), 7)
            self.assertEqual(result["forecast"][2], 15)
    
    @pytest.mark.asyncio
    async def test_forecast_with_prophet(self):
        """Test forecasting with Prophet model"""
        # Skip if Prophet not installed
        try:
            import prophet
        except ImportError:
            self.skipTest("Prophet not installed")
        
        # Create a mock ForecastingAgent
        with patch('src.agents.forecasting_agent.ForecastingAgent') as MockAgent:
            # Configure the mock
            mock_instance = MockAgent.return_value
            mock_instance._forecast_with_prophet.return_value = {
                "forecast": [10, 12, 15, 11, 14, 16, 13],
                "metrics": {
                    "mape": 0.15,
                    "mae": 2.5
                }
            }
            
            # Create test data
            weekly_df = pd.DataFrame({
                "ds": pd.date_range(start="2025-01-01", periods=20, freq="W"),
                "y": np.random.randint(5, 20, 20)
            })
            
            # Call the method
            from src.agents.forecasting_agent import forecasting_agent
            result = await forecasting_agent._forecast_with_prophet(
                weekly_df, "config1", 7
            )
            
            # Check the result
            self.assertIn("forecast", result)
            self.assertIn("metrics", result)
            self.assertEqual(len(result["forecast"]), 7)


class TestLearning(unittest.TestCase):
    """Tests for the learning agent"""
    
    @pytest.mark.asyncio
    async def test_optimize_weights(self):
        """Test optimizing weights based on feedback"""
        # Create a mock LearningAgent
        with patch('src.agents.learning_agent.LearningAgent') as MockAgent:
            # Configure the mock
            mock_instance = MockAgent.return_value
            mock_instance.optimize_weights.return_value = {
                "status": "success",
                "old_weights": {
                    "profit_weight": 0.30,
                    "ddt_weight": 0.25,
                    "market_weight": 0.25,
                    "forecast_weight": 0.20
                },
                "new_weights": {
                    "profit_weight": 0.35,
                    "ddt_weight": 0.25,
                    "market_weight": 0.20,
                    "forecast_weight": 0.20
                },
                "improvement": 0.15
            }
            
            # Call the method
            from src.agents.learning_agent import learning_agent
            result = await learning_agent.optimize_weights(
                optimization_target="balanced"
            )
            
            # Check the result
            self.assertEqual(result["status"], "success")
            self.assertIn("old_weights", result)
            self.assertIn("new_weights", result)
            self.assertEqual(result["new_weights"]["profit_weight"], 0.35)
            self.assertEqual(result["improvement"], 0.15)
    
    @pytest.mark.asyncio
    async def test_get_current_configuration(self):
        """Test getting the current weights configuration"""
        # Create a mock LearningAgent
        with patch('src.agents.learning_agent.LearningAgent') as MockAgent:
            # Configure the mock
            mock_instance = MockAgent.return_value
            mock_instance.get_current_configuration.return_value = {
                "status": "success",
                "config": {
                    "profit_weight": 0.35,
                    "ddt_weight": 0.25,
                    "market_weight": 0.20,
                    "forecast_weight": 0.20
                }
            }
            
            # Call the method
            from src.agents.learning_agent import learning_agent
            result = await learning_agent.get_current_configuration()
            
            # Check the result
            self.assertEqual(result["status"], "success")
            self.assertIn("config", result)
            self.assertEqual(result["config"]["profit_weight"], 0.35)


class TestDosoWorkflow(unittest.TestCase):
    """Tests for the DOSO workflow"""
    
    @pytest.mark.asyncio
    async def test_run_full_cycle(self):
        """Test running a full learning cycle"""
        # Create a mock DosoWorkflow
        with patch('src.workflow.doso_workflow.DosoWorkflow') as MockWorkflow:
            # Configure the mock
            mock_instance = MockWorkflow.return_value
            mock_instance.run_full_cycle.return_value = {
                "status": "success",
                "message": "Learning cycle completed successfully",
                "steps": {
                    "forecast": {"status": "success", "message": "Generated forecasts for 5 configs"},
                    "feedback": {"status": "success", "message": "Processed 20 feedback records"},
                    "optimize": {"status": "success", "message": "Updated learning weights"},
                    "recommend": {"status": "success", "message": "Generated recommendations"}
                },
                "cycle_duration_seconds": 3.5
            }
            
            # Create a temporary file for testing
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                f.write(b"date,config_id,quantity\n")
                f.write(b"2025-01-01,config1,10\n")
                sales_history_path = f.name
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                f.write(b"config_id,sale_date,gross_profit,ddt,recommended_qty,actual_sold,outcome_rating\n")
                f.write(b"config1,2025-03-01,1850,35,10,12,0.85\n")
                feedback_path = f.name
            
            try:
                # Call the method
                from src.workflow.doso_workflow import doso_workflow
                result = await doso_workflow.run_full_cycle(
                    sales_history_path=sales_history_path,
                    feedback_file_path=feedback_path,
                    optimization_target="balanced",
                    forecast_model_type="prophet",
                    learning_model_type="elasticnet"
                )
                
                # Check the result
                self.assertEqual(result["status"], "success")
                self.assertIn("steps", result)
                self.assertIn("forecast", result["steps"])
                self.assertIn("feedback", result["steps"])
                self.assertIn("optimize", result["steps"])
                self.assertIn("recommend", result["steps"])
                self.assertEqual(result["cycle_duration_seconds"], 3.5)
            finally:
                # Clean up the temporary files
                os.unlink(sales_history_path)
                os.unlink(feedback_path)


class TestDatabaseIntegration:
    """Database integration tests for the learning system components"""
    
    @classmethod
    def setup_class(cls):
        """Set up the database for testing"""
        # Create an in-memory SQLite database for testing
        cls.engine = create_engine("sqlite:///:memory:")
        
        # Create the required tables
        with cls.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS configurations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    parameters TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id TEXT PRIMARY KEY,
                    config_id TEXT REFERENCES configurations(id),
                    model_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    config_id TEXT REFERENCES configurations(id),
                    sale_date DATE NOT NULL,
                    gross_profit NUMERIC NOT NULL,
                    ddt INTEGER NOT NULL,
                    recommended_qty INTEGER NOT NULL,
                    actual_sold INTEGER NOT NULL,
                    outcome_rating NUMERIC NOT NULL,
                    embedding TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS learning_cycles (
                    id TEXT PRIMARY KEY,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    optimization_target TEXT NOT NULL,
                    forecast_model TEXT NOT NULL,
                    learning_model TEXT NOT NULL,
                    old_weights TEXT,
                    new_weights TEXT,
                    improvement NUMERIC,
                    status TEXT NOT NULL,
                    details TEXT
                )
            """))
            
            # Insert some test data
            conn.execute(text("""
                INSERT INTO configurations (id, name, description, parameters)
                VALUES ('config1', 'Test Config', 'Test Configuration', '{"model": "F-150", "trim": "XLT"}')
            """))
            
            conn.execute(text("""
                INSERT INTO feedback (id, config_id, sale_date, gross_profit, ddt, recommended_qty, actual_sold, outcome_rating)
                VALUES ('feedback1', 'config1', '2025-03-01', 1850, 35, 10, 12, 0.85)
            """))
            
            conn.commit()
    
    @classmethod
    def teardown_class(cls):
        """Clean up after tests"""
        cls.engine.dispose()
    
    @pytest.mark.asyncio
    async def test_db_store_forecast(self):
        """Test storing a forecast in the database"""
        from src.agents.forecasting_agent import ForecastingAgent
        
        # Mock the database connection
        with patch('sqlalchemy.create_engine', return_value=self.engine):
            agent = ForecastingAgent()
            
            # Create a forecast
            forecast_data = {
                "config_id": "config1",
                "model_type": "prophet",
                "data": [10, 12, 15, 11, 14, 16, 13],
                "metrics": {
                    "mape": 0.15,
                    "mae": 2.5
                }
            }
            
            # Store the forecast
            result = await agent.store_forecast(forecast_data)
            
            # Check the result
            assert result["status"] == "success"
            
            # Verify the forecast was stored
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM forecasts WHERE config_id = 'config1'"))
                row = result.fetchone()
                assert row is not None
                assert row.model_type == "prophet"
    
    @pytest.mark.asyncio
    async def test_db_learn_from_feedback(self):
        """Test learning from feedback data in the database"""
        from src.agents.learning_agent import LearningAgent
        
        # Mock the database connection
        with patch('sqlalchemy.create_engine', return_value=self.engine):
            agent = LearningAgent()
            
            # Learn from feedback
            result = await agent.learn_from_feedback(optimization_target="balanced")
            
            # Check the result
            assert result["status"] == "success"
            
            # Verify the learning cycle was stored
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM learning_cycles"))
                row = result.fetchone()
                assert row is not None
                assert row.optimization_target == "balanced"


if __name__ == "__main__":
    unittest.main()
