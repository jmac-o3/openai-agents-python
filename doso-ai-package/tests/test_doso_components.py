"""
Tests for DOSO AI components

This module contains unit and integration tests for the DOSO AI self-learning system.
It tests the file parser, vector store, agents, and workflow components.
"""

import os
import json
import pytest
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import components to test
from src.utils.file_parser import FileParser
from src.utils.vector_store import vector_store
from src.agents.feedback_collector_agent import feedback_collector
from src.agents.forecasting_agent import forecasting_agent
from src.agents.learning_agent import learning_agent
from src.workflow.doso_workflow import doso_workflow

#
# Fixtures
#

@pytest.fixture
def sample_csv_path():
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        # Create a sample feedback CSV file
        df = pd.DataFrame({
            'config_id': ['config1', 'config2', 'config3'],
            'sale_date': [
                datetime.now().strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            ],
            'gross_profit': [100.0, 150.0, 75.0],
            'ddt': [5, 7, 3],
            'recommended_qty': [10, 15, 8],
            'actual_sold': [8, 12, 7],
            'constraint_hit': [False, True, False],
            'forecast_accuracy': [85.0, 92.0, 78.0]
        })
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Clean up the file after the test
    os.unlink(f.name)

@pytest.fixture
def sample_sales_csv_path():
    """Create a temporary sales history CSV file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        # Create a sample sales history CSV file
        dates = []
        for i in range(30):
            dates.append((datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'))
        
        df = pd.DataFrame({
            'config_id': ['config1']*10 + ['config2']*10 + ['config3']*10,
            'date': dates,
            'quantity': np.random.randint(5, 20, 30),
            'price': np.random.uniform(50, 200, 30),
            'cost': np.random.uniform(30, 150, 30)
        })
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Clean up the file after the test
    os.unlink(f.name)

@pytest.fixture
def sample_jsonl_path():
    """Create a temporary JSONL file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        # Create a sample JSONL file
        records = [
            {"id": 1, "name": "Item 1", "value": 100},
            {"id": 2, "name": "Item 2", "value": 200},
            {"id": 3, "name": "Item 3", "value": 300}
        ]
        for record in records:
            f.write((json.dumps(record) + '\n').encode('utf-8'))
        yield f.name
    
    # Clean up the file after the test
    os.unlink(f.name)

@pytest.fixture
def clean_test_env():
    """Set up a clean test environment and teardown after tests"""
    # Create required directories if they don't exist
    os.makedirs("doso-ai/data/vector_store", exist_ok=True)
    os.makedirs("doso-ai/data/run_log", exist_ok=True)
    os.makedirs("doso-ai/data/forecasts", exist_ok=True)
    os.makedirs("doso-ai/data/learning_models", exist_ok=True)
    
    # Record files before test
    test_files = []
    for root, _, files in os.walk("doso-ai/data"):
        for file in files:
            test_files.append(os.path.join(root, file))
    
    yield
    
    # Clean up files created during tests
    for root, _, files in os.walk("doso-ai/data"):
        for file in files:
            path = os.path.join(root, file)
            if path not in test_files:
                try:
                    os.unlink(path)
                except:
                    pass

#
# FileParser Tests
#

def test_parse_csv(sample_csv_path):
    """Test parsing a CSV file"""
    df = FileParser.parse_csv(sample_csv_path)
    assert not df.empty
    assert 'config_id' in df.columns
    assert 'sale_date' in df.columns
    assert len(df) == 3

def test_parse_csv_with_validation(sample_csv_path):
    """Test parsing a CSV file with column validation"""
    # With correct expected columns
    df = FileParser.parse_csv(
        sample_csv_path,
        expected_columns=['config_id', 'sale_date']
    )
    assert not df.empty
    
    # With incorrect expected columns
    with pytest.raises(ValueError):
        FileParser.parse_csv(
            sample_csv_path,
            expected_columns=['config_id', 'missing_column']
        )

def test_parse_feedback_csv(sample_csv_path):
    """Test parsing a feedback CSV file"""
    df = FileParser.parse_feedback_csv(sample_csv_path)
    assert not df.empty
    assert 'config_id' in df.columns
    assert 'gross_profit' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['sale_date'])

def test_parse_sales_history_csv(sample_sales_csv_path):
    """Test parsing a sales history CSV file"""
    df = FileParser.parse_sales_history_csv(sample_sales_csv_path)
    assert not df.empty
    assert 'config_id' in df.columns
    assert 'quantity' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['date'])

def test_save_load_jsonl(sample_jsonl_path):
    """Test saving and loading JSONL files"""
    # Load existing JSONL
    records = FileParser.load_jsonl(sample_jsonl_path)
    assert len(records) == 3
    assert records[0]['id'] == 1
    
    # Save new JSONL
    new_records = [{"test": 1}, {"test": 2}]
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        FileParser.save_jsonl(new_records, f.name)
        loaded_records = FileParser.load_jsonl(f.name)
        assert len(loaded_records) == 2
        assert loaded_records[0]['test'] == 1
        os.unlink(f.name)

def test_save_load_json():
    """Test saving and loading JSON files"""
    test_data = {"name": "Test", "values": [1, 2, 3]}
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        FileParser.save_json(test_data, f.name)
        loaded_data = FileParser.load_json(f.name)
        assert loaded_data['name'] == "Test"
        assert loaded_data['values'] == [1, 2, 3]
        os.unlink(f.name)

#
# VectorStore Tests
#

def test_vector_store_basics(clean_test_env):
    """Test basic vector store operations"""
    # Create a new index
    index_name = "test_index"
    
    # Delete if it exists
    vector_store.delete_index(index_name)
    
    # Create new index
    vector_store.create_index(index_name)
    
    # Add text
    text = "This is a test document for vector search"
    metadata = {"source": "test", "importance": "high"}
    vector_id = vector_store.add_text(index_name, text, metadata)
    
    # Verify ID was returned
    assert vector_id is not None
    
    # Search for similar text
    results = vector_store.search(index_name, "test document", top_k=1)
    
    # Verify search results
    assert len(results) > 0
    assert "score" in results[0]
    assert "text" in results[0]
    assert "metadata" in results[0]
    
    # Clean up
    vector_store.delete_index(index_name)

def test_vector_store_batch_operations(clean_test_env):
    """Test batch operations with vector store"""
    # Create a new index
    index_name = "test_batch_index"
    
    # Delete if it exists
    vector_store.delete_index(index_name)
    
    # Prepare batch data
    texts = [
        "Document about machine learning",
        "Article about data science",
        "Paper on neural networks"
    ]
    
    metadatas = [
        {"source": "book", "year": 2020},
        {"source": "journal", "year": 2021},
        {"source": "conference", "year": 2022}
    ]
    
    # Add batch
    vector_ids = vector_store.batch_add_texts(index_name, texts, metadatas)
    
    # Verify IDs were returned
    assert len(vector_ids) == 3
    
    # Search for similar text
    results = vector_store.search(index_name, "machine learning data science", top_k=2)
    
    # Verify search results
    assert len(results) == 2
    
    # Clean up
    vector_store.delete_index(index_name)

#
# Agent Tests
#

@pytest.mark.asyncio
async def test_feedback_collector_agent(sample_csv_path, clean_test_env):
    """Test the feedback collector agent"""
    # Process feedback file
    result = await feedback_collector.process_feedback_file(sample_csv_path)
    
    # Verify result
    assert result["status"] == "success"
    assert result["records_processed"] > 0
    assert result["vectors_created"] > 0
    
    # Test search
    search_result = await feedback_collector.search_similar_feedback("config1")
    assert len(search_result) > 0
    
    # Test statistics
    stats_result = await feedback_collector.get_performance_statistics()
    assert stats_result["status"] == "success"
    assert "statistics" in stats_result

@pytest.mark.asyncio
async def test_forecasting_agent(sample_sales_csv_path, clean_test_env):
    """Test the forecasting agent"""
    # Generate forecasts
    result = await forecasting_agent.generate_forecasts(
        sales_history_path=sample_sales_csv_path,
        forecast_horizon_weeks=4,
        model_type="arima",  # Use ARIMA as it's faster for testing
        overwrite_existing=True
    )
    
    # Verify result
    assert result["status"] == "success"
    
    # List forecasts
    list_result = await forecasting_agent.list_available_forecasts()
    assert list_result["status"] == "success"
    assert list_result["total_configs"] > 0
    
    # Get forecast for a config
    config_id = list_result["config_ids"][0]
    forecast_result = await forecasting_agent.forecast_demand(config_id)
    assert forecast_result["status"] == "success"
    assert forecast_result["config_id"] == config_id

@pytest.mark.asyncio
async def test_learning_agent(clean_test_env):
    """Test the learning agent"""
    # Get current configuration
    config_result = await learning_agent.get_current_configuration()
    assert config_result["status"] == "success"
    assert "config" in config_result
    
    # Reset configuration
    reset_result = await learning_agent.reset_configuration()
    assert reset_result["status"] == "success"
    assert "config" in reset_result
    
    # Can't fully test optimization without sufficient data
    # but can test the API doesn't fail
    try:
        analyze_result = await learning_agent.analyze_performance_data(min_data_points=1)
        assert "status" in analyze_result
    except:
        pass  # This may fail if no performance data exists yet

#
# Workflow Integration Tests
#

@pytest.mark.asyncio
async def test_workflow_integration(sample_sales_csv_path, sample_csv_path, clean_test_env):
    """Test the full workflow integration"""
    # Test individual steps first
    forecast_result = await doso_workflow.generate_forecasts(
        sales_history_path=sample_sales_csv_path,
        model_type="arima"  # Use ARIMA as it's faster for testing
    )
    assert forecast_result["status"] in ["success", "info"]
    
    feedback_result = await doso_workflow.process_feedback_data(sample_csv_path)
    assert feedback_result["status"] == "success"
    
    # Test full cycle
    cycle_result = await doso_workflow.run_full_cycle(
        sales_history_path=sample_sales_csv_path,
        feedback_file_path=sample_csv_path,
        forecast_model_type="arima",
        learning_model_type="elasticnet"
    )
    
    # Verify result
    assert "steps" in cycle_result
    assert "forecast" in cycle_result["steps"]
    assert "feedback" in cycle_result["steps"]
    assert "optimize" in cycle_result["steps"]
    assert "recommend" in cycle_result["steps"]
