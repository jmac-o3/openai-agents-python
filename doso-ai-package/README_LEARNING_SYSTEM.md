# DOSO AI Self-Learning System

The DOSO AI Self-Learning System is a specialized component of the Dealer Inventory Optimization System that continuously improves its recommendations through a closed-loop learning process. This system analyzes feedback data from past recommendations to optimize weighting parameters used for future inventory suggestions.

## Key Features

- **Closed-Loop Learning**: Continuously improves by analyzing outcomes of past recommendations
- **Multi-Objective Optimization**: Balances profit, days-to-turn (DTT), market trends, and forecast accuracy 
- **Vector-Based Pattern Recognition**: Uses embeddings to identify patterns across similar configurations
- **Ensemble Forecasting**: Leverages multiple time-series models (Prophet, ARIMA, ETS) for robust predictions
- **Interactive Visualization**: Provides transparent insights into the learning process

## How It Works

The self-learning system operates in a continuous cycle:

1. **Forecasting**: Generates demand predictions using time series forecasting models
2. **Recommendation**: Produces inventory recommendations based on weighted objectives
3. **Feedback Collection**: Processes outcome data from recommendations put into practice
4. **Learning**: Optimizes weighting parameters based on historical performance patterns

## Getting Started

### Prerequisites

- Python 3.9+
- Streamlit
- pandas, numpy
- plotly
- OpenAI Agents SDK

### Running the Application

The application can be run in demo mode (without backend dependencies) using:

```bash
# From the project root directory
./doso-ai/run_learning_app.py
```

Or with command-line options:

```bash
# Use a specific port
./doso-ai/run_learning_app.py --port 8888

# Automatically open browser
./doso-ai/run_learning_app.py --browser
```

## Sample Data

The system comes with sample data files to demonstrate its functionality:

- **sales_sample.csv**: Historical sales data for generating forecasts
- **feedback_sample.csv**: Feedback data from previous recommendations
- **inventory_sample.csv**: Current inventory status data
- **market_sample.csv**: Market trend information

These sample files can be uploaded through the UI to test the system's functionality.

## Learning Parameters

The learning process can be customized with several parameters:

- **Optimization Target**: Primary business objective (balanced, gross_profit, ddt, market_share)
- **Forecasting Model**: Time series model for demand forecasting (prophet, arima, ets)
- **Learning Model**: Machine learning model for parameter optimization (elasticnet, ridge, lasso, randomforest)

## Interface Sections

### System Overview
Provides a high-level view of the system's performance metrics and learning cycle visualization.

### Data Input
Upload data files and configure learning parameters to run cycles or individual components.

### Learning History
View detailed information about past learning cycles, including weight changes and results.

### Forecasts
Visualize demand forecasts for different product configurations.

### Semantic Search
Use natural language to search for patterns in historical feedback data.

## Production Deployment

For a production deployment:

1. Configure database settings in `src/config.py`
2. Set up the vector store for embedding storage
3. Configure API keys for forecasting services
4. Run the application with proper security and authentication

## Demo Mode

When run without backend dependencies, the system operates in demo mode with simulated data processing. This is useful for understanding the workflow and interface without requiring a full database setup.
