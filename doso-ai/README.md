# DOSO AI - Dealer Inventory Optimization System

DOSO AI is a comprehensive AI system that helps Ford dealerships optimize inventory, track allocations, analyze market trends, and plan orders effectively using AI-powered recommendations.

## Overview

The DOSO AI system includes:

- Inventory analysis and optimization recommendations
- Market trend analysis with competitive insights
- Gap analysis for optimizing inventory mix
- Sales velocity tracking and forecasting
- Order planning and allocation tracking
- AI-powered recommendations engine

## Architecture

The system offers two implementation options:

### 1. Standard Implementation (PostgreSQL + Redis)

- FastAPI backend for serving the REST API
- PostgreSQL database for storing dealership data and analysis results
- Redis for caching and performance optimization
- Vector store for semantic search capabilities
- Streamlit frontend for interactive dashboard and reporting

### 2. OpenAI Assistants Implementation (NEW)

- FastAPI backend integrated with OpenAI Assistants API
- No database required - uses OpenAI's file storage and vector search
- Direct file upload to OpenAI for processing
- Same Streamlit frontend interface with OpenAI backend
- Uses OpenAI file attachments for persistent storage

## OpenAI Assistants Implementation

This new implementation completely removes dependencies on PostgreSQL and Redis, leveraging OpenAI's native capabilities:

- **File Storage**: Uploads files directly to OpenAI and attaches them to an Assistant
- **Vector Search**: Uses OpenAI's semantic search for document retrieval
- **Assistants**: Uses OpenAI's Assistants API for all agent queries and analysis
- **Threads**: Maintains context through OpenAI Assistant threads

The OpenAI implementation offers:
- Simplified deployment (no database setup required)
- Enhanced semantic search capabilities
- Streamlined document processing
- Reduced infrastructure requirements

## Running the Standard Version

From the project root directory, run:

```bash
cd doso-ai
python run_streamlit_demo.py
```

This will start the Streamlit server and open the application in your default web browser.

## Running the OpenAI Assistants Version

From the project root directory, run:

```bash
# Make sure you have set OPENAI_API_KEY in your environment
export OPENAI_API_KEY=your_api_key_here

# Run the OpenAI version
python doso-ai/run_openai_app.py
```

This will start:
1. The FastAPI backend using OpenAI Assistants (port 8080)
2. The Streamlit frontend for the OpenAI version (port 8501)

## Sample Data

The repository includes sample data files in the `sample_data` directory:

- `inventory_sample.csv` - Sample inventory data
- `sales_sample.csv` - Sample sales history data
- `market_sample.csv` - Sample market analysis data

You can upload these files in the Streamlit interface to test the analysis capabilities of the demo.

## Full System Setup (For Production)

The full DOSO AI system (Standard version) requires additional setup for the backend services:

1. Install Redis
2. Configure PostgreSQL with pgvector extension
3. Set up environment variables in `.env`
4. Run database migrations using Alembic
5. Start the FastAPI server
6. Launch the Streamlit application

Detailed instructions for the full setup can be found in the project documentation.

## Features

### Inventory Analysis
- Aging analysis
- Turn rate optimization
- Value distribution
- Mix optimization

### Market Analysis
- Competitive positioning
- Market share tracking
- Trend analysis
- Regional preferences

### Gap Analysis
- Inventory vs. ideal mix comparison
- Opportunity identification
- Order recommendation

### Order Planning
- Constraint-aware recommendations
- Allocation optimization
- Factory order tracking

## License

Copyright © 2025 Ford Motor Company
