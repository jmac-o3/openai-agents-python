# DOSO AI System Patterns

## Architecture Overview

The DOSO AI system is built as a modular application with clear separation of concerns. It follows a layered architecture with multiple deployment options.

```
┌─────────────────┐
│   UI Layer      │
│  (Streamlit)    │
└───────┬─────────┘
        │
┌───────▼─────────┐
│   API Layer     │
│   (FastAPI)     │
└───────┬─────────┘
        │
┌───────▼─────────┐      ┌─────────────────┐
│  Business Logic │◄────►│   Vector Store  │
│     Layer       │      │                 │
└───────┬─────────┘      └─────────────────┘
        │
┌───────▼─────────┐      ┌─────────────────┐
│   Data Layer    │◄────►│    Database     │
│                 │      │ (PostgreSQL/    │
└─────────────────┘      │  OpenAI Files)  │
                         └─────────────────┘
```

## Key Design Patterns

### Multi-Tenant Architecture

The application is designed to support multiple dealerships, each with their own data and settings, while sharing the same infrastructure.

- **Data Isolation**: Each dealership's data is logically isolated
- **Shared Processing**: Core ML models and processing logic are shared across tenants
- **Customizable Parameters**: Each dealership can have customized settings and parameters

### Agent-Based Processing

The system uses specialized agents for different tasks, each focused on a specific part of the analysis or recommendation process:

1. **Market Analysis Agent**: Processes market data and competitive information
2. **Inventory Analysis Agent**: Analyzes current inventory state and aging
3. **Gap Analysis Agent**: Identifies opportunities in the current inventory mix
4. **Forecasting Agent**: Predicts future demand patterns
5. **Recommendation Agent**: Combines insights to produce actionable recommendations
6. **Feedback Collector Agent**: Gathers outcome data to improve recommendations
7. **Learning Agent**: Optimizes system parameters based on feedback
8. **Triage Agent**: Routes queries to appropriate specialized agents

### Workflow Orchestration

Multiple orchestration approaches depending on deployment mode:

1. **Standard Implementation**:
   - Centralized workflow orchestration via `doso_workflow.py`
   - Event-driven processing with Redis for async operations
   - Database-backed tracking of workflow state

2. **OpenAI Assistants Implementation**:
   - Thread-based orchestration via `orchestration_openai.py`
   - OpenAI Assistant threads for maintaining context
   - File attachments for data persistence

3. **Self-Learning System**:
   - Closed-loop learning cycle managed by `orchestration.py`
   - Periodic training based on feedback data
   - Ensemble learning for optimizing parameters

### Repository Pattern

Data access is abstracted through repository classes that provide:

- **Data Access Layer**: Consistent interface for data operations
- **Query Construction**: SQL or vector-based queries generated as needed
- **Entity Mapping**: Translation between database models and domain entities

### Factory Pattern

Component creation is managed through factory methods:

- **Agent Factory**: Creates specialized agents based on needs
- **Model Factory**: Initializes different forecasting models
- **Recommendation Factory**: Builds recommendations based on configuration

### Strategy Pattern

Pluggable algorithms allow for flexibility in implementation:

- **Forecasting Strategy**: Different time series algorithms (ARIMA, Prophet, etc.)
- **Recommendation Strategy**: Different approaches to generating recommendations
- **Optimization Strategy**: Different methods for parameter optimization

## Data Management

### Vector Store

The system uses vector embeddings for semantic search and pattern recognition:

- **Document Storage**: Feedback data, market reports, and configurations are stored as embeddings
- **Similarity Search**: Vector similarity operations power the recommendation system
- **Clustering**: Similar configurations are grouped for pattern recognition

### Time Series Data

Historical data is stored and managed to support forecasting:

- **Sales History**: Tracked over time with seasonality analysis
- **Inventory Metrics**: Days-to-turn and stock levels tracked over time
- **Market Share**: Competitive position tracked over time

### PostgreSQL + pgvector

The standard implementation uses PostgreSQL with pgvector extension:

- **Relational Data**: Standard entity relationships
- **Vector Columns**: Store embeddings for semantic search
- **SQL + Vector Queries**: Combine traditional filtering with vector similarity

### OpenAI File Storage

The OpenAI Assistants implementation uses OpenAI's file storage:

- **File Attachments**: Data stored as files attached to assistants
- **Vector Search**: OpenAI's vector search capabilities for retrieval
- **Reduced Infrastructure**: Eliminates need for database management

## API Design

### RESTful Endpoints

The FastAPI backend provides RESTful endpoints:

- **Resource-Based**: Endpoints follow resource names (inventory, market, etc.)
- **Standard Methods**: GET, POST, PUT, DELETE used consistently
- **Query Parameters**: Filtering and pagination supported
- **Authentication**: JWT-based authentication for security

### OpenAPI Documentation

All endpoints are fully documented with OpenAPI:

- **Auto-Generated Docs**: FastAPI generates Swagger documentation
- **Request/Response Schemas**: All data structures are documented
- **Example Values**: Provided for easy testing

### WebSockets (Planned)

Future versions may include WebSocket support for:

- **Real-Time Updates**: Push notifications for inventory changes
- **Live Dashboard**: Real-time metrics and alerts

## User Interface Patterns

### Streamlit Components

The UI is built with Streamlit for rapid development:

- **Dashboard Layout**: Cards and grid layouts for key metrics
- **Interactive Charts**: Plotly visualizations for data exploration
- **File Upload**: Simple interface for data import
- **Configuration Forms**: Parameter setting and customization

### Responsive Design

The interface adapts to different screen sizes:

- **Column Layout**: Multi-column on desktop, single column on mobile
- **Chart Scaling**: Visualizations resize to fit available space
- **Prioritized Content**: Critical information shown first on small screens

### Progressive Disclosure

Information is presented with a hierarchy of detail:

- **Summary Cards**: High-level metrics shown first
- **Expandable Sections**: Detailed information available on demand
- **Drill-Down**: Users can navigate from summary to detail views

## Deployment Patterns

### Docker Containerization

The system can be deployed using Docker containers:

- **docker-compose.yml**: Defines services (app, database, redis)
- **Environment Variables**: Configuration via environment
- **Volume Mounts**: Persistent storage for data

### Local Development

Developers can run the system locally:

- **setup_full_environment.sh**: Script to set up development environment
- **run_*.py Scripts**: Different entry points for different modes
- **Sample Data**: Provided for local testing

### Production Checks

Production readiness is verified with:

- **verify_production_readiness.py**: Script to check for common issues
- **PRODUCTION_CHECKLIST.md**: Manual verification steps
- **Monitoring**: Tracing and telemetry built into the system
