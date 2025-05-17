# DOSO AI Technical Context

## Technology Stack

### Core Technologies

1. **Python**: Primary programming language
   - Version: 3.9+
   - Key libraries: pandas, numpy, scikit-learn

2. **FastAPI**: Backend API framework
   - Enables high-performance async endpoints
   - OpenAPI documentation generation
   - Dependency injection system

3. **Streamlit**: User interface framework
   - Rapid application development
   - Interactive data visualization
   - Multiple page layout options

4. **PostgreSQL**: Primary database
   - pgvector extension for vector operations
   - Relational data storage
   - Complex query capabilities

5. **Redis**: Caching and messaging
   - Caching for performance optimization
   - Pub/sub for event-driven processing
   - Session management

6. **Docker**: Containerization
   - Multi-container deployment
   - Environment isolation
   - Simplified deployment and scaling

### AI/ML Technologies

1. **OpenAI API**: Core AI capabilities
   - Embedding generation for vector search
   - Text generation for recommendations
   - Assistants API for OpenAI-based implementation

2. **Time Series Forecasting**:
   - Prophet for sales forecasting
   - ARIMA for trend analysis
   - ETS for seasonality modeling

3. **Vector Search**:
   - Cosine similarity for semantic matching
   - pgvector for database-integrated vector operations
   - Clustering for pattern identification

4. **Machine Learning**:
   - Regression models for parameter optimization
   - Ensemble methods for combining forecasts
   - Reinforcement learning (planned) for optimization

### Frontend Technologies

1. **Plotly**: Interactive visualizations
   - Time series charts
   - Geospatial visualizations
   - Customizable dashboards

2. **Streamlit Components**:
   - Custom chart components
   - Interactive forms
   - Data upload widgets

## Development Environment

### Tools

1. **VS Code**: Primary IDE
   - Python extension
   - Streamlit extension
   - Docker extension

2. **Git**: Version control
   - GitHub repository
   - Branch protection rules
   - CI/CD integration

3. **Pytest**: Testing framework
   - Unit tests
   - Integration tests
   - Mocking and fixtures

4. **Alembic**: Database migration
   - Version-controlled schema changes
   - Forward and backward migrations
   - Integration with SQLAlchemy models

### Development Workflow

1. **Local Development**:
   - Local Python environment setup with `setup_full_environment.sh`
   - Docker-based services (PostgreSQL, Redis)
   - Sample data for testing

2. **Testing Strategy**:
   - Unit tests with pytest
   - Component tests for modules
   - Integration tests for workflows

3. **CI/CD Pipeline**:
   - Automated testing on commit
   - Code quality checks (flake8, mypy)
   - Deployment packaging

## Data Storage and Management

### Database Schema

1. **Core Tables**:
   - Inventory: Current inventory status
   - Sales: Historical sales data
   - Configurations: Vehicle configurations
   - Feedback: Recommendation outcomes
   - Market: Market analysis data

2. **Vector Storage**:
   - Embeddings stored in pgvector columns
   - External vector store for high-volume operations
   - Caching layer for frequent queries

### File Storage

1. **Sample Data Files**:
   - CSV format for easy import/export
   - JSON for configuration storage
   - Parquet for large dataset compression

2. **Report Generation**:
   - PDF report export
   - Excel spreadsheet generation
   - CSV data export

## API Integration

### Internal APIs

1. **Core Service Endpoints**:
   - `/api/inventory`: Inventory management
   - `/api/market`: Market analysis
   - `/api/recommendations`: Recommendation generation
   - `/api/feedback`: Feedback collection
   - `/api/learning`: Learning system management

2. **Authentication**:
   - JWT-based authentication
   - Role-based access control
   - API key management for service accounts

### External APIs

1. **OpenAI API**:
   - Embeddings API for vector generation
   - Completion API for text generation
   - Assistants API for thread management

2. **Future Integrations**:
   - Dealer Management System (DMS) integration
   - Factory ordering system integration
   - Market data provider integration

## Deployment Architecture

### Standard Deployment

1. **Components**:
   - FastAPI application container
   - PostgreSQL database
   - Redis cache
   - Vector store
   - Streamlit UI container

2. **Infrastructure**:
   - Docker Compose for development/testing
   - Kubernetes for production (planned)
   - Cloud provider flexibility

### OpenAI Assistants Deployment

1. **Components**:
   - FastAPI application container (lightweight)
   - OpenAI Assistants (cloud-based)
   - File storage (OpenAI-managed)
   - Streamlit UI container

2. **Infrastructure**:
   - Simplified deployment
   - Reduced infrastructure requirements
   - Higher OpenAI API usage

## Security Considerations

1. **Data Protection**:
   - Encryption at rest for sensitive data
   - TLS for all communications
   - Access control for all endpoints

2. **Authentication**:
   - Multi-factor authentication for admin access
   - JWT token expiration and rotation
   - Role-based permissions

3. **API Security**:
   - Rate limiting
   - Request validation
   - Input sanitization

## Technical Debt and Challenges

1. **Current Limitations**:
   - Performance optimizations needed for large datasets
   - Enhanced caching strategy required
   - Real-time processing improvements needed

2. **Technical Risks**:
   - OpenAI API changes and versioning
   - Vector store scaling challenges
   - Time series forecast accuracy

3. **Future Technical Considerations**:
   - Integration with additional data sources
   - Mobile optimized interfaces
   - Real-time notification system

## Monitoring and Observability

1. **Logging**:
   - Structured logging with contextual information
   - Log aggregation and search
   - Error tracking and alerting

2. **Metrics**:
   - System performance metrics
   - Business metrics tracking
   - API usage and performance

3. **Tracing**:
   - Request tracing across components
   - Performance bottleneck identification
   - Error source tracing
