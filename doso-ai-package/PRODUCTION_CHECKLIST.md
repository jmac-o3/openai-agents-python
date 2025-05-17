# DOSO AI Self-Learning System Production Checklist

This checklist helps ensure that the DOSO AI Self-Learning System is properly configured and ready for production use. Follow these steps to verify your setup before deployment.

## System Requirements

- [ ] Python 3.10+ installed
- [ ] Docker and Docker Compose installed (for database infrastructure)
- [ ] Minimum 4GB RAM, 8GB recommended
- [ ] At least 1GB free disk space

## Environment Setup

- [ ] Run `./doso-ai/setup_full_environment.sh` script
- [ ] Update `.env` file with proper API keys and configuration
  - [ ] Set `OPENAI_API_KEY` to valid OpenAI API key
  - [ ] Verify `DATABASE_URL` points to correct PostgreSQL instance
  - [ ] Set `REDIS_URL` if using Redis for caching
  - [ ] Configure `VECTOR_STORE_PATH` for embedding storage

## Database Configuration

- [ ] Start database services with `cd doso-ai && docker-compose up -d`
- [ ] Verify PostgreSQL service is running (`docker ps`)
- [ ] Verify Redis service is running (if using)
- [ ] Confirm database schema was created correctly
  - [ ] Table: configurations
  - [ ] Table: forecasts
  - [ ] Table: feedback
  - [ ] Table: learning_cycles
- [ ] Run `python doso-ai/load_sample_data.py` to populate sample data

## Dependencies

- [ ] Required Packages:
  - [ ] streamlit
  - [ ] pandas
  - [ ] numpy
  - [ ] plotly
  - [ ] faiss-cpu
  - [ ] scikit-learn
  - [ ] openai
  - [ ] sqlalchemy
  - [ ] pytest
  - [ ] python-dotenv
- [ ] Optional Packages:
  - [ ] prophet (for production forecasting)
  - [ ] pmdarima (for ARIMA forecasting)
  - [ ] statsmodels (for ETS forecasting)
  - [ ] redis
  - [ ] psycopg2-binary

## Testing

- [ ] Run verification script: `./doso-ai/verify_production_readiness.py`
- [ ] Run unit tests: `pytest -xvs doso-ai/tests/test_learning_system.py`
- [ ] Test in demo mode: `./doso-ai/run_learning_app.py`
- [ ] Test file uploads with sample data
- [ ] Test forecasting generation
- [ ] Test learning cycle
- [ ] Test semantic search

## Production Readiness

- [ ] Configure logging appropriately in `.env`
- [ ] Set up monitoring (if applicable)
- [ ] Schedule regular database backups
- [ ] Configure network access and security
- [ ] Set up SSL for Streamlit if exposed externally
- [ ] Document access procedures for end users
- [ ] Verify resource limits and scaling plan

## Deployment Steps

1. Start database infrastructure:
   ```bash
   cd doso-ai && docker-compose up -d
   ```

2. Verify database services are running:
   ```bash
   docker ps
   ```

3. Load sample data (if needed):
   ```bash
   python doso-ai/load_sample_data.py
   ```

4. Start the application:
   ```bash
   ./doso-ai/run_learning_app.py
   ```

5. Access the application in your browser:
   ```
   http://localhost:8501
   ```

## Troubleshooting

### Database Connection Issues

- Verify Docker containers are running
- Check database URL in `.env` file
- Ensure PostgreSQL extensions are enabled
- Check firewall settings

### Vector Store Issues

- Verify OpenAI API key is valid
- Check vector store path exists and is writable
- Ensure vector extension is loaded in PostgreSQL

### Forecasting Issues

- Verify prophet, pmdarima, or statsmodels are installed
- Check sales history data format
- Ensure sufficient historical data for forecasting

### App Startup Issues

- Check Streamlit installation is complete
- Verify all dependencies are installed
- Check port availability for Streamlit
- Check for errors in run_learning_app.py

## Production Maintenance

- Monitor API usage and costs
- Schedule regular database maintenance
- Update dependencies regularly
- Backup configuration and feedback data
- Monitor system performance
- Update models periodically

## Additional Resources

- [README_LEARNING_SYSTEM.md](./README_LEARNING_SYSTEM.md): Complete system documentation
- [setup_full_environment.sh](./setup_full_environment.sh): Environment setup script
- [verify_production_readiness.py](./verify_production_readiness.py): Automated readiness verification
- [load_sample_data.py](./load_sample_data.py): Sample data loading script
