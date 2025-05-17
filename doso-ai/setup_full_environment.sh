#!/bin/bash
# DOSO AI Self-Learning System - Full Environment Setup Script
# This script sets up the complete environment for the DOSO AI Self-Learning System

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}" || exit 1  # Change to script directory

echo "==================================================="
echo "DOSO AI Self-Learning System - Full Setup"
echo "==================================================="

# Find Python executable
if command -v python3 &> /dev/null; then
    PYTHON_EXE=python3
elif command -v python &> /dev/null; then
    PYTHON_EXE=python
else
    echo "❌ Error: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_EXE --version)
echo "Detected Python: $PYTHON_VERSION"

# Create necessary directories
echo "Creating required directories..."
mkdir -p "${SCRIPT_DIR}/data/run_log"
mkdir -p "${SCRIPT_DIR}/data/vector_store"
mkdir -p "${SCRIPT_DIR}/data/forecasts"

# Check/create initial .env file if it doesn't exist
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
    echo "Creating .env file template..."
    cat > "${SCRIPT_DIR}/.env" << EOF
# DOSO AI Environment Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/doso_ai
REDIS_URL=redis://localhost:6379/0

# Vector Store Configuration
VECTOR_STORE_PATH=./data/vector_store

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_TRACING=true

# Model Configuration
DEFAULT_FORECAST_MODEL=prophet
DEFAULT_LEARNING_MODEL=elasticnet
DEFAULT_OPTIMIZATION_TARGET=balanced
EOF
    echo "✅ Created .env template file"
    echo "   ⚠️ Please edit this file to add your API keys"
fi

# Check requirements.txt file
if [ ! -f "${SCRIPT_DIR}/requirements.txt" ]; then
    echo "❌ Error: requirements.txt file not found in ${SCRIPT_DIR}"
    exit 1
fi

# Install dependencies
echo "Installing required dependencies..."
${PYTHON_EXE} -m pip install --upgrade pip
${PYTHON_EXE} -m pip install -r "${SCRIPT_DIR}/requirements.txt"

# Make scripts executable
chmod +x "${SCRIPT_DIR}/run_production_learning_app.py" 2>/dev/null || true
chmod +x "${SCRIPT_DIR}/streamlit_minimal.py" 2>/dev/null || true
chmod +x "${SCRIPT_DIR}/verify_production_readiness.py" 2>/dev/null || true

echo "==================================================="
echo "✅ DOSO AI Self-Learning System setup complete"
echo "==================================================="
echo ""
echo "To run the system:"
echo "1. Edit .env file with your API keys"
echo "2. Launch the application:"
echo "   ${SCRIPT_DIR}/streamlit_minimal.py  # For direct launch"
echo "   ${SCRIPT_DIR}/run_production_learning_app.py  # For production mode with checks"
echo ""
echo "For more information, see README_LEARNING_SYSTEM.md and PRODUCTION_CHECKLIST.md"
echo "==================================================="
