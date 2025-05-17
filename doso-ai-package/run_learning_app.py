#!/usr/bin/env python
"""
DOSO AI Self-Learning System Launcher

This script launches the DOSO AI Self-Learning System Streamlit application.
It automatically detects if database connections are available and launches
the application in production or demo mode accordingly.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Define colors for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Get the project root directory
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Determine the correct path for the project
if SCRIPT_DIR.name == 'doso-ai':
    PROJECT_ROOT = SCRIPT_DIR
else:
    # We might be running from outside the doso-ai directory
    PROJECT_ROOT = SCRIPT_DIR / "doso-ai"
    if not PROJECT_ROOT.exists():
        # We might be at project root with doso-ai as subdirectory
        PROJECT_ROOT = SCRIPT_DIR.parent / "doso-ai"
        if not PROJECT_ROOT.exists():
            print(f"{RED}Error: Could not find doso-ai directory{ENDC}")
            sys.exit(1)

# Define paths
STREAMLIT_APP = PROJECT_ROOT / "streamlit_doso_learning.py"
ENV_FILE = PROJECT_ROOT / ".env"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt" 
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


def print_header(message):
    """Print a formatted header message"""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{ENDC}")
    print(f"{BOLD}{BLUE}| {message}{' ' * (77 - len(message))}|{ENDC}")
    print(f"{BOLD}{BLUE}{'=' * 80}{ENDC}\n")


def check_database_connection():
    """Check if database connection is available"""
    try:
        # Try to import sqlalchemy
        import sqlalchemy
        from sqlalchemy import create_engine, text
        from dotenv import load_dotenv
        
        # Load environment variables
        env_path = PROJECT_ROOT / ".env"
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
        
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print(f"{YELLOW}No DATABASE_URL found in environment. Running in demo mode.{ENDC}")
            return False
        
        # Try to connect to the database
        print(f"Checking database connection to: {database_url}")
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print(f"{GREEN}✓ Database connection successful!{ENDC}")
        return True
    
    except Exception as e:
        print(f"{YELLOW}Database connection failed: {e}{ENDC}")
        print(f"{YELLOW}Running in demo mode.{ENDC}")
        return False


def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "sqlalchemy",
        "python-dotenv",
        "faiss-cpu"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"{RED}Missing required packages: {', '.join(missing_packages)}{ENDC}")
        print(f"Install them with: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_optional_packages():
    """Check if optional packages are installed"""
    optional_packages = [
        "prophet",
        "pmdarima",
        "statsmodels",
        "redis",
        "psycopg2-binary",
        "openai"
    ]
    
    missing_packages = []
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"{YELLOW}Some optional packages are missing: {', '.join(missing_packages)}{ENDC}")
        print(f"{YELLOW}Some advanced features may not be available.{ENDC}")
        print(f"Install them with: pip install {' '.join(missing_packages)}")
    
    return True


def check_sample_data():
    """Check if sample data files exist"""
    required_files = [
        PROJECT_ROOT / "sample_data" / "feedback_sample.csv",
        PROJECT_ROOT / "sample_data" / "inventory_sample.csv",
        PROJECT_ROOT / "sample_data" / "sales_sample.csv",
        PROJECT_ROOT / "sample_data" / "market_sample.csv"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"{YELLOW}Some sample data files are missing: {ENDC}")
        for file_path in missing_files:
            print(f"{YELLOW}- {file_path.relative_to(PROJECT_ROOT)}{ENDC}")
        print(f"{YELLOW}Demo mode will use simulated data.{ENDC}")
    else:
        print(f"{GREEN}✓ All sample data files found!{ENDC}")


def check_docker_services():
    """Check if Docker services are running"""
    try:
        # Try to check Docker services
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Check for PostgreSQL and Redis containers
            services = result.stdout.strip().split("\n")
            postgres_running = any(name for name in services if "postgres" in name.lower())
            redis_running = any(name for name in services if "redis" in name.lower())
            
            if postgres_running:
                print(f"{GREEN}✓ PostgreSQL service is running!{ENDC}")
            else:
                print(f"{YELLOW}PostgreSQL service is not running.{ENDC}")
            
            if redis_running:
                print(f"{GREEN}✓ Redis service is running!{ENDC}")
            else:
                print(f"{YELLOW}Redis service is not running.{ENDC}")
            
            return postgres_running and redis_running
        else:
            print(f"{YELLOW}Could not check Docker services: {result.stderr}{ENDC}")
            return False
    
    except Exception as e:
        print(f"{YELLOW}Error checking Docker services: {e}{ENDC}")
        return False


def create_env_file():
    """Create a default .env file if it doesn't exist"""
    env_path = PROJECT_ROOT / ".env"
    
    if not os.path.exists(env_path):
        print(f"{YELLOW}Creating default .env file at {env_path}{ENDC}")
        
        with open(env_path, "w") as f:
            f.write("# DOSO AI Environment Configuration\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
            f.write("# Database Configuration\n")
            f.write("DATABASE_URL=postgresql://postgres:postgres@localhost:5432/doso_ai\n")
            f.write("REDIS_URL=redis://localhost:6379/0\n\n")
            f.write("# Vector Store Configuration\n")
            f.write("VECTOR_STORE_PATH=./data/vector_store\n\n")
            f.write("# Logging Configuration\n")
            f.write("LOG_LEVEL=INFO\n")
            f.write("ENABLE_TRACING=true\n\n")
            f.write("# Model Configuration\n")
            f.write("DEFAULT_FORECAST_MODEL=prophet\n")
            f.write("DEFAULT_LEARNING_MODEL=elasticnet\n")
            f.write("DEFAULT_OPTIMIZATION_TARGET=balanced\n")
        
        print(f"{YELLOW}Default .env file created. Please edit it with your actual values.{ENDC}")
        return False
    
    return True


def run_streamlit_app(port=8501, browser=False):
    """Run the Streamlit application"""
    cmd = ["streamlit", "run", str(STREAMLIT_APP)]
    
    # Add port argument if specified
    if port != 8501:
        cmd.extend(["--server.port", str(port)])
    
    # Add browser argument if specified
    if browser:
        cmd.append("--server.headless=false")
    else:
        cmd.append("--server.headless=true")
    
    try:
        # Run the Streamlit app
        print(f"\n{GREEN}Starting DOSO AI Self-Learning System...{ENDC}")
        print(f"URL: http://localhost:{port}")
        print(f"\n{BLUE}Press Ctrl+C to stop the application{ENDC}\n")
        
        process = subprocess.Popen(cmd)
        
        # Wait for the process to complete
        process.wait()
    
    except KeyboardInterrupt:
        print(f"\n{BLUE}Stopping DOSO AI Self-Learning System...{ENDC}")
        process.terminate()
    
    except Exception as e:
        print(f"{RED}Error running Streamlit: {e}{ENDC}")
        sys.exit(1)


def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='DOSO AI Self-Learning System Launcher')
    parser.add_argument('--port', type=int, default=8501, help='Port to run the Streamlit app on')
    parser.add_argument('--browser', action='store_true', help='Automatically open browser')
    parser.add_argument('--setup', action='store_true', help='Run full environment setup')
    args = parser.parse_args()
    
    print_header("DOSO AI Self-Learning System Launcher")
    
    # Check if Streamlit app exists
    if not os.path.exists(STREAMLIT_APP):
        print(f"{RED}Error: Streamlit app not found at {STREAMLIT_APP}{ENDC}")
        sys.exit(1)
    
    # Run full setup if requested
    if args.setup:
        setup_script = PROJECT_ROOT / "setup_full_environment.sh"
        if os.path.exists(setup_script):
            print(f"{BLUE}Running full environment setup...{ENDC}")
            subprocess.run(["bash", str(setup_script)])
        else:
            print(f"{RED}Error: Setup script not found at {setup_script}{ENDC}")
            sys.exit(1)
    
    # Check required packages
    if not check_required_packages():
        print(f"{RED}Please install the required packages and try again.{ENDC}")
        sys.exit(1)
    
    # Check optional packages
    check_optional_packages()
    
    # Create .env file if it doesn't exist
    create_env_file()
    
    # Check sample data
    check_sample_data()
    
    # Check Docker services
    check_docker_services()
    
    # Check database connection
    production_mode = check_database_connection()
    
    # Print mode
    if production_mode:
        print(f"\n{GREEN}Running in PRODUCTION mode with database connection!{ENDC}")
    else:
        print(f"\n{YELLOW}Running in DEMO mode (no database connection).{ENDC}")
        print(f"{YELLOW}To enable production mode, make sure to set up the database and update the .env file.{ENDC}")
    
    # Run the Streamlit app
    run_streamlit_app(port=args.port, browser=args.browser)


if __name__ == "__main__":
    main()
