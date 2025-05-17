#!/usr/bin/env python
"""
DOSO AI Self-Learning System Production Readiness Verification

This script performs comprehensive checks to ensure the self-learning system
is ready for production deployment. It verifies:

1. Required files exist and have correct permissions
2. Database schema is correctly configured
3. Dependencies are installed
4. Test suite passes
5. Configuration files have proper settings
"""

import os
import sys
import importlib
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Define colors for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = PROJECT_ROOT / "src"
TESTS_PATH = PROJECT_ROOT / "tests"
SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_data"
REQUIRED_PATHS = [
    SRC_PATH,
    SRC_PATH / "agents",
    SRC_PATH / "models",
    SRC_PATH / "utils",
    SRC_PATH / "workflow",
    SRC_PATH / "db",
    TESTS_PATH,
    SAMPLE_DATA_PATH
]


def print_header(message: str) -> None:
    """Print a formatted header message"""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{ENDC}")
    print(f"{BOLD}{BLUE}| {message}{' ' * (77 - len(message))}|{ENDC}")
    print(f"{BOLD}{BLUE}{'=' * 80}{ENDC}\n")


def print_status(message: str, status: str, details: Optional[str] = None) -> None:
    """Print a status message with appropriate coloring"""
    status_color = {
        "PASS": GREEN,
        "WARN": YELLOW,
        "FAIL": RED,
        "INFO": BLUE
    }.get(status, BLUE)
    
    print(f"{message:<60} [{status_color}{status}{ENDC}]")
    if details:
        print(f"  {details}")


def check_file_exists(path: Path) -> bool:
    """Check if a file exists"""
    return path.exists() and path.is_file()


def check_directory_exists(path: Path) -> bool:
    """Check if a directory exists"""
    return path.exists() and path.is_dir()


def check_executable(path: Path) -> bool:
    """Check if a file is executable"""
    return path.exists() and os.access(path, os.X_OK)


def check_required_files() -> bool:
    """
    Check if all required files exist and have the correct permissions
    Returns True if all checks pass, False otherwise
    """
    print_header("Checking Required Files and Directories")
    
    all_passed = True
    
    # Check project structure
    for path in REQUIRED_PATHS:
        if check_directory_exists(path):
            print_status(f"Directory: {path.relative_to(PROJECT_ROOT)}", "PASS")
        else:
            print_status(f"Directory: {path.relative_to(PROJECT_ROOT)}", "FAIL", 
                       "Required directory missing")
            all_passed = False
    
    # Check required files
    required_files = [
        (PROJECT_ROOT / "run_learning_app.py", True),
        (PROJECT_ROOT / "setup_full_environment.sh", True),
        (PROJECT_ROOT / "streamlit_doso_learning.py", False),
        (PROJECT_ROOT / "README_LEARNING_SYSTEM.md", False),
        (SAMPLE_DATA_PATH / "feedback_sample.csv", False),
        (SAMPLE_DATA_PATH / "inventory_sample.csv", False),
        (SAMPLE_DATA_PATH / "sales_sample.csv", False),
        (SAMPLE_DATA_PATH / "market_sample.csv", False),
        (TESTS_PATH / "test_learning_system.py", False),
    ]
    
    for file_path, should_be_executable in required_files:
        if check_file_exists(file_path):
            if should_be_executable and not check_executable(file_path):
                print_status(f"File: {file_path.relative_to(PROJECT_ROOT)}", "WARN", 
                           "File exists but is not executable")
                os.chmod(file_path, 0o755)
                print_status(f"Made {file_path.relative_to(PROJECT_ROOT)} executable", "INFO")
            else:
                print_status(f"File: {file_path.relative_to(PROJECT_ROOT)}", "PASS")
        else:
            print_status(f"File: {file_path.relative_to(PROJECT_ROOT)}", "FAIL", 
                       "Required file missing")
            all_passed = False
    
    return all_passed


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed
    Returns True if all checks pass, False otherwise
    """
    print_header("Checking Required Dependencies")
    
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "sqlalchemy",
        "pytest",
        "python-dotenv",
        "openai",
        "faiss-cpu",
        "scikit-learn"
    ]
    
    optional_packages = [
        "prophet",
        "pmdarima",
        "statsmodels",
        "redis",
        "psycopg2-binary"
    ]
    
    all_passed = True
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print_status(f"Required package: {package}", "PASS")
        except ImportError:
            print_status(f"Required package: {package}", "FAIL", 
                       f"Missing required package. Install with pip install {package}")
            all_passed = False
    
    for package in optional_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print_status(f"Optional package: {package}", "PASS")
        except ImportError:
            print_status(f"Optional package: {package}", "WARN", 
                      f"Missing optional package. Some features may not work. Install with pip install {package}")
    
    return all_passed


def run_unit_tests() -> bool:
    """
    Run the test suite
    Returns True if all tests pass, False otherwise
    """
    print_header("Running Unit Tests")
    
    if not check_file_exists(TESTS_PATH / "test_learning_system.py"):
        print_status("Unit tests", "FAIL", "Missing test_learning_system.py file")
        return False
    
    try:
        print("Running tests...")
        result = subprocess.run(
            ["python", "-m", "pytest", "-xvs", str(TESTS_PATH / "test_learning_system.py")],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print_status("Unit tests", "PASS", f"All tests passed")
            return True
        else:
            print_status("Unit tests", "FAIL", f"Some tests failed. See output above.")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            return False
    
    except Exception as e:
        print_status("Unit tests", "FAIL", f"Error running tests: {str(e)}")
        return False


def verify_docker_compose() -> bool:
    """
    Verify the docker-compose.yml file
    Returns True if verification passes, False otherwise
    """
    print_header("Verifying Docker Configuration")
    
    docker_compose_path = PROJECT_ROOT / "docker-compose.yml"
    
    if not check_file_exists(docker_compose_path):
        print_status("docker-compose.yml", "FAIL", "File does not exist")
        return False
    
    try:
        # Run docker-compose config to validate the file
        result = subprocess.run(
            ["docker-compose", "-f", str(docker_compose_path), "config"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_status("docker-compose.yml", "PASS", "Configuration is valid")
            
            # Check for required services
            if "postgres" in result.stdout and "redis" in result.stdout:
                print_status("Required services", "PASS", "Found postgres and redis services")
            else:
                print_status("Required services", "FAIL", "Missing required services (postgres and/or redis)")
                return False
            
            return True
        else:
            print_status("docker-compose.yml", "FAIL", f"Invalid configuration: {result.stderr}")
            return False
    
    except Exception as e:
        # Docker might not be installed
        print_status("docker-compose", "WARN", f"Could not validate: {str(e)}")
        print_status("docker-compose.yml exists", "PASS", "File exists but could not validate format")
        return True


def check_env_file() -> bool:
    """
    Check the .env file for required variables
    Returns True if all checks pass, False otherwise
    """
    print_header("Checking Environment Configuration")
    
    env_path = PROJECT_ROOT / ".env"
    required_vars = ["DATABASE_URL", "OPENAI_API_KEY"]
    optional_vars = ["REDIS_URL", "LOG_LEVEL", "VECTOR_STORE_PATH", 
                     "DEFAULT_FORECAST_MODEL", "DEFAULT_LEARNING_MODEL", 
                     "DEFAULT_OPTIMIZATION_TARGET"]
    
    # Create .env file if it doesn't exist
    if not check_file_exists(env_path):
        print_status(".env file", "WARN", "File does not exist, creating template")
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
        
        print_status(".env template created", "INFO", f"Please edit {env_path} with your actual values")
        return False
    
    # Check existing .env file
    with open(env_path, "r") as f:
        env_content = f.read()
    
    all_passed = True
    
    for var in required_vars:
        if var + "=" in env_content:
            value = env_content.split(var + "=")[1].split("\n")[0].strip()
            if var == "OPENAI_API_KEY" and value == "your_openai_api_key_here":
                print_status(f"Required variable: {var}", "FAIL", "Default placeholder value")
                all_passed = False
            else:
                print_status(f"Required variable: {var}", "PASS")
        else:
            print_status(f"Required variable: {var}", "FAIL", f"Missing from .env file")
            all_passed = False
    
    for var in optional_vars:
        if var + "=" in env_content:
            print_status(f"Optional variable: {var}", "PASS")
        else:
            print_status(f"Optional variable: {var}", "WARN", f"Missing from .env file")
    
    return all_passed


def verify_db_config() -> bool:
    """
    Verify database configuration
    Returns True if verification passes, False otherwise
    """
    print_header("Verifying Database Configuration")
    
    init_db_path = PROJECT_ROOT / "init-db.sql"
    
    if not check_file_exists(init_db_path):
        print_status("init-db.sql", "FAIL", "File does not exist")
        return False
    
    with open(init_db_path, "r") as f:
        sql_content = f.read()
    
    # Check for required tables
    required_tables = ["configurations", "forecasts", "feedback", "learning_cycles"]
    all_passed = True
    
    for table in required_tables:
        if f"CREATE TABLE" in sql_content and table in sql_content:
            print_status(f"Table definition: {table}", "PASS")
        else:
            print_status(f"Table definition: {table}", "FAIL", f"Missing table creation in init-db.sql")
            all_passed = False
    
    # Check for vector extension
    if "CREATE EXTENSION IF NOT EXISTS \"vector\"" in sql_content:
        print_status("Vector extension", "PASS")
    else:
        print_status("Vector extension", "WARN", "Vector extension not found in init-db.sql")
    
    return all_passed


def main() -> None:
    """Main function to run all verification checks"""
    print(f"{BOLD}{BLUE}DOSO AI Self-Learning System Production Readiness Verification{ENDC}")
    print(f"{BLUE}Running verification checks...{ENDC}")
    
    start_time = time.time()
    
    # Run all checks
    files_check = check_required_files()
    deps_check = check_dependencies()
    env_check = check_env_file()
    docker_check = verify_docker_compose()
    db_check = verify_db_config()
    test_check = run_unit_tests()
    
    # Print summary
    print_header("Verification Summary")
    
    all_passed = True
    
    checks = [
        ("Required files and directories", files_check),
        ("Dependencies", deps_check),
        ("Environment configuration", env_check),
        ("Docker configuration", docker_check),
        ("Database configuration", db_check),
        ("Unit tests", test_check)
    ]
    
    for check_name, result in checks:
        if result:
            print_status(check_name, "PASS")
        else:
            print_status(check_name, "FAIL")
            all_passed = False
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{BLUE}Verification completed in {elapsed_time:.2f} seconds{ENDC}")
    
    if all_passed:
        print(f"\n{GREEN}✓ All verification checks passed! The system is ready for production.{ENDC}")
        print(f"\n{BLUE}To start the full production system:{ENDC}")
        print(f"  1. {BOLD}Start the database infrastructure:{ENDC}")
        print(f"     cd {PROJECT_ROOT} && docker-compose up -d")
        print(f"  2. {BOLD}Load sample data into the database:{ENDC}")
        print(f"     python {PROJECT_ROOT}/load_sample_data.py")
        print(f"  3. {BOLD}Run the DOSO AI Self-Learning System:{ENDC}")
        print(f"     {PROJECT_ROOT}/run_learning_app.py")
    else:
        print(f"\n{RED}✗ Some verification checks failed. Please address the issues before deploying to production.{ENDC}")
        print(f"\n{BLUE}Run this script again after fixing the issues:{ENDC}")
        print(f"  python {__file__}")
    
    print(f"\n{BLUE}{'=' * 80}{ENDC}")


if __name__ == "__main__":
    main()
