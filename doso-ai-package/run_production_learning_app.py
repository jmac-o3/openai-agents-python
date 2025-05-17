#!/usr/bin/env python
"""
DOSO AI Self-Learning System Production Launcher

This script launches the DOSO AI Self-Learning System with full production functionality.
It ensures the environment is properly set up before starting the application.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = DATA_DIR / "run_log"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
ENV_FILE = ROOT_DIR / ".env"
APP_FILE = ROOT_DIR / "streamlit_minimal.py"


def create_required_directories():
    """Create required directories if they don't exist"""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR / "forecasts", exist_ok=True)
    print(f"✓ Created required directories")


def check_environment():
    """Check if the environment is properly set up"""
    # Check if .env file exists
    if not ENV_FILE.exists():
        print(f"✗ Environment file not found: {ENV_FILE}")
        print("  Please run setup_full_environment.sh first")
        return False
    
    # Check for required environment variables
    with open(ENV_FILE, "r") as f:
        env_content = f.read()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY=your_openai_api_key_here" in env_content:
        print("✗ OPENAI_API_KEY not configured. Please edit .env file")
        return False
    
    print("✓ Environment configuration looks good")
    return True


def check_required_packages():
    """Check if required packages are installed"""
    required_packages = ["streamlit", "pandas", "plotly", "sqlalchemy", "faiss-cpu"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"✗ Missing required packages: {', '.join(missing_packages)}")
        print("  Please run: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All required packages installed")
    return True


def run_streamlit(port=8501, browser=False):
    """Run the Streamlit application"""
    print(f"Launching DOSO AI Self-Learning System on port {port}...")
    
    cmd = [
        "streamlit", "run", str(APP_FILE),
        "--server.port", str(port)
    ]
    
    if browser:
        cmd.append("--server.headless=false")
    else:
        cmd.append("--server.headless=true")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error launching application: {e}")
        return False
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="DOSO AI Self-Learning System Launcher")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the application on")
    parser.add_argument("--browser", action="store_true", help="Open browser automatically")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    args = parser.parse_args()
    
    print("=" * 70)
    print("DOSO AI Self-Learning System - Production Mode")
    print("=" * 70)
    
    # Create required directories
    create_required_directories()
    
    # Run checks if not skipped
    if not args.skip_checks:
        env_ok = check_environment()
        pkg_ok = check_required_packages()
        
        if not (env_ok and pkg_ok):
            print("\nPlease fix the issues above and try again.")
            print("You can run with --skip-checks to bypass these checks.")
            sys.exit(1)
    
    # All checks passed, run the application
    print("\nStarting DOSO AI Self-Learning System...")
    print(f"Access the application at http://localhost:{args.port}")
    print("Press Ctrl+C to stop the application")
    print("-" * 70)
    
    run_streamlit(port=args.port, browser=args.browser)


if __name__ == "__main__":
    main()
