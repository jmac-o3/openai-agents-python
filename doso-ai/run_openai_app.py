"""
Startup script for running the DOSO AI application with OpenAI Assistants

This script starts both the FastAPI backend and Streamlit frontend
for the DOSO AI application that uses OpenAI Assistants API.
"""

import os
import subprocess
import sys
import time
from threading import Thread

# Configuration
FASTAPI_PORT = 8080
STREAMLIT_PORT = 8501


def run_fastapi():
    """Run the FastAPI backend server using the OpenAI implementation"""
    print(f"\n🚀 Starting FastAPI backend (OpenAI version) on port {FASTAPI_PORT}...")
    os.environ["PORT"] = str(FASTAPI_PORT)
    
    # Use uvicorn to run the OpenAI-based FastAPI app
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.main_openai:app", 
        "--host", "0.0.0.0", 
        "--port", str(FASTAPI_PORT),
        "--reload"
    ]
    
    fastapi_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    
    # Continuously display FastAPI logs
    for line in fastapi_process.stdout:
        print(f"[FastAPI] {line.strip()}")


def run_streamlit():
    """Run the Streamlit frontend using the OpenAI implementation"""
    print(f"\n🚀 Starting Streamlit frontend (OpenAI version) on port {STREAMLIT_PORT}...")
    
    # Configure environment for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = str(STREAMLIT_PORT)
    os.environ["API_URL"] = f"http://localhost:{FASTAPI_PORT}"
    
    # Wait briefly to ensure the API is running
    time.sleep(3)
    
    # Run the OpenAI-based Streamlit app
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app_openai.py",
        "--server.port", str(STREAMLIT_PORT),
    ]
    
    streamlit_process = subprocess.Popen(
        cmd, 
        cwd="doso-ai",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    
    # Continuously display Streamlit logs
    for line in streamlit_process.stdout:
        print(f"[Streamlit] {line.strip()}")


def main():
    """Main function to start both servers"""
    print("=" * 80)
    print("DOSO AI - OpenAI Assistants Version")
    print("=" * 80)
    print("\nThis script will start:")
    print(f"1. FastAPI backend on port {FASTAPI_PORT}")
    print(f"2. Streamlit frontend on port {STREAMLIT_PORT}")
    print("\nPress Ctrl+C to stop both servers.")
    
    # Start FastAPI in a separate thread
    fastapi_thread = Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    # Start Streamlit in the main thread
    run_streamlit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Shutting down servers... (This may take a few seconds)")
        print("\n✓ DOSO AI servers stopped.\n")
        sys.exit(0)
