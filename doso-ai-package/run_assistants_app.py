"""
Startup script for running the DOSO AI application with openai-agents Assistants SDK

This script starts both the FastAPI backend and Streamlit frontend
for the DOSO AI application using the openai-agents SDK for Assistants integration.
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
    """Run the FastAPI backend server using the openai-agents Assistants implementation"""
    print(f"\n🚀 Starting FastAPI backend (Assistants SDK version) on port {FASTAPI_PORT}...")
    os.environ["PORT"] = str(FASTAPI_PORT)
    
    # Use uvicorn to run the Assistants SDK-based FastAPI app
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.main_assistants:app", 
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
    """Run the Streamlit frontend for the Assistants SDK implementation"""
    print(f"\n🚀 Starting Streamlit frontend (Assistants SDK version) on port {STREAMLIT_PORT}...")
    
    # Configure environment for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = str(STREAMLIT_PORT)
    os.environ["API_URL"] = f"http://localhost:{FASTAPI_PORT}"
    
    # Wait briefly to ensure the API is running
    time.sleep(3)
    
    # Create a simplified version of the openai-based Streamlit app
    # that works with the assistants-based backend
    streamlit_app_path = os.path.join("doso-ai", "streamlit_app_openai.py")
    
    # Run the Streamlit app
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        streamlit_app_path,
        "--server.port", str(STREAMLIT_PORT),
    ]
    
    streamlit_process = subprocess.Popen(
        cmd, 
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
    print("DOSO AI - openai-agents Assistants SDK Version")
    print("=" * 80)
    print("\nThis script will start:")
    print(f"1. FastAPI backend on port {FASTAPI_PORT}")
    print(f"2. Streamlit frontend on port {STREAMLIT_PORT}")
    print("\nBefore running, make sure your OpenAI API key is set:")
    print("export OPENAI_API_KEY=your_api_key")
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
