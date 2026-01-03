"""
Railway startup script - Runs both API server and trading bot
"""
import subprocess
import sys
import os
import signal
import time
from pathlib import Path

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    print("="*60)
    print("QUADRICK TRADING SYSTEM - RAILWAY DEPLOYMENT")
    print("="*60)
    print("\n🚀 Starting services...\n")
    
    # Option 1: Run API server only (recommended for Railway)
    # The bot will connect to API server via HTTP or Supabase
    print("🌐 Starting API Server...")
    os.execvp("python", ["python", "api_server.py"])
    
    # Option 2: If you want to run both in separate processes
    # (Uncomment if needed, but Railway works better with one process)
    # api_process = subprocess.Popen([sys.executable, "api_server.py"])
    # bot_process = subprocess.Popen([sys.executable, "main.py"])
    # 
    # try:
    #     api_process.wait()
    #     bot_process.wait()
    # except KeyboardInterrupt:
    #     api_process.terminate()
    #     bot_process.terminate()
    #     api_process.wait()
    #     bot_process.wait()

