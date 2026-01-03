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
    
    # Run both API server and trading bot
    print("🌐 Starting API Server on port 8001...")
    api_process = subprocess.Popen(
        [sys.executable, "-u", "api_server.py"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    
    print("🤖 Starting Trading Bot...")
    bot_process = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    
    try:
        # Keep the main script alive and monitor subprocesses
        while True:
            if api_process.poll() is not None:
                print("❌ API Server stopped unexpectedly!")
                break
            if bot_process.poll() is not None:
                print("❌ Trading Bot stopped unexpectedly!")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Interrupt received, shutting down...")
    finally:
        api_process.terminate()
        bot_process.terminate()
        api_process.wait()
        bot_process.wait()

