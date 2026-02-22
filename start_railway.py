"""
Railway startup script - runs API server and trading bot together.
"""
import os
import signal
import subprocess
import sys
import time


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print("\nShutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    api_port = os.getenv("PORT") or os.getenv("DASHBOARD_INTERNAL_PORT", "8001")

    print("=" * 60)
    print("QUADRICK TRADING SYSTEM - RAILWAY DEPLOYMENT")
    print("=" * 60)
    print("\nStarting services...\n")

    print(f"Starting API server on port {api_port}...")
    api_process = subprocess.Popen(
        [sys.executable, "-u", "api_server.py"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    print("Starting trading bot...")
    bot_process = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    def cleanup():
        print("\nShutting down services...")
        api_process.terminate()
        bot_process.terminate()
        try:
            api_process.wait(timeout=5)
            bot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Services did not stop in time, forcing kill...")
            api_process.kill()
            bot_process.kill()
        print("Shutdown complete.")

    try:
        while True:
            if api_process.poll() is not None:
                print("API server stopped unexpectedly.")
                break
            if bot_process.poll() is not None:
                print("Trading bot stopped unexpectedly.")
                break
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cleanup()