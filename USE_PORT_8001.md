# Use Port 8001 Instead (Quick Fix)

Since port 8000 seems stuck, let's use port 8001 temporarily.

## Quick Fix: Use Port 8001

1. Set environment variable before running:
   ```powershell
   $env:PORT=8001
   python api_server.py
   ```

2. Then use ngrok with port 8001:
   ```powershell
   ngrok http 8001
   ```

This will work exactly the same, just on a different port!

