# Quadrick Dashboard

Modern Next.js dashboard for monitoring and controlling the Quadrick AI Trading System.

## Features

- 📊 **Real-time Balance** - Live USDT balance, PnL, milestone progress
- 🤖 **DeepSeek Decisions** - All trading decisions with confidence and reasoning
- 📈 **Active Positions** - Live position tracking with SL/TP levels
- 📝 **System Logs** - Streaming logs with level filters
- ⚙️ **Settings** - API key management and trading parameters
- 🎮 **Control Panel** - Start/Stop/Pause trading

## Quick Start

### 1. Install Dependencies

```bash
cd quadend
npm install
```

### 2. Start the API Server (from parent directory)

```bash
# Install FastAPI if not already installed
pip install fastapi uvicorn

# Start the API server
python api_server.py
```

The API will run on `http://localhost:8000`

### 3. Start the Dashboard

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

## Tech Stack

- **Frontend**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS 4
- **Icons**: Lucide React
- **Data Fetching**: SWR
- **Backend**: FastAPI (Python)

## Color Theme

- Background: Black (#0a0a0a)
- Cards: Dark Gray (#111111)
- Accent: Blue (#3b82f6)
- Success: Green (#22c55e)
- Error: Red (#ef4444)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Bot status |
| `/api/balance` | GET | Account balance |
| `/api/positions` | GET | Active positions |
| `/api/decisions` | GET | Recent decisions |
| `/api/logs` | GET | System logs |
| `/api/logs/stream` | GET | SSE log stream |
| `/api/control` | POST | Start/Stop/Pause |
| `/api/settings` | GET/POST | Settings |

## Environment Variables

Create `.env.local` in the quadend folder:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
