# Quadrick — DeepSeek/Bybit Futures Trading System

Quadrick is an automated futures trading system designed for **scalping** on Bybit. It combines:

- A Python trading engine (Bybit integration, risk controls, execution, and strategy orchestration).
- A multi‑agent LLM decision layer (DeepSeek) with deterministic guardrails.
- A Next.js dashboard for monitoring and control.
- Supabase for shared state, telemetry, and memory/RAG.

This README covers **setup**, **configuration**, **running the bot**, **dashboard**, and **deployment**.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Project Layout](#project-layout)
- [Requirements](#requirements)
- [Quick Start (Local)](#quick-start-local)
- [Configuration (.env)](#configuration-env)
- [Running the Bot](#running-the-bot)
- [Running the API Server](#running-the-api-server)
- [Running the Dashboard](#running-the-dashboard)
- [Backtest / Walk‑Forward Validation](#backtest--walk-forward-validation)
- [Deployment Options](#deployment-options)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [Security & Risk Disclaimer](#security--risk-disclaimer)

---

## Architecture

**High‑level flow**

```
Bybit Market Data  ->  Trading Engine (Python)  ->  Decisions/Orders
                                 |
                                 v
                           Supabase (state + logs + RAG memory)
                                 |
                                 v
Next.js Dashboard  <---  API (FastAPI or Vercel functions)
```

**Core components**

- **Trading Bot** (`main.py`): orchestrates data fetch → analysis → decision → execution.
- **LLM Decision Layer** (`src/llm/` + Council agents): DeepSeek model + deterministic filters.
- **Risk & Execution** (`src/risk/`, `src/execution/`): SL/TP guardrails, ATR trailing, partial profit.
- **Supabase** (`src/database/`): shared state + memory retrieval.
- **Dashboard** (`quadend/`): monitoring, logs, decisions, positions, control panel.

---

## Key Features

- ✅ **Deterministic scalping pre‑selection** (filters watchlist before LLM evaluation).
- ✅ **Volatility‑aware risk scaling** within the 10–30% band.
- ✅ **ATR‑based SL/TP guardrails** and trailing stops.
- ✅ **Order‑flow + spread filters** for microstructure gating.
- ✅ **Supabase memory/RAG** with enriched trade context.
- ✅ **Walk‑forward backtest harness** for validation.

---

## Project Layout

```
.
├─ main.py                    # Trading bot entrypoint
├─ api_server.py              # FastAPI server for dashboard
├─ src/                       # Core trading system
│  ├─ agents/                 # LLM Council (Analyst/Strategist/Risk)
│  ├─ analysis/               # TA + order flow
│  ├─ exchange/               # Bybit client
│  ├─ execution/              # Trailing/partial profit
│  ├─ llm/                    # DeepSeek client + prompts
│  ├─ risk/                   # Risk manager + volatility scaling
│  └─ database/               # Supabase (RAG + state)
├─ quadend/                   # Next.js dashboard
├─ scripts/
│  └─ backtest_walk_forward.py
├─ requirements.txt
├─ VERCEL_DEPLOYMENT.md
└─ USE_PORT_8001.md
```

---

## Requirements

### Backend (Python)
- Python **3.10+**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Frontend (Dashboard)
- Node.js **18+**
- Install dependencies:
  ```bash
  cd quadend
  npm install
  ```

---

## Quick Start (Local)

### 1) Configure environment
Create a `.env` file in the project root:

```bash
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=true

DEEPSEEK_API_KEY=your_deepseek_key
LLM_PROVIDER=deepseek

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_key
```

> Tip: use Bybit **testnet** first.

### 2) Start the bot
```bash
python main.py
```

### 3) Start the API server (for dashboard)
```bash
python api_server.py
```

### 4) Start the dashboard
```bash
cd quadend
npm run dev
```

Dashboard: http://localhost:3000  
API: http://localhost:8000 (or set `PORT=8001` if 8000 is unavailable — see `USE_PORT_8001.md`)

---

## Configuration (.env)

Required:

```
BYBIT_API_KEY=
BYBIT_API_SECRET=
BYBIT_TESTNET=true|false
DEEPSEEK_API_KEY=
LLM_PROVIDER=deepseek
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

Optional:

```
BYBIT_BASE_URL_MAINNET=
BYBIT_BASE_URL_TESTNET=
BYBIT_WS_URL_MAINNET=
BYBIT_WS_URL_TESTNET=
```

---

## Running the Bot

```bash
python main.py
```

The bot:
- pulls data
- analyzes signals
- filters candidates
- makes a decision
- executes through Bybit

> For production, run the bot on a **VPS or Railway** (always‑on process).

---

## Running the API Server

```bash
python api_server.py
```

If port 8000 is already in use, set a different port before running:

```bash
PORT=8001 python api_server.py
```

The API server exposes endpoints for the dashboard:

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Bot status |
| `/api/balance` | GET | Account balance |
| `/api/positions` | GET | Active positions |
| `/api/decisions` | GET | Recent decisions |
| `/api/logs` | GET | System logs |
| `/api/control` | POST | Start/Stop/Pause |

---

## Running the Dashboard

```bash
cd quadend
npm run dev
```

Set `NEXT_PUBLIC_API_URL` if needed:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Backtest / Walk‑Forward Validation

The backtest harness uses OHLCV CSV data:

```bash
python scripts/backtest_walk_forward.py --csv /path/to/ohlcv.csv --train 1500 --test 500
```

Required CSV columns:
```
timestamp,open,high,low,close,volume
```

This script performs rolling windows and prints aggregate stats.

---

## Deployment Options

### 1) Bot (always‑on process)
Recommended:
- VPS (DigitalOcean/AWS/Hetzner)
- Railway
- Render

### 2) Dashboard + API
You can deploy the frontend + API to Vercel.  
See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for the full guide.

---

## Operational Notes

- **Testnet first**. Validate SL/TP behavior and execution.
- Use conservative sizing until performance is verified.
- Monitor Bybit API limits and latency.
- Watch logs and Supabase dashboards for system health.

---

## Troubleshooting

**Port 8000 conflicts**
Use port 8001:
```bash
export PORT=8001
python api_server.py
```
See [USE_PORT_8001.md](USE_PORT_8001.md).

**Dashboard shows no data**
- Check Supabase credentials
- Verify the bot is running and writing to Supabase
- Confirm API server is up

---

## Security & Risk Disclaimer

This system trades leveraged futures. Losses are possible and can be rapid.  
Only trade with funds you can afford to lose.

You are responsible for:
- API key security
- Risk configuration
- Deployment safety

---
