# Quadrick (Railway Backend + Vercel Frontend)

Quadrick is an automated Bybit futures trading system with:
- Python trading runtime (`main.py`)
- FastAPI dashboard backend (`api_server.py`)
- Next.js dashboard frontend (`quadend/`)
- Supabase persistence (trades, decisions, RAG memory, bot status)

## Runtime Architecture
- Backend runtime: Railway/VPS (always-on process)
- Frontend hosting: Vercel (frontend-only, deploy `quadend`)
- Dashboard bridge: bot pushes to local FastAPI internal endpoints

## Quick Start (Local)

1. Create `.env` from template:
```bash
copy config\\env.example .env
```

2. Install Python deps:
```bash
pip install -r requirements.txt
```

3. Start backend API:
```bash
python api_server.py
```

4. Start bot:
```bash
python main.py
```

5. Start frontend:
```bash
cd quadend
npm install
npm run dev
```

## Live Trading Safety

- `BYBIT_TESTNET=true` is recommended by default.
- Live orders are blocked unless:
  - `BYBIT_TESTNET=false`, and
  - `ALLOW_LIVE_TRADING=true`

## Supabase Setup and Reset

1. Backup current tables:
```bash
python database/backup_supabase.py
```

2. Apply canonical reset:
- Run `migrations/004_full_reset.sql`
- Then run `migrations/005_seed_state.sql`

3. For fresh setup, `database/schema.sql` and `supabase_setup.sql` mirror the same canonical model.

## Vercel Frontend Deployment

Deploy only `quadend` to Vercel:
- Set project Root Directory to `quadend`
- Set env var in Vercel:
  - `NEXT_PUBLIC_API_URL=https://<your-railway-backend-domain>`

Backend stays on Railway/VPS and must allow CORS for your frontend domain.
Set backend env var `VERCEL_FRONTEND_URL=https://<your-frontend>.vercel.app` in production.
