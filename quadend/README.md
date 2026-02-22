# Quadrick Dashboard (`quadend`)

Next.js frontend for monitoring and controlling the Quadrick trading backend.

## Local Development

1. Install dependencies:

```bash
cd quadend
npm install
```

2. Set API URL in `quadend/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8001
```

3. Run frontend:

```bash
npm run dev
```

4. Run backend API from repo root:

```bash
python api_server.py
```

## Backend API Endpoints Used

- `GET /api/status`
- `GET /api/balance`
- `GET /api/positions`
- `GET /api/decisions`
- `GET /api/logs`
- `GET /api/logs/stream`
- `POST /api/control`
- `GET/POST /api/settings`

## Production

- Deploy this folder (`quadend`) to Vercel.
- Set `NEXT_PUBLIC_API_URL=https://<railway-backend-domain>`.
- Backend runtime remains on Railway/VPS.
