# Quadend Deployment (Vercel Frontend Only)

This deployment guide is for the `quadend` Next.js app only.
Trading runtime and API stay on Railway/VPS.

## Architecture

- Frontend: Vercel (`quadend`)
- Backend API + trading bot: Railway/VPS (`api_server.py` + `main.py`)

## 1) Deploy Frontend to Vercel

1. Import your repository in Vercel.
2. Set **Root Directory** to `quadend`.
3. Keep default Next.js build settings.
4. Add environment variable:

```
NEXT_PUBLIC_API_URL=https://<your-railway-backend-domain>
```

5. Deploy.

## 2) Backend CORS Requirement

Set backend env var so FastAPI allows your Vercel domain:

```
VERCEL_FRONTEND_URL=https://<your-frontend>.vercel.app
```

If not set, backend falls back to permissive `*` CORS for development.

## 3) Verify End-to-End

From deployed frontend, confirm these endpoints return data:

- `/api/status`
- `/api/balance`
- `/api/positions`
- `/api/decisions`
- `/api/logs`

## Notes

- Do not deploy trading serverless routes to Vercel in this architecture.
- Keep secret keys on backend only.
