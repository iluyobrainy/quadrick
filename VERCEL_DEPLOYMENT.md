# Deploying Both Frontend and Backend to Vercel

Yes! You can deploy both your frontend and backend API to Vercel. Here's how:

## Architecture Overview

- **Frontend**: Next.js app in `quadend/` folder → Deploys as Vercel Next.js app
- **Backend API**: Python serverless functions in `api/` folder → Deploys as Vercel Python functions
- **Trading Bot** (`main.py`): Must run separately (VPS, Railway, etc.) - writes data to Supabase
- **Shared State**: Supabase database stores all bot state, logs, decisions, etc.

## How It Works

1. Your trading bot (`main.py`) runs continuously on a separate server/VPS
2. The bot writes all data (status, balance, positions, logs, decisions) to Supabase
3. Vercel API functions (`api/*.py`) read from Supabase
4. Frontend calls Vercel API endpoints
5. Everything is connected through Supabase as the central database

## Deployment Steps

### Step 1: Ensure Supabase is Set Up

Make sure your Supabase database has these tables:
- `bot_status` - Stores bot status and balance info
- `active_positions` - Stores current trading positions
- `system_logs` - Stores system logs
- `decisions` - Stores trading decisions

### Step 2: Update Vercel Configuration

The `vercel.json` in the root is already configured for this setup. It should:

1. Deploy `api/` folder as Python serverless functions
2. Deploy `quadend/` folder as Next.js app
3. Route `/api/*` requests to Python functions
4. Route all other requests to Next.js

### Step 3: Set Environment Variables in Vercel

Go to your Vercel project → Settings → Environment Variables and add:

**Required:**
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
# OR use service role key for full access:
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

**For Frontend:**
```
NEXT_PUBLIC_API_URL=
# Leave empty or set to your Vercel domain (e.g., https://your-project.vercel.app)
# The frontend will use relative URLs, so this can be empty
```

### Step 4: Deploy to Vercel

1. Push your code to GitHub/GitLab/Bitbucket
2. Import project to Vercel (if not already)
3. Vercel will auto-detect and build both:
   - The Next.js frontend from `quadend/`
   - The Python API functions from `api/`
4. Deploy!

### Step 5: Update Frontend API Client

The frontend should use relative URLs or your Vercel domain:

```typescript
// In quadend/src/lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
// If empty, it will use relative URLs (same domain as frontend)
```

### Step 6: Configure Your Bot to Write to Supabase

Your `main.py` bot needs to write data to Supabase instead of HTTP POST to localhost.

The bot already uses `SupabaseBridge` and `SupabaseClient` - make sure:
1. Bot has Supabase credentials in `.env`
2. Bot writes status, balance, positions, logs, and decisions to Supabase
3. Bot reads control commands from Supabase (for start/stop/pause)

## Important Notes

### What Runs on Vercel:
✅ Frontend (Next.js)  
✅ API endpoints (Python serverless functions)  
❌ Trading bot (`main.py`) - **Cannot run on Vercel** (needs to run continuously)

### Trading Bot Must Run Separately:
The trading bot (`main.py`) is a long-running process that needs to:
- Run 24/7
- Make trading decisions continuously
- Connect to Bybit API
- Use WebSocket connections

**Options for running the bot:**
1. **VPS** (DigitalOcean, AWS EC2, etc.) - Best for reliability
2. **Railway** - Easy deployment, good for bots
3. **Render** - Web service (free tier available)
4. **Your local machine** - For testing (not production)

### API Endpoints Available:

All endpoints from `api_server.py` are available:
- `GET /api/status` - Bot status
- `GET /api/balance` - Account balance
- `GET /api/positions` - Active positions
- `GET /api/decisions` - Recent decisions
- `GET /api/logs` - System logs
- `POST /api/control` - Start/Stop/Pause bot (updates Supabase)
- `GET /api/settings` - Settings (read-only)
- `GET /api/performance` - Performance metrics

### How Control Works:

1. Frontend calls `POST /api/control` with action ("start"/"stop"/"pause")
2. Vercel API function updates `bot_status` table in Supabase
3. Your bot reads from `bot_status` table periodically
4. Bot changes its mode based on the status

## Testing

After deployment:

1. Visit your Vercel frontend URL
2. Check browser console for API calls
3. Verify data is loading from Supabase
4. Test control buttons (start/stop/pause)

## Troubleshooting

### API Returns Empty Data

- Check Supabase credentials in Vercel environment variables
- Verify bot is writing data to Supabase
- Check Supabase dashboard to see if tables have data

### CORS Errors

- API functions already have CORS enabled (allows all origins)
- If issues persist, check Vercel function logs

### Bot Not Responding to Controls

- Make sure bot is reading from Supabase `bot_status` table
- Check bot logs to see if it's reading status updates
- Verify Supabase connection in bot

## Benefits of This Setup

✅ **Single Platform**: Frontend and API on Vercel  
✅ **Scalable**: Serverless functions scale automatically  
✅ **Cost Effective**: Vercel free tier is generous  
✅ **No ngrok Needed**: Everything is publicly accessible  
✅ **Fast**: Edge functions close to users  
✅ **Reliable**: Vercel has great uptime  

## Next Steps

1. Deploy to Vercel using the steps above
2. Set up your bot to run on a VPS or Railway
3. Configure bot to write to Supabase
4. Test the full system!

