# Deploying Quadend Frontend to Vercel

This guide will help you deploy the Quadend frontend to Vercel.

## Prerequisites

1. A Vercel account (sign up at https://vercel.com)
2. Git repository with your code (GitHub, GitLab, or Bitbucket)
3. Your backend API URL (can be updated later)

## Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to Git**
   - Make sure your code is pushed to GitHub, GitLab, or Bitbucket
   - The repository should include the `quadend` folder

2. **Import Project to Vercel**
   - Go to https://vercel.com/new
   - Import your Git repository
   - **Important**: Set the **Root Directory** to `quadend`
     - In the project settings, click "Configure Project"
     - Under "Root Directory", select `quadend`

3. **Configure Build Settings**
   - Vercel will auto-detect Next.js
   - Build Command: `npm run build` (should be auto-detected)
   - Output Directory: `.next` (should be auto-detected)
   - Install Command: `npm install` (should be auto-detected)

4. **Set Environment Variables**
   - Go to Project Settings → Environment Variables
   - Add the following variable:
     ```
     Name: NEXT_PUBLIC_API_URL
     Value: http://localhost:8000  (temporary, update after backend deployment)
     ```
   - Add it to all environments (Production, Preview, Development)

5. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete
   - Your frontend will be live at `https://your-project.vercel.app`

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Navigate to frontend folder**
   ```bash
   cd quadend
   ```

3. **Login to Vercel**
   ```bash
   vercel login
   ```

4. **Deploy**
   ```bash
   vercel
   ```
   - Follow the prompts
   - When asked about root directory, make sure you're in the `quadend` folder

5. **Set Environment Variables**
   ```bash
   vercel env add NEXT_PUBLIC_API_URL
   ```
   - Enter your backend API URL when prompted
   - Or set it in the Vercel dashboard

6. **Deploy to Production**
   ```bash
   vercel --prod
   ```

## After Deployment

### 1. Update Backend API URL

Once your backend is deployed (either on Vercel or another service):

1. Go to your Vercel project dashboard
2. Navigate to Settings → Environment Variables
3. Update `NEXT_PUBLIC_API_URL` to your backend URL:
   - If backend is on Vercel: `https://your-backend-project.vercel.app`
   - If backend is elsewhere: `https://your-backend-domain.com`
4. Redeploy the frontend (or it will auto-redeploy on next push)

### 2. Configure CORS on Backend

Make sure your backend API server allows requests from your Vercel frontend domain:

```python
# In your api_server.py or FastAPI app
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend.vercel.app",
        "http://localhost:3000"  # Keep for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. Verify Deployment

- Visit your Vercel deployment URL
- Check browser console for any API connection errors
- Test the dashboard functionality

## Troubleshooting

### Build Fails

- Check that you've set the Root Directory to `quadend`
- Ensure `package.json` exists in the `quadend` folder
- Check build logs in Vercel dashboard for specific errors

### API Connection Issues

- Verify `NEXT_PUBLIC_API_URL` is set correctly in Vercel environment variables
- Check that your backend is running and accessible
- Verify CORS is configured on the backend
- Check browser console for CORS or connection errors

### Environment Variables Not Working

- Make sure variable names start with `NEXT_PUBLIC_` for client-side access
- Redeploy after adding/updating environment variables
- Check that variables are added to the correct environment (Production/Preview/Development)

## Notes

- The frontend can communicate with a backend deployed elsewhere (doesn't have to be on Vercel)
- Environment variables prefixed with `NEXT_PUBLIC_` are exposed to the browser
- After updating environment variables, you need to redeploy for changes to take effect
- Vercel provides automatic HTTPS and custom domains




