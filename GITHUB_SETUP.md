# GitHub Setup and Deployment

## Step 1: Initialize Git Repository

```bash
cd video-tracker
git init
```

## Step 2: Add Files to Git

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: Video Object Tracker with YOLOv8"
```

## Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `video-tracker`
3. Description: Real-time multi-object tracking in video using YOLOv8
4. Make it Public (for free Render deployment)
5. Click "Create repository"

## Step 5: Connect Local to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/video-tracker.git
git branch -M main
git push -u origin main
```

## Step 6: Deploy to Render

### Option A: Render.com (Free)

1. Go to https://render.com
2. Sign up/login with GitHub
3. Click "New Web Service"
4. Connect your GitHub account
5. Select `video-tracker` repository
6. Render will detect Dockerfile automatically
7. Click "Create Web Service"
8. Wait for deployment (2-3 minutes)
9. Access your app at: `https://video-tracker.onrender.com`

### Option B: Railway.app (Free)

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

## Important Notes

- The YOLO model (yolov8n.pt) will be downloaded automatically on first run
- Free tiers have limitations but work for this application
- Webcam access requires HTTPS (provided by cloud platforms)
- The app uses pre-trained COCO model - no training needed

## Troubleshooting

### Git Push Fails
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### Render Deployment Fails
- Check Render dashboard for error logs
- Ensure Dockerfile is present
- Verify requirements.txt is complete

### Model Not Loading
- The model downloads automatically on first run
- Check logs for download errors
- Ensure internet connectivity on deployment
