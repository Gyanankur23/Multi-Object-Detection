# Deployment Guide

## Quick Deployment Options

### Option 1: Docker (Recommended for Production)

#### Local Docker Deployment
```bash
# Build and run with Docker
docker build -t yolo-tracker .
docker run -p 8000:8000 yolo-tracker
```

#### Docker Compose
```bash
docker-compose up -d
```

### Option 2: Render (Free Cloud Deployment)

1. Create a `render.yaml` file:
```yaml
services:
  - type: web
    name: yolo-tracker
    env: docker
    plan: free
    dockerfilePath: ./Dockerfile
    envVars:
      - key: PORT
        value: 8000
```

2. Push your code to GitHub
3. Connect your GitHub repo to Render.com
4. Render will automatically deploy

### Option 3: Railway (Free Cloud Deployment)

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login and deploy:
```bash
railway login
railway init
railway up
```

### Option 4: PythonAnywhere (Paid but Easy)

1. Create a PythonAnywhere account
2. Create a new web app
3. Upload your files
4. Install requirements in virtual environment
5. Configure WSGI to run `uvicorn main:app`

## Pre-built Deployment Files

The project includes:
- `Dockerfile` - For containerized deployment
- `docker-compose.yml` - For easy local deployment
- `requirements.txt` - Python dependencies

## Accessing the Deployed App

After deployment, access your app at:
- Local: http://localhost:8000
- Cloud: Your platform's provided URL (e.g., https://yolo-tracker.onrender.com)

## Requirements for Cloud Deployment

- GitHub repository with your code
- Account on chosen platform (Render/Railway are free)
- The YOLO model file (yolov8n.pt) must be included in your repo

## Notes

- The app uses the pre-trained YOLOv8n model (6.2MB)
- No training required - works out of the box
- Webcam access requires HTTPS on cloud platforms (most provide this)
- Free tiers may have limitations (RAM, CPU, bandwidth)
