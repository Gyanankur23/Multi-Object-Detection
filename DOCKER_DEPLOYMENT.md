# Production Docker Deployment Guide

## Quick Start (Live Presentation)

### 1. Build Docker Image
```bash
docker build -t object-detection .
```

### 2. Run Docker Container
```bash
docker run -d -p 8000:8000 --name object-detection object-detection
```

### 3. Check Health Status
```bash
curl http://localhost:8000/health
```

Expected response: `{"status":"healthy","model_loaded":true}`

### 4. Access Web Interface
Open browser: `http://localhost:8000`

### 5. Stop Container (After Presentation)
```bash
docker stop object-detection
docker rm object-detection
```

## Production Status
- **Web Interface**: ✅ Working
- **API Endpoints**: ✅ Working
- **Model Loading**: ✅ ONNX model loaded successfully
- **Health Check**: ✅ Returns healthy status with model loaded
- **Object Detection**: ✅ Fully functional

## Model Information
- **Model Type**: Custom trained YOLOv8 (ONNX format)
- **Classes**: bagpack, bottle, toothbrush, person, phone, book
- **Model Path**: runs/train/custom_model/weights/best.onnx
- **Input Size**: 640x640
- **Confidence Threshold**: 0.5
- **Docker Base**: Python 3.11-slim with latest ONNX runtime

## Using Docker Compose (Alternative)
```bash
docker-compose up -d
```

## Troubleshooting

### Port Already in Use
```bash
# Use different port
docker run -d -p 8001:8000 --name object-detection object-detection
```

### Check Container Logs
```bash
docker logs object-detection
```

### Rebuild if Changes Made
```bash
docker stop object-detection
docker rm object-detection
docker build -t object-detection .
docker run -d -p 8000:8000 --name object-detection object-detection
```

## Presentation Tips
1. **Pre-build image before presentation**: `docker build -t object-detection .`
2. **Test deployment**: Run through steps 1-4 before presentation
3. **Have backup**: Keep terminal ready with commands
4. **Show health check**: Demonstrate `curl http://localhost:8000/health`
5. **Show model info**: `curl http://localhost:8000/model_info`
6. **Demo detection**: Upload test images via web interface or use webcam
7. **Explain Docker benefits**: Production-ready, containerized deployment

## Features Available
- ✅ Production-ready Docker deployment
- ✅ Web interface with webcam support
- ✅ Image upload for object detection
- ✅ Real-time detection results
- ✅ Health check endpoint
- ✅ Model information endpoint
- ✅ Custom trained model with 6 classes
- ✅ ONNX runtime optimized inference
