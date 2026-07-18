# Multi-Object Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.1.0-orange)
![ONNX](https://img.shields.io/badge/ONNX-1.17%2B-purple)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)

**Production-grade real-time object detection system with end-to-end custom model training pipeline. Server-side rendering architecture with ONNX-optimized inference for maximum performance.**

---

## System Overview

This is a **complete, production-ready object detection framework** that demonstrates advanced computer vision engineering:

- **Custom YOLOv8 Model Training**: Full transfer learning pipeline with 200-epoch training
- **ONNX Optimization**: PyTorch → ONNX conversion for production deployment
- **Server-Side Rendering**: Backend inference with real-time annotation streaming
- **Professional Web Interface**: Modern frontend with real-time detection visualization
- **6 Custom Classes**: bagpack, bottle, toothbrush, person, phone, book
- **Complete MLOps Pipeline**: Data collection → Annotation → Training → Validation → Deployment

---

## Technical Architecture

### Server-Side Rendering Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Model Engine  │
│   (index.html)  │───▶│   (FastAPI)     │───▶│   (ONNX Runtime)│
│                 │    │                 │    │                 │
│ - Webcam Feed   │    │ - HTTP Endpoints│    │ - best.onnx     │
│ - Display UI    │    │ - Image Process │    │ - Inference     │
│ - Real-time     │    │ - Annotation    │    │ - Bounding Box  │
│   Updates       │    │   Drawing       │    │   Detection     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Data Flow:**
1. **Frontend**: Captures webcam video frames via MediaDevices API
2. **Transmission**: Sends frames to backend via HTTP POST `/detect` endpoint
3. **Backend**: Processes images with ONNX-optimized custom YOLO model
4. **Inference**: Runs real-time object detection with 6 custom classes
5. **Rendering**: Draws bounding boxes and labels on server side
6. **Response**: Streams annotated images back to frontend for display

---

## 📊 Complete Workflow Pipeline

### Phase 1: Data Collection & Annotation
```bash
# 1. Organize raw images by class
images/
├── bagpack/
├── bottle/
├── toothbrush/
├── person/
├── phone/
└── book/

# 2. Manual annotation with LabelImg (YOLO format)
pip install labelImg
labelImg images/bagpack/  # Draw bounding boxes, save as .txt
```

### Phase 2: Dataset Preparation
```bash
# 3. Convert COCO annotations to YOLO format (if needed)
python convert_coco_to_yolo.py

# 4. Organize dataset structure
python organize_dataset.py

# 5. Split dataset (70/20/10 train/val/test)
python split_dataset.py

# 6. Validate dataset quality
python validate_dataset.py
```

### Phase 3: Model Training
```bash
# 7. Train custom YOLOv8 model (200 epochs)
python train_custom_model.py

# Output: runs/train/custom_model/weights/best.pt
# Training metrics: Precision, Recall, mAP50, mAP50-95
```

### Phase 4: Model Optimization
```bash
# 8. Convert PyTorch model to ONNX for production
python convert_to_onnx.py

# Output: runs/train/custom_model/weights/best.onnx
# Benefits: Faster inference, cross-platform deployment
```

### Phase 5: Production Deployment
```bash
# 9. Start FastAPI server with ONNX model
python main.py

# Server loads best.onnx automatically
# Runs on http://0.0.0.0:8000
# Health check: http://localhost:8000/health
```

### Phase 6: Real-Time Detection
```
Access web interface: http://localhost:8000
- Click "Start Camera"
- Real-time detection with server-side rendering
- Bounding boxes drawn on backend
- Annotated frames streamed to frontend
```

---

## Custom Object Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | bagpack | Backpack detection |
| 1 | bottle | Bottle detection |
| 2 | toothbrush | Toothbrush detection |
| 3 | person | Person detection |
| 4 | phone | Mobile phone detection |
| 5 | book | Book detection |

---

## Training Configuration

**Hyperparameters:**
- **Base Model**: YOLOv8n (transfer learning)
- **Epochs**: 200 with early stopping (patience=50)
- **Batch Size**: 16 (GPU-optimized)
- **Image Size**: 640x640
- **Learning Rate**: 0.01 (cosine annealing)
- **Optimizer**: AdamW with weight decay
- **Data Augmentation**: Mosaic, Mixup, Flip, Rotate, HSV
- **Confidence Threshold**: 0.25
- **IOU Threshold**: 0.7

**Training Metrics (200 epochs):**
- **Final Precision**: 92.6%
- **Final Recall**: 85.5%
- **Final mAP50**: 66.1%
- **Final mAP50-95**: 64.5%

---

## Technology Stack

### Backend
- **FastAPI**: High-performance async web framework
- **Ultralytics YOLOv8**: State-of-the-art object detection
- **ONNX Runtime**: Cross-platform optimized inference
- **OpenCV**: Real-time image processing
- **NumPy**: High-performance numerical computing

### Frontend
- **HTML5/CSS3**: Modern responsive UI
- **JavaScript**: Client-side logic
- **Fetch API**: HTTP communication
- **MediaDevices API**: Webcam access

### Infrastructure
- **Docker**: Containerization support
- **Railway/Vercel**: Cloud deployment platforms
- **Git**: Version control

---

## Project Structure

```
multi-object-detection/
├── main.py                          # FastAPI server with ONNX inference
├── train_custom_model.py           # 200-epoch custom training pipeline
├── convert_to_onnx.py               # PyTorch → ONNX conversion
├── split_dataset.py                # 70/20/10 dataset splitting
├── validate_dataset.py              # Dataset quality validation
├── annotation.py                    # Annotation generation utilities
├── convert_coco_to_yolo.py         # COCO → YOLO format conversion
├── organize_dataset.py             # Dataset structure organization
├── data.yaml                        # Dataset configuration
├── requirements.txt                # Production dependencies
├── Dockerfile                       # Container configuration
├── vercel.json                      # Vercel deployment config
├── templates/
│   └── index.html                  # Server-side rendering interface
├── runs/train/custom_model/
│   ├── weights/
│   │   ├── best.pt                # Best PyTorch model
│   │   ├── best.onnx              # Optimized ONNX model
│   │   └── last.pt                # Last checkpoint
│   ├── results.csv                 # 200-epoch training metrics
│   ├── train_batch*.jpg           # Training visualizations
│   └── labels*.jpg                # Label distributions
└── images/                         # Raw dataset (gitignored)
    ├── bagpack/
    ├── bottle/
    ├── toothbrush/
    ├── person/
    ├── phone/
    └── book/
```

---

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start detection server
python main.py

# Access web interface
# http://localhost:8000
```

### Docker Deployment
```bash
# Build container
docker build -t object-detection .

# Run container
docker run -d -p 8001:8000 --name object-detection object-detection

# Health check
curl http://localhost:8001/health
```

---

## API Endpoints

### `POST /detect`
**Description**: Real-time object detection on uploaded image

**Request**: Multipart form data with image file

**Response**: JPEG image with bounding boxes and labels

**Performance**: ~50ms inference time per frame

### `GET /health`
**Description**: System health check

**Response**: 
```json
{
  "status": "healthy",
  "model_loaded": true,
  "ml_dependencies": true,
  "ultralytics_available": true
}
```

### `GET /model_info`
**Description**: Model information and classes

**Response**:
```json
{
  "model_type": "Custom YOLO",
  "model_path": "runs/train/custom_model/weights/best.onnx",
  "classes": {
    "0": "bagpack",
    "1": "bottle",
    "2": "toothbrush",
    "3": "person",
    "4": "phone",
    "5": "book"
  },
  "num_classes": 6
}
```

---

## Performance Metrics

**Inference Performance:**
- **ONNX Model**: ~50ms per frame (640x640)
- **PyTorch Model**: ~80ms per frame (640x640)
- **Optimization**: 37.5% speedup with ONNX

**Training Performance:**
- **Training Time**: ~2 hours (200 epochs, GPU)
- **Validation Time**: ~5 minutes per epoch
- **Dataset Size**: 600+ images (100+ per class)

---

## Security Features

- **Input Validation**: File type and size validation
- **Rate Limiting**: API endpoint protection
- **CORS Configuration**: Controlled cross-origin access
- **Error Handling**: Comprehensive exception management
- **Health Monitoring**: Real-time system status

---

## Dataset Requirements

**Minimum Requirements:**
- 100 images per class (600 total recommended)
- YOLO format annotations (.txt files)
- 70/20/10 train/validation/test split
- Diverse lighting conditions
- Multiple angles and perspectives
- Various backgrounds

**Quality Standards:**
- Precise bounding box annotations
- Consistent labeling conventions
- Balanced class distribution
- High-resolution images (640x640 minimum)

---

## Deployment Options

### Railway (Recommended)
- Automatic ONNX model loading
- Optimized for production
- Scalable infrastructure

### Vercel
- Static frontend hosting
- Railway backend integration
- Global CDN distribution

### Docker
- Containerized deployment
- Port mapping: 8000
- Volume mounting for models

---

## License

This project uses:
- **YOLOv8**: Ultralytics (AGPL-3.0 license)
- **FastAPI**: MIT License
- **ONNX Runtime**: MIT License
- **OpenCV**: Apache 2.0 License

---

## Key Achievements

- ✅ Custom 6-class object detection model
- ✅ 200-epoch training with 92.6% precision
- ✅ ONNX optimization for 37.5% speedup
- ✅ Server-side rendering architecture
- ✅ Real-time detection at 20+ FPS
- ✅ Complete MLOps pipeline
- ✅ Production-ready deployment

---

**Built with advanced computer vision engineering for production-grade object detection.**

## Authors and Contributors

This project was built with the combined team effort of:

*   **Gyanankur Baruah** - [@Gyanankur23](https://github.com)
*   **Huzeafa Khan**
*   **Hritik Poojari** - [@hrit06](https://github.com)
*   **Mansoor Ali**
*   **Rehaan Khan**

## Contribute to Repository

We welcome contributions and research proposals! To propose changes to the project, feel free to:

1.  **Fork** the repository.
2.  **Clone** and reproduce the environment.
3.  Submit a **Pull Request** for review.

*Note: Please ensure you mention **@Gyanankur23** when reproducing or publishing related work.*

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
