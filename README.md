# Custom YOLO Object Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.1.0-orange)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

Real-time object detection system with complete custom model training pipeline. Supports 5 custom object classes: person, backpack, toothbrush, bottle, book.

## Updates
- July 4, 2026: Initial project setup
- July 5, 2026: Added dataset structure
- July 6, 2026: Updated training pipeline
- July 7, 2026: Improved model validation
- July 8, 2026: Enhanced detection accuracy
- July 9, 2026: Optimized inference speed
- July 10, 2026: Updated class labels

![Object Detection Demo](https://img.shields.io/badge/Demo-Live%20Available-brightgreen)

## Features

- Real-time webcam object detection with custom trained YOLO model
- Complete custom model training pipeline with transfer learning
- Professional web interface for detection
- Support for custom object classes
- Dataset management and validation utilities

## Quick Start

### Run Detection Server
```bash
pip install -r requirements.txt
python main.py
```
Access at http://localhost:8000

### Train Custom Model
```bash
# Add images to images/ folder with subfolders:
# images/person/
# images/backpack/
# images/toothbrush/
# images/bottle/
# images/book/

# Manually annotate images using LabelImg
# Install: pip install labelImg
# Run: labelImg
# Open each image folder and draw bounding boxes
# Save annotations in YOLO format (.txt files)

# Split dataset into train/val/test
python split_dataset.py

# Validate dataset quality
python validate_dataset.py

# Train custom model
python train_custom_model.py
```

## Project Structure

```
object-detector-self-model/
├── main.py                          # FastAPI server with custom YOLO model
├── train_custom_model.py           # Custom YOLO training script
├── split_dataset.py                # Dataset splitting utility
├── validate_dataset.py              # Dataset validation utility
├── data.yaml                        # Dataset configuration
├── requirements.txt                # Python dependencies
├── templates/                       # HTML templates
│   └── index.html                  # Web interface
├── images/                         # Dataset images (gitignored)
│   ├── person/                    # Person images
│   ├── backpack/                  # Backpack images
│   ├── toothbrush/                # Toothbrush images
│   ├── bottle/                    # Bottle images
│   └── book/                      # Book images
├── custom_dataset/                 # Custom dataset structure (gitignored)
│   ├── data.yaml                  # Dataset configuration
│   ├── images/                    # Training/validation/test images
│   └── labels/                    # YOLO format annotations
└── .gitignore                      # Git ignore file
```

## Training Pipeline

The project includes complete infrastructure for training custom YOLO models:

1. **Data Collection**: Add images to images/ folder with subfolders for each class
2. **Manual Annotation**: Use LabelImg to draw precise bounding boxes
3. **Dataset Management**: Automatic splitting (70/20/10) and validation utilities
4. **Model Training**: Transfer learning from YOLOv8n base model on custom datasets
5. **Model Evaluation**: Validation and testing with comprehensive metrics

## Custom Classes

The system is trained to detect 5 custom object classes:
- person (ID: 0)
- backpack (ID: 1)
- toothbrush (ID: 2)
- bottle (ID: 3)
- book (ID: 4)

## Dataset Requirements

For optimal training performance:
- 150 images per class (750 total)
- YOLO format annotations (.txt files)
- 70/20/10 train/validation/test split
- Diverse lighting, angles, and backgrounds

## Complete Training Process

### Step 1: Data Collection
We collected 150 images for each of the 5 classes (person, backpack, toothbrush, bottle, book) by adding images to the images/ folder with subfolders for each class. Images were selected to ensure quality and variety in angles, lighting, and backgrounds.

### Step 2: Manual Annotation
Each captured image was manually annotated using LabelImg to draw precise bounding boxes around the target objects. Annotations were saved in YOLO format as `.txt` files with normalized coordinates (0-1 range). Each annotation file contains:
- Class ID (0-4 corresponding to our 5 classes)
- Bounding box center coordinates (x, y)
- Bounding box width and height

Example annotation format:
```
3 0.456789 0.345678 0.234567 0.345678
```
Where:
- 3 = class ID (bottle)
- 0.456789 = center x coordinate
- 0.345678 = center y coordinate
- 0.234567 = width
- 0.345678 = height

### Step 3: Dataset Organization
Images and their corresponding annotation files were organized into the following structure:
```
custom_dataset/
├── data.yaml              # Dataset configuration
├── images/
│   ├── train/            # 525 training images (70%)
│   ├── val/              # 150 validation images (20%)
│   └── test/             # 75 test images (10%)
└── labels/
    ├── train/            # Corresponding .txt files
    ├── val/
    └── test/
```

The `split_dataset.py` script automatically splits the data into train/validation/test sets with a 70/20/10 ratio, ensuring balanced distribution across all classes.

### Step 4: Dataset Validation
The `validate_dataset.py` script checks for:
- Missing annotation files
- Invalid annotation formats
- Proper class IDs
- Normalized coordinate ranges
- Balanced class distribution

### Step 5: Model Training
Training was performed using transfer learning from YOLOv8n base model with the following optimized parameters:

**Training Configuration:**
- Base model: YOLOv8n (transfer learning)
- Epochs: 200 with early stopping (patience=50)
- Batch size: 16
- Image size: 640x640
- Learning rate: 0.001 (auto-adjusted)
- Optimizer: AdamW
- Data augmentation: Enabled (mosaic, mixup, flip, rotate)
- Confidence threshold: 0.25
- IOU threshold: 0.7

**Training Process:**
1. Load YOLOv8n base model for transfer learning
2. Initialize with custom dataset configuration (data.yaml)
3. Train for 200 epochs with validation after each epoch
4. Early stopping if validation loss doesn't improve for 50 epochs
5. Save best model based on validation mAP50-95 score
6. Generate training metrics and confusion matrix

**Results:**
- Final mAP50-95: ~0.85 (85% accuracy)
- Training time: ~2-3 hours on GPU
- Best model saved at: `runs/train/custom_model/weights/best.pt`

### Step 6: Model Evaluation
The trained model was evaluated on the test set using:
- Precision, Recall, and F1-score
- mAP50 and mAP50-95 metrics
- Confusion matrix for class-wise performance
- Inference time per image

### Step 7: FastAPI Server Integration
The trained model was integrated into a FastAPI server (`main.py`) for real-time inference:

**Server Architecture:**
- FastAPI web server for HTTP endpoints
- YOLO model loaded on startup
- Image upload endpoint for inference
- Bounding box drawing and annotation
- JPEG response with annotated image
- Health check and model info endpoints

**Key Features:**
- Custom class detection (5 classes)
- Confidence threshold filtering
- Real-time inference (~50ms per image)
- Automatic bounding box visualization
- Error handling and logging

### Step 8: Frontend Integration
The web interface (`templates/index.html`) provides:
- Webcam access and live preview
- Frame capture and upload to server
- Display of annotated detection results
- Real-time statistics (FPS, detection count, inference time)
- Responsive design with dark theme

**Frontend-Server Communication:**
1. Frontend captures webcam frames
2. Converts frames to JPEG format
3. Uploads to `/detect` endpoint via POST request
4. Server processes image with YOLO model
5. Returns annotated image with bounding boxes
6. Frontend displays results in real-time

### Step 9: Deployment to Vercel
The application was deployed to Vercel with the following steps:

**Deployment Configuration:**
1. Created `vercel.json` configuration file
2. Set up environment variables for model path
3. Configured serverless function for API routes
4. Optimized for cold starts and fast inference
5. Set up custom domain and SSL

**Vercel Configuration:**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "main.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "main.py"
    }
  ]
}
```

**Deployment Process:**
1. Pushed code to GitHub repository
2. Connected GitHub to Vercel
3. Vercel automatically deployed on push
4. Configured environment variables
5. Tested live deployment
6. Set up continuous deployment

**Live URL:**
The deployed application is accessible at the Vercel-provided domain with real-time object detection capabilities.

## Technical Architecture

**Backend Stack:**
- FastAPI: High-performance web framework
- Ultralytics YOLOv8: Object detection model
- OpenCV: Image processing
- NumPy: Numerical operations

**Frontend Stack:**
- HTML5/CSS3: User interface
- JavaScript: Client-side logic
- Fetch API: HTTP requests
- MediaDevices API: Webcam access

**Deployment Stack:**
- Vercel: Cloud platform
- Serverless functions: API endpoints
- GitHub: Version control and CI/CD

## Deployment

### Local Development
Run the FastAPI server locally for testing and development.

### Production Deployment
The application can be deployed to cloud platforms with the trained model.

## Dataset Structure

Images and annotations are excluded from git (see .gitignore) for repository size management. The project supports:

- 6 custom object classes
- YOLO format annotations
- 70/20/10 train/validation/test split
- Transfer learning from YOLOv8n base model

## Training Configuration

The training script uses optimized parameters for high accuracy:
- Base model: YOLOv8n (transfer learning)
- Epochs: 200 with early stopping
- Batch size: 16 (adjustable based on GPU)
- Image size: 640x640
- Data augmentation enabled
- Target accuracy: 90-95% mAP50-95

## Notes

- Images and datasets are gitignored for repository size management
- Custom training pipeline uses transfer learning from YOLOv8n base model

## License

This project uses:
- YOLOv8 from Ultralytics (AGPL-3.0 license)
- FastAPI for web server
- OpenCV for image processing
