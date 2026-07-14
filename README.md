# Custom YOLO Object Detection System
 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.1.0-orange)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)

Real-time object detection system with custom model training pipeline. Supports 6 custom object classes: bagpack, bottle, toothbrush, person, phone, book.

## Features

- Real-time webcam object detection with custom trained YOLO model
- Complete custom model training pipeline with transfer learning
- Professional web interface for detection
- Support for 6 custom object classes
- Dataset management and validation utilities
- ONNX model support for optimized deployment

## Quick Start

### Run Detection Server (Local)
```bash
pip install -r requirements-local.txt
python main.py
```
Access at http://localhost:8000

### Train Custom Model
```bash
# Add images to images/ folder with subfolders:
# images/bagpack/
# images/bottle/
# images/toothbrush/
# images/person/
# images/phone/
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

# Convert to ONNX for optimized deployment
python convert_to_onnx.py
```

## Project Structure

```
multi-object-detection/
├── main.py                          # FastAPI server with custom YOLO model
├── train_custom_model.py           # Custom YOLO training script
├── convert_to_onnx.py               # ONNX conversion script
├── split_dataset.py                # Dataset splitting utility
├── validate_dataset.py              # Dataset validation utility
├── annotation.py                    # Annotation generation utility
├── data.yaml                        # Dataset configuration
├── requirements.txt                # Python dependencies
├── requirements-local.txt          # Local development dependencies
├── vercel.json                      # Vercel deployment configuration
├── templates/                       # HTML templates
│   └── index.html                  # Web interface
├── images/                         # Dataset images (gitignored)
│   ├── bagpack/                   # Bagpack images
│   ├── bottle/                    # Bottle images
│   ├── toothbrush/                # Toothbrush images
│   ├── person/                    # Person images
│   ├── phone/                     # Phone images
│   └── book/                      # Book images
├── custom_dataset/                 # Custom dataset structure (gitignored)
│   ├── data.yaml                  # Dataset configuration
│   ├── images/                    # Training/validation/test images
│   └── labels/                    # YOLO format annotations
├── runs/                           # Training outputs (gitignored)
│   └── train/
│       └── custom_model/
│           └── weights/
│               ├── best.pt        # Best PyTorch model
│               └── best.onnx      # Best ONNX model
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

The system is trained to detect 6 custom object classes:
- bagpack (ID: 0)
- bottle (ID: 1)
- toothbrush (ID: 2)
- person (ID: 3)
- phone (ID: 4)
- book (ID: 5)

## Dataset Requirements

For optimal training performance:
- Minimum 100 images per class (600 total recommended)
- YOLO format annotations (.txt files)
- 70/20/10 train/validation/test split
- Diverse lighting, angles, and backgrounds

## Training Configuration

**Training Parameters:**
- Base model: YOLOv8n (transfer learning)
- Epochs: 200 with early stopping (patience=50)
- Batch size: 16 (adjustable based on GPU)
- Image size: 640x640
- Learning rate: 0.01 (auto-adjusted)
- Optimizer: AdamW
- Data augmentation: Enabled (mosaic, mixup, flip, rotate)
- Confidence threshold: 0.25
- IOU threshold: 0.7

## Technical Architecture

**Backend Stack:**
- FastAPI: High-performance web framework
- Ultralytics YOLOv8: Object detection model
- OpenCV: Image processing
- NumPy: Numerical operations
- ONNX Runtime: Optimized model inference

**Frontend Stack:**
- HTML5/CSS3: User interface
- JavaScript: Client-side logic
- Fetch API: HTTP requests
- MediaDevices API: Webcam access

## Deployment

The application can be deployed to cloud platforms with the trained model.

## Dataset Structure

Images and annotations are excluded from git (see .gitignore) for repository size management. The project supports:

- 6 custom object classes
- YOLO format annotations
- 70/20/10 train/validation/test split
- Transfer learning from YOLOv8n base model

## License

This project uses:
- YOLOv8 from Ultralytics (AGPL-3.0 license)
- FastAPI for web server
- OpenCV for image processing
