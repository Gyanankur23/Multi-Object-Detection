"""
Custom YOLO Model Training Script
Trains a YOLOv8 model on custom dataset using transfer learning
Base model: YOLOv8n, fine-tuned on custom dataset
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import torch

def train_custom_model():
    """
    Train custom YOLO model with optimized parameters for high accuracy
    """
    
    # Configuration for high accuracy training
    config = {
        'data': 'custom_dataset/data.yaml',
        'epochs': 200,  # Increased epochs for better convergence
        'batch': 16,  # Batch size (adjust based on GPU memory)
        'imgsz': 640,  # Image size
        'patience': 50,  # Early stopping patience
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Cls loss gain
        'dfl': 1.5,  # DFL loss gain
        'pose': 12.0,  # Pose loss gain
        'kobj': 1.0,  # Keypoint obj loss gain
        'label_smoothing': 0.0,  # Label smoothing
        'nbs': 64,  # Nominal batch size
        'hsv_h': 0.015,  # Image HSV-Hue augmentation
        'hsv_s': 0.7,  # Image HSV-Saturation augmentation
        'hsv_v': 0.4,  # Image HSV-Value augmentation
        'degrees': 0.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,  # Image scale (+/- gain)
        'shear': 0.0,  # Image shear (+/- deg)
        'perspective': 0.0,  # Image perspective (+/- fraction)
        'flipud': 0.0,  # Image flip up-down
        'fliplr': 0.5,  # Image flip left-right
        'mosaic': 1.0,  # Image mosaic
        'mixup': 0.0,  # Image mixup
        'copy_paste': 0.0,  # Segment copy-paste
    }
    
    # Load YOLOv8n model as base for transfer learning on custom dataset
    print("Loading YOLOv8n base model for transfer learning...")
    model = YOLO('yolov8n.pt')
    
    # Train the model
    print("Starting training...")
    print(f"Dataset: {config['data']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch']}")
    print(f"Image size: {config['imgsz']}")
    
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        patience=config['patience'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        warmup_bias_lr=config['warmup_bias_lr'],
        box=config['box'],
        cls=config['cls'],
        dfl=config['dfl'],
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        shear=config['shear'],
        perspective=config['perspective'],
        flipud=config['flipud'],
        fliplr=config['fliplr'],
        mosaic=config['mosaic'],
        mixup=config['mixup'],
        copy_paste=config['copy_paste'],
        plots=True,  # Save training plots
        save=True,  # Save checkpoints
        project='runs/train',
        name='custom_model',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,  # Random seed for reproducibility
        deterministic=True,  # Deterministic training
        single_cls=False,  # Multi-class training
        device='0' if torch.cuda.is_available() else 'cpu',  # Auto-detect GPU/CPU
        workers=8 if torch.cuda.is_available() else 4,  # Adjust workers based on device
        close_mosaic=10,  # Disable mosaic in final epochs
    )
    
    print("Training completed!")
    print(f"Best model saved at: runs/train/custom_model/weights/best.pt")
    
    # Validate the model
    print("\nValidating model...")
    metrics = model.val(
        data=config['data'],
        split='val',
        conf=0.001,  # Confidence threshold for validation
        iou=0.6,  # IoU threshold for validation
        max_det=300,  # Maximum detections per image
        half=True,  # Use FP16 precision
        dnn=False,  # Use OpenCV DNN
        plots=True,  # Save validation plots
        save_json=False,  # Save results to JSON
        project='runs/val',
        name='custom_model',
        exist_ok=True,
    )
    
    print(f"\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Test the model
    print("\nTesting model...")
    test_metrics = model.val(
        data=config['data'],
        split='test',
        conf=0.001,
        iou=0.6,
        max_det=300,
        half=True,
        dnn=False,
        plots=True,
        project='runs/test',
        name='custom_model',
        exist_ok=True,
    )
    
    print(f"\nTest Results:")
    print(f"mAP50: {test_metrics.box.map50:.4f}")
    print(f"mAP50-95: {test_metrics.box.map:.4f}")
    print(f"Precision: {test_metrics.box.mp:.4f}")
    print(f"Recall: {test_metrics.box.mr:.4f}")
    
    return model, metrics, test_metrics

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('custom_dataset/data.yaml'):
        print("Error: custom_dataset/data.yaml not found!")
        print("Please set up your dataset first.")
        exit(1)
    
    # Check if training images exist
    train_images = list(Path('custom_dataset/images/train').glob('*'))
    if len(train_images) == 0:
        print("Error: No training images found in custom_dataset/images/train/")
        print("Please add your training images and annotations first.")
        exit(1)
    
    print(f"Found {len(train_images)} training images")
    
    # Start training
    model, val_metrics, test_metrics = train_custom_model()
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Validation mAP50-95: {val_metrics.box.map:.4f}")
    print(f"Test mAP50-95: {test_metrics.box.map:.4f}")
    
    if val_metrics.box.map >= 0.90:
        print("✓ Target accuracy achieved (90%+)!")
    elif val_metrics.box.map >= 0.85:
        print("⚠ Good accuracy achieved (85%+). Consider more training data.")
    else:
        print("✗ Accuracy below target. Consider:")
        print("  - More training images")
        print("  - Better quality annotations")
        print("  - Longer training")
        print("  - Data augmentation")
