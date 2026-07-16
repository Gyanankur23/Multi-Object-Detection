"""Train custom YOLO model"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import torch
from ultralytics.utils import SETTINGS

# Override Ultralytics dataset directory
SETTINGS['datasets_dir'] = Path(os.getcwd())

def train_custom_model():
    config = {
        'data': 'custom_dataset/data.yaml',
        'epochs': 10,  # Reduced for faster training
        'batch': 16,
        'imgsz': 640,
        'patience': 50,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    print("Loading YOLOv8n base model...")
    model = YOLO('yolov8n.pt')
    
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
        plots=True,
        save=True,
        project='runs/train',
        name='custom_model',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=8 if torch.cuda.is_available() else 4,
        close_mosaic=10,
    )
    
    print("Training completed!")
    print(f"Best model saved at: runs/train/custom_model/weights/best.pt")
    
    print("\nValidating model...")
    metrics = model.val(
        data=config['data'],
        split='val',
        conf=0.001,
        iou=0.6,
        max_det=300,
        half=True,
        dnn=False,
        plots=True,
        save_json=False,
        project='runs/val',
        name='custom_model',
        exist_ok=True,
    )
    
    print(f"\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
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
    if not os.path.exists('custom_dataset/data.yaml'):
        print("Error: custom_dataset/data.yaml not found!")
        print("Set up your dataset first.")
        exit(1)
    
    train_images = list(Path('custom_dataset/images/train').glob('*'))
    if len(train_images) == 0:
        print("Error: No training images found in custom_dataset/images/train/")
        print("Add training images and annotations first.")
        exit(1)
    
    print(f"Found {len(train_images)} training images")
    
    model, val_metrics, test_metrics = train_custom_model()
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Validation mAP50-95: {val_metrics.box.map:.4f}")
    print(f"Test mAP50-95: {test_metrics.box.map:.4f}")
    
    if val_metrics.box.map >= 0.90:
        print("✓ Target accuracy achieved (90%+)!")
    elif val_metrics.box.map >= 0.85:
        print("⚠ Good accuracy (85%+). Consider more data.")
    else:
        print("✗ Accuracy below target. Consider:")
        print("  - More training images")
        print("  - Better annotations")
        print("  - Longer training")
