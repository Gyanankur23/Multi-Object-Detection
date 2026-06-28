# Custom YOLO Dataset Preparation Guide

## Overview
This guide explains how to prepare your custom dataset for training a YOLO model to detect: **backpack, pen, person, mobile phone, watch, book**.

## Target: 90-95% Accuracy
To achieve this accuracy, you need:
- **150 images per class** (900 total minimum)
- **High-quality annotations**
- **Diverse training data**
- **Proper data split** (70% train, 20% val, 10% test)

## Step 1: Acquire Training Images

### Option A: Use Public Datasets (Recommended)
Download images from these sources:

**COCO Dataset:**
- Download from: https://cocodataset.org/#download
- Contains: person, backpack, cell phone, book
- Filter for your classes

**Open Images Dataset:**
- Download from: https://storage.googleapis.com/openimages/web/index.html
- Contains: All your classes with annotations
- Use OIDv6 toolkit for filtering

**Kaggle Datasets:**
- Search: "backpack dataset", "watch detection", "mobile phone detection"
- Many curated datasets available

### Option B: Collect Your Own Images
Take photos of each class in various conditions:
- Different lighting (indoor, outdoor, bright, dim)
- Different angles (front, side, top, bottom)
- Different backgrounds
- Different sizes/positions
- Multiple instances per image when possible

**Image Requirements:**
- Minimum resolution: 640x640 pixels
- Format: JPG or PNG
- Clear, focused images
- Good lighting
- Objects should be clearly visible

## Step 2: Annotate Images

### Recommended Annotation Tools

**LabelImg (Easiest):**
```bash
pip install labelImg
labelImg
```
- Export format: YOLO
- Draw bounding boxes around objects
- Save .txt files with same name as images

**CVAT (Professional):**
- Web-based annotation tool
- Supports collaborative annotation
- Export to YOLO format
- Install: https://github.com/opencv/cvat

**Roboflow (Online):**
- Free tier available
- Auto-annotation with pre-trained models
- Easy export to YOLO format
- Website: https://roboflow.com

### Annotation Guidelines

**YOLO Format:**
Each annotation file (.txt) should contain one line per object:
```
<class_id> <center_x> <center_y> <width> <height>
```

Where:
- class_id: 0-5 (backpack=0, pen=1, person=2, mobile phone=3, watch=4, book=5)
- center_x, center_y: Object center (normalized 0-1)
- width, height: Object dimensions (normalized 0-1)

**Example:**
```
0 0.5 0.5 0.3 0.4  # backpack at center
1 0.2 0.3 0.1 0.15 # pen at left
```

**Quality Rules:**
- Bounding boxes should tightly fit objects
- Include entire object (no cropping)
- Minimum box size: 10x10 pixels
- Avoid overlapping boxes for same object
- Label all instances of each class

## Step 3: Organize Dataset

### Directory Structure
```
custom_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── image001.txt
    │   ├── image002.txt
    │   └── ...
    ├── val/
    └── test/
```

**Important:**
- Image and label files must have matching names
- Place images in `images/` folders
- Place labels in `labels/` folders
- Use same split for images and labels

## Step 4: Split Dataset

### Automatic Split Script
Run the provided script to split your dataset:
```bash
python split_dataset.py
```

This will:
- Randomly split 70% train, 20% val, 10% test
- Maintain class balance across splits
- Organize files into correct directories

### Manual Split
If you prefer manual splitting:
- 630 images for training (105 per class)
- 180 images for validation (30 per class)
- 90 images for testing (15 per class)

## Step 5: Data Augmentation (Optional but Recommended)

The training script includes automatic augmentation:
- Color variations (HSV)
- Random scaling
- Random translation
- Horizontal flipping
- Mosaic augmentation

To manually augment your dataset before training:
```bash
python augment_dataset.py
```

## Step 6: Validate Dataset

### Check Dataset Quality
```bash
python validate_dataset.py
```

This will verify:
- All images have corresponding labels
- Labels are in correct format
- No missing files
- Class distribution is balanced

### Expected Class Distribution
- backpack: ~150 images
- pen: ~150 images  
- person: ~150 images
- mobile phone: ~150 images
- watch: ~150 images
- book: ~150 images

## Step 7: Train Model

Once dataset is ready:
```bash
python train_custom_model.py
```

Training will:
- Load pre-trained YOLOv8n model
- Train for 200 epochs (adjustable)
- Save best model to `runs/train/custom_model/weights/best.pt`
- Generate training plots and metrics

## Step 8: Evaluate Results

After training, check:
- mAP50-95 should be >0.90 for target accuracy
- Precision and Recall should be balanced
- Confusion matrix for class-wise performance

If accuracy is below target:
1. Add more training images
2. Improve annotation quality
3. Increase training epochs
4. Try data augmentation
5. Use larger model (yolov8s instead of yolov8n)

## Common Issues & Solutions

**Issue: Low accuracy on specific class**
- Solution: Add more images for that class
- Solution: Improve annotation quality
- Solution: Check for class imbalance

**Issue: Model overfitting**
- Solution: Add more training data
- Solution: Increase data augmentation
- Solution: Reduce model complexity

**Issue: Poor generalization**
- Solution: Add diverse images (different lighting, angles)
- Solution: Increase validation set size
- Solution: Use more aggressive augmentation

## Timeline Estimate

- Image collection: 2-3 days
- Annotation: 3-5 days (150 images/day with tools)
- Dataset preparation: 1 day
- Training: 2-4 hours (depending on GPU)
- Total: 1 week

## Tips for 90-95% Accuracy

1. **Quality over quantity**: Better annotations > more images
2. **Diversity**: Vary lighting, angles, backgrounds
3. **Balance**: Equal images per class
4. **Augmentation**: Use both automatic and manual
5. **Validation**: Regular validation during annotation
6. **Testing**: Keep test set separate until final evaluation

## Next Steps

1. Collect images for each class
2. Annotate using LabelImg or CVAT
3. Run split_dataset.py
4. Run validate_dataset.py
5. Start training with train_custom_model.py
6. Evaluate results and iterate if needed
