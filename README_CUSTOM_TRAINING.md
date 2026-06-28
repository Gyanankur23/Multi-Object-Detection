# Custom YOLO Model Training Project

## Project Overview
This project is set up to train a custom YOLOv8 model to detect 6 specific classes:
- **backpack**
- **pen** 
- **person**
- **mobile phone**
- **watch**
- **book**

**Target Accuracy:** 90-95% mAP50-95

## Project Structure
```
object-detector-self-model/
├── custom_dataset/
│   ├── data.yaml              # Dataset configuration
│   ├── images/
│   │   ├── train/            # Training images (70%)
│   │   ├── val/              # Validation images (20%)
│   │   └── test/             # Test images (10%)
│   └── labels/
│       ├── train/            # Training labels
│       ├── val/              # Validation labels
│       └── test/             # Test labels
├── train_custom_model.py     # Training script
├── split_dataset.py          # Dataset splitting script
├── validate_dataset.py       # Dataset validation script
├── DATASET_PREPARATION_GUIDE.md  # Detailed guide
└── yolov8n.pt               # Pre-trained base model
```

## What You Need to Provide

### 1. Training Images (900 total minimum)
- **150 images per class** (backpack, pen, person, mobile phone, watch, book)
- Format: JPG or PNG
- Resolution: Minimum 640x640 pixels
- Quality: Clear, well-lit, focused images

### 2. Bounding Box Annotations
- Each image needs a corresponding .txt file with same name
- YOLO format: `<class_id> <center_x> <center_y> <width> <height>`
- All coordinates normalized (0-1)

### 3. Class IDs
- backpack: 0
- pen: 1
- person: 2
- mobile phone: 3
- watch: 4
- book: 5

## Step-by-Step Process

### Step 1: Collect Images
**Option A: Download from Public Datasets**
- COCO Dataset: https://cocodataset.org/#download
- Open Images: https://storage.googleapis.com/openimages/web/index.html
- Kaggle: Search for specific object detection datasets

**Option B: Take Your Own Photos**
- Photograph each object in various conditions
- Different lighting, angles, backgrounds
- Multiple instances per image when possible

### Step 2: Annotate Images
**Recommended Tools:**
- **LabelImg** (Easiest): `pip install labelImg`
- **CVAT** (Professional): https://github.com/opencv/cvat
- **Roboflow** (Online): https://roboflow.com

**Process:**
1. Install annotation tool
2. Load images
3. Draw bounding boxes around objects
4. Assign correct class IDs
5. Export in YOLO format
6. Save .txt files with same names as images

### Step 3: Organize Files
Place files in the source directories:
```
custom_dataset/images/     # Put all images here
custom_dataset/labels/     # Put all labels here
```

### Step 4: Split Dataset
Run the split script:
```bash
python split_dataset.py
```
This automatically splits into:
- 70% training (630 images)
- 20% validation (180 images)  
- 10% testing (90 images)

### Step 5: Validate Dataset
Run validation script:
```bash
python validate_dataset.py
```
This checks:
- Directory structure
- Image-label pairs
- Label format
- Class distribution

### Step 6: Train Model
Start training:
```bash
python train_custom_model.py
```

**Training Parameters:**
- Base model: YOLOv8n (pre-trained)
- Epochs: 200
- Batch size: 16
- Image size: 640x640
- Data augmentation: Enabled
- Early stopping: 50 epochs patience

**Expected Training Time:**
- With GPU: 2-4 hours
- With CPU: 12-24 hours

### Step 7: Evaluate Results
After training, check:
- mAP50-95 (target: >0.90)
- Precision and Recall
- Class-wise performance
- Confusion matrix

Results saved in: `runs/train/custom_model/`

### Step 8: Deploy Model
Best model saved at: `runs/train/custom_model/weights/best.pt`

Update your web application to use this model instead of yolov8n.pt.

## Important Notes

### For 90-95% Accuracy:
1. **Quality Annotations**: Precise bounding boxes are critical
2. **Diverse Data**: Various lighting, angles, backgrounds
3. **Sufficient Data**: 150+ images per class minimum
4. **Balanced Classes**: Equal distribution across classes
5. **Proper Split**: 70/20/10 train/val/test ratio

### Common Issues:
- **Low accuracy**: Add more training data, improve annotations
- **Overfitting**: Increase data augmentation, add more diverse images
- **Class imbalance**: Add more images for underrepresented classes
- **Poor generalization**: Add more diverse training conditions

## Timeline Estimate
- Image collection: 2-3 days
- Annotation: 3-5 days (using tools like LabelImg)
- Dataset preparation: 1 day
- Training: 2-4 hours (GPU) or 12-24 hours (CPU)
- **Total: ~1 week**

## What I Cannot Do
- Download images from internet (no web scraping capability)
- Automatically annotate images (requires human interaction)
- Guarantee 90-95% accuracy without quality data

## What I Can Do
- Set up complete training infrastructure
- Configure optimal training parameters
- Create data augmentation scripts
- Validate dataset quality
- Train and evaluate models
- Deploy to web application

## Next Steps

1. **Collect images** for your 6 classes (150 per class minimum)
2. **Annotate images** using LabelImg or similar tool
3. **Place files** in `custom_dataset/images/` and `custom_dataset/labels/`
4. **Run** `python split_dataset.py`
5. **Run** `python validate_dataset.py`
6. **Run** `python train_custom_model.py`
7. **Evaluate** results and iterate if needed

## Support Files
- `DATASET_PREPARATION_GUIDE.md` - Detailed dataset preparation instructions
- `split_dataset.py` - Automatic dataset splitting
- `validate_dataset.py` - Dataset quality validation
- `train_custom_model.py` - Optimized training script

## Contact
If you encounter issues:
1. Check `DATASET_PREPARATION_GUIDE.md` for detailed instructions
2. Run `validate_dataset.py` to check for issues
3. Review training logs in `runs/train/custom_model/`

## Success Criteria
Your model is successful when:
- mAP50-95 > 0.90 on validation set
- mAP50-95 > 0.85 on test set
- All 6 classes are detected reliably
- No significant class imbalance in performance
