"""Organize dataset for YOLO training"""

import os
import shutil
from pathlib import Path
import random

def organize_yolo_dataset():
    """Organize images into YOLO format structure"""
    
    # Create YOLO dataset structure
    yolo_base = "custom_dataset"
    os.makedirs(f"{yolo_base}/images/train", exist_ok=True)
    os.makedirs(f"{yolo_base}/images/val", exist_ok=True)
    os.makedirs(f"{yolo_base}/images/test", exist_ok=True)
    os.makedirs(f"{yolo_base}/labels/train", exist_ok=True)
    os.makedirs(f"{yolo_base}/labels/val", exist_ok=True)
    os.makedirs(f"{yolo_base}/labels/test", exist_ok=True)
    
    # Class mapping
    class_names = ['bagpack', 'bottle', 'toothbrush', 'person', 'phone', 'book']
    
    # Process each class
    for class_id, class_name in enumerate(class_names):
        source_dir = f"images/{class_name}"
        
        if not os.path.exists(source_dir):
            print(f"Skipping {class_name} - directory not found")
            continue
        
        # Get all images from this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(source_dir).rglob(ext))
        
        if len(image_files) == 0:
            print(f"Skipping {class_name} - no images found")
            continue
        
        print(f"Processing {class_name}: {len(image_files)} images")
        
        # Split into train/val/test (70/20/10)
        random.shuffle(image_files)
        train_split = int(0.7 * len(image_files))
        val_split = int(0.9 * len(image_files))
        
        train_files = image_files[:train_split]
        val_files = image_files[train_split:val_split]
        test_files = image_files[val_split:]
        
        # Copy images and find/create labels
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for img_path in files:
                # Copy image
                dest_img = f"{yolo_base}/images/{split}/{class_name}_{img_path.name}"
                shutil.copy2(img_path, dest_img)
                
                # Check for existing label
                label_name = img_path.stem + '.txt'
                label_path = img_path.parent / label_name
                
                if label_path.exists():
                    # Copy existing label
                    dest_label = f"{yolo_base}/labels/{split}/{class_name}_{label_name}"
                    shutil.copy2(label_path, dest_label)
                else:
                    # Create placeholder label (you'll need to annotate these)
                    dest_label = f"{yolo_base}/labels/{split}/{class_name}_{label_name}"
                    with open(dest_label, 'w') as f:
                        # Placeholder - center box with class_id
                        f.write(f"{class_id} 0.5 0.5 0.3 0.4\n")
    
    print(f"\nDataset organized in {yolo_base}")
    print("Note: Some labels are placeholders and need proper annotation")

if __name__ == "__main__":
    organize_yolo_dataset()
