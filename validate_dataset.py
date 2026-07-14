"""
Dataset Validation Script
Validates YOLO dataset format and quality
"""

import os
from pathlib import Path
from collections import defaultdict

def validate_dataset(dataset_dir='custom_dataset'):
    """
    Validate YOLO dataset structure and quality
    """
    
    print("="*50)
    print("Dataset Validation")
    print("="*50)
    
    errors = []
    warnings = []
    
    # Check directory structure
    print("\n1. Checking directory structure...")
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    for dir_path in required_dirs:
        full_path = Path(dataset_dir) / dir_path
        if not full_path.exists():
            errors.append(f"Missing directory: {dir_path}")
        else:
            print(f"  ✓ {dir_path}")
    
    if errors:
        print("\n❌ Directory structure errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    # Check data.yaml
    print("\n2. Checking data.yaml...")
    data_yaml = Path(dataset_dir) / 'data.yaml'
    if not data_yaml.exists():
        errors.append("Missing data.yaml file")
    else:
        print(f"  ✓ data.yaml exists")
    
    # Check image-label pairs
    print("\n3. Checking image-label pairs...")
    for split in ['train', 'val', 'test']:
        img_dir = Path(dataset_dir) / 'images' / split
        label_dir = Path(dataset_dir) / 'labels' / split
        
        if not img_dir.exists() or not label_dir.exists():
            continue
        
        images = list(img_dir.glob('*'))
        images = [f for f in images if f.suffix in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        missing_labels = []
        orphan_labels = []
        
        for img in images:
            label_file = label_dir / f"{img.stem}.txt"
            if not label_file.exists():
                missing_labels.append(img.name)
        
        labels = list(label_dir.glob('*.txt'))
        for label in labels:
            img_file = img_dir / f"{label.stem}.jpg"
            if not img_file.exists():
                img_file = img_dir / f"{label.stem}.png"
            if not img_file.exists():
                orphan_labels.append(label.name)
        
        if missing_labels:
            warnings.append(f"{split}: {len(missing_labels)} images without labels")
        
        if orphan_labels:
            warnings.append(f"{split}: {len(orphan_labels)} labels without images")
        
        if not missing_labels and not orphan_labels:
            print(f"  ✓ {split}: All images have matching labels")
    
    # Check label format
    print("\n4. Checking label format...")
    class_counts = defaultdict(int)
    invalid_labels = []
    
    for split in ['train', 'val', 'test']:
        label_dir = Path(dataset_dir) / 'labels' / split
        if not label_dir.exists():
            continue
        
        labels = list(label_dir.glob('*.txt'))
        for label_file in labels:
            try:
                with open(label_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            invalid_labels.append(f"{label_file.name}:{line_num} - Invalid format")
                            continue
                        
                        class_id = int(parts[0])
                        if class_id < 0 or class_id > 5:
                            invalid_labels.append(f"{label_file.name}:{line_num} - Invalid class ID: {class_id}")
                            continue
                        
                        # Check if coordinates are normalized (0-1)
                        for i in range(1, 5):
                            val = float(parts[i])
                            if val < 0 or val > 1:
                                invalid_labels.append(f"{label_file.name}:{line_num} - Coordinate not normalized: {val}")
                                continue
                        
                        class_counts[class_id] += 1
            except Exception as e:
                invalid_labels.append(f"{label_file.name} - Read error: {str(e)}")
    
    if invalid_labels:
        errors.append(f"{len(invalid_labels)} invalid label entries")
        print(f"  ⚠ Found {len(invalid_labels)} invalid entries")
    else:
        print(f"  ✓ All labels have valid format")
    
    # Class distribution
    print("\n5. Class distribution:")
    class_names = ['bagpack', 'bottle', 'toothbrush', 'person', 'phone', 'book']
    for class_id in range(6):
        count = class_counts.get(class_id, 0)
        print(f"  {class_names[class_id]} (ID {class_id}): {count} annotations")
    
    # Check for class imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 2:
            warnings.append(f"Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        else:
            print(f"  ✓ Class distribution is balanced")
    
    # Summary
    print("\n" + "="*50)
    print("Validation Summary")
    print("="*50)
    
    if errors:
        print(f"❌ {len(errors)} error(s) found:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("✓ Dataset is valid and ready for training!")
        return True
    elif not errors:
        print("⚠ Dataset has warnings but can be used for training")
        return True
    else:
        print("❌ Dataset has errors that must be fixed before training")
        return False

if __name__ == "__main__":
    is_valid = validate_dataset('custom_dataset')
    
    if is_valid:
        print("\nNext steps:")
        print("1. Review warnings if any")
        print("2. Run training: python train_custom_model.py")
    else:
        print("\nPlease fix the errors before training")
