"""Split dataset into train/val/test sets"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(
    source_images_dir='custom_dataset/images',
    source_labels_dir='custom_dataset/labels',
    output_dir='custom_dataset',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    random.seed(seed)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    
    image_files = list(Path(source_images_dir).glob('*'))
    image_files = [f for f in image_files if f.suffix in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if len(image_files) == 0:
        print(f"Error: No images found in {source_images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    class_images = defaultdict(list)
    
    for img_file in image_files:
        label_file = Path(source_labels_dir) / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            print(f"Warning: No label for {img_file.name}, skipping")
            continue
        
        with open(label_file, 'r') as f:
            classes = set()
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    classes.add(class_id)
            
            if classes:
                primary_class = min(classes)
                class_images[primary_class].append(img_file)
    
    print(f"\nClass distribution:")
    for class_id, files in sorted(class_images.items()):
        print(f"  Class {class_id}: {len(files)} images")
    
    train_files = []
    val_files = []
    test_files = []
    
    for class_id, files in class_images.items():
        random.shuffle(files)
        
        n_train = int(len(files) * train_ratio)
        n_val = int(len(files) * val_ratio)
        n_test = len(files) - n_train - n_val
        
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])
    
    def copy_files(files, split):
        for img_file in files:
            dst_img = Path(output_dir) / "images" / split / img_file.name
            shutil.copy2(img_file, dst_img)
            
            label_file = Path(source_labels_dir) / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = Path(output_dir) / "labels" / split / label_file.name
                shutil.copy2(label_file, dst_label)
    
    print(f"\nCopying files...")
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print(f"\nDataset split completed:")
    print(f"  Train: {len(train_files)} images ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"  Val: {len(val_files)} images ({len(val_files)/len(image_files)*100:.1f}%)")
    print(f"  Test: {len(test_files)} images ({len(test_files)/len(image_files)*100:.1f}%)")
    
    print(f"\nVerifying split...")
    for split in ['train', 'val', 'test']:
        img_dir = Path(output_dir) / "images" / split
        label_dir = Path(output_dir) / "labels" / split
        
        n_images = len(list(img_dir.glob('*')))
        n_labels = len(list(label_dir.glob('*')))
        
        print(f"  {split}: {n_images} images, {n_labels} labels")
        
        if n_images != n_labels:
            print(f"  Warning: Mismatch in {split}")

if __name__ == "__main__":
    print("="*50)
    print("Dataset Split Script")
    print("="*50)
    
    source_images = "custom_dataset/images"
    source_labels = "custom_dataset/labels"
    
    if not os.path.exists(source_images):
        print(f"Error: {source_images} not found!")
        print("Place images in custom_dataset/images/")
        exit(1)
    
    if not os.path.exists(source_labels):
        print(f"Error: {source_labels} not found!")
        print("Place labels in custom_dataset/labels/")
        exit(1)
    
    split_dataset(
        source_images_dir=source_images,
        source_labels_dir=source_labels,
        output_dir='custom_dataset',
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    print("\n✓ Split completed!")
    print("Run: python train_custom_model.py")
