"""Convert COCO annotations to YOLO format"""

import json
import os
from pathlib import Path
import shutil

def coco_to_yolo(coco_json_path, output_dir, class_mapping):
    """Convert COCO format annotations to YOLO format"""
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build image id to filename mapping
    image_id_to_filename = {}
    for image in coco_data['images']:
        image_id_to_filename[image['id']] = image['file_name']
    
    # Build annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Convert each image's annotations
    for image_id, annotations in annotations_by_image.items():
        filename = image_id_to_filename[image_id]
        # Remove .jpg extension and add .txt
        txt_filename = filename.rsplit('.', 1)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for ann in annotations:
                # Get category id and map to class id
                category_id = ann['category_id']
                class_id = class_mapping.get(category_id, 0)
                
                # Get bounding box [x, y, width, height]
                bbox = ann['bbox']
                x, y, width, height = bbox
                
                # Get image dimensions
                image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
                img_width = image_info['width']
                img_height = image_info['height']
                
                # Convert to YOLO format (normalized center_x, center_y, width, height)
                center_x = (x + width / 2) / img_width
                center_y = (y + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"Converted {len(annotations_by_image)} annotations to YOLO format")

def main():
    # Define class mapping (COCO category_id -> YOLO class_id)
    # You'll need to adjust this based on your actual COCO categories
    class_mapping = {
        0: 0,  # bagpack
        1: 1,  # bottle
        2: 2,  # toothbrush
        3: 3,  # person
        4: 4,  # phone
        5: 5,  # book
    }
    
    # Base images directory
    base_dir = "images"
    
    # Process each class and split
    classes = ['bagpack', 'book', 'bottle', 'toothbrush']
    splits = ['train', 'test', 'valid']
    
    for class_name in classes:
        for split in splits:
            # Try different annotation file names
            possible_files = [
                os.path.join(base_dir, class_name, split, '_annotations.coco.json'),
                os.path.join(base_dir, class_name, split, 'instances_val.json'),
                os.path.join(base_dir, class_name, split, f'instances_{split}.json'),
            ]
            
            coco_json = None
            for file in possible_files:
                if os.path.exists(file):
                    coco_json = file
                    break
            
            output_dir = os.path.join(base_dir, class_name, split)
            
            if coco_json:
                print(f"Processing {class_name}/{split}...")
                coco_to_yolo(coco_json, output_dir, class_mapping)
            else:
                print(f"Skipping {class_name}/{split} - no COCO annotations found")

if __name__ == "__main__":
    main()
