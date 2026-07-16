"""Convert YOLO model to ONNX format"""

from ultralytics import YOLO
import os

def convert_model_to_onnx(model_path: str, output_path: str = None):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    print(f"Exporting to ONNX at {output_path}...")
    model.export(format='onnx', simplify=True, opset=12)
    
    print(f"Converted: {output_path}")
    return True

if __name__ == "__main__":
    custom_model_path = 'runs/train/custom_model/weights/best.pt'
    base_model_path = 'yolov8n.pt'
    
    if os.path.exists(custom_model_path):
        print("Converting custom model...")
        convert_model_to_onnx(custom_model_path)
    elif os.path.exists(base_model_path):
        print("Converting base model...")
        convert_model_to_onnx(base_model_path)
    else:
        print("No model found")
