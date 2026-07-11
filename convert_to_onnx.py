"""
Convert YOLO model to ONNX format for optimized deployment
Supports both custom trained models and base YOLO models
"""

from ultralytics import YOLO
import os

def convert_model_to_onnx(model_path: str, output_path: str = None):
    """
    Convert YOLO model to ONNX format
    
    Args:
        model_path: Path to the .pt model file
        output_path: Path for the output .onnx file (optional)
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Set default output path if not provided
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    # Export to ONNX
    print(f"Exporting to ONNX format at {output_path}...")
    model.export(format='onnx', simplify=True, opset=12)
    
    print(f"Successfully converted to ONNX: {output_path}")
    return True

if __name__ == "__main__":
    # Try to convert custom trained model first
    custom_model_path = 'runs/train/custom_model/weights/best.pt'
    base_model_path = 'yolov8n.pt'
    
    if os.path.exists(custom_model_path):
        print("Converting custom trained model...")
        convert_model_to_onnx(custom_model_path)
    elif os.path.exists(base_model_path):
        print("Custom model not found, converting base model...")
        convert_model_to_onnx(base_model_path)
    else:
        print("No model found. Please train a model first or download yolov8n.pt")
