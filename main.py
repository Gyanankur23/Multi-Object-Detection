"""
FastAPI Application for Custom YOLO Object Detection
Server-side implementation using locally trained custom YOLO model
Supports custom object classes: backpack, pen, person, mobile phone, notebook
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from contextlib import asynccontextmanager
import cv2
import numpy as np
from ultralytics import YOLO
import os
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

# Custom class names for trained model
CUSTOM_CLASSES = {
    0: "backpack",
    1: "pen", 
    2: "person",
    3: "mobile phone",
    4: "notebook"
}

def get_class_name(class_id: int):
    return CUSTOM_CLASSES.get(class_id, f"class_{class_id}")

# Lifespan manager replaces deprecated startup events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading custom YOLO model on startup...")
    
    # Try to load custom trained model first
    if os.path.exists('runs/train/custom_model/weights/best.pt'):
        logger.info("Loading custom trained model...")
        model = YOLO('runs/train/custom_model/weights/best.pt')
    elif os.path.exists('yolov8n.pt'):
        logger.info("Custom model not found, loading base model for transfer learning...")
        model = YOLO('yolov8n.pt')
    else:
        logger.info("No model found, downloading base model...")
        model = YOLO('yolov8n.pt')
    
    logger.info("Model loaded successfully.")
    yield
    # Clean up resources here if needed
    logger.info("Shutting down application...")

# Initialize FastAPI app
app = FastAPI(title="Custom YOLO Object Detector", version="1.0.0", lifespan=lifespan)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Mount static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def draw_detections(frame, results, confidence_threshold: float = 0.5):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            if confidence < confidence_threshold:
                continue
            
            class_name = get_class_name(class_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10),
                          (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Note: Removed "async" so blocking YOLO processing runs safely in FastAPI's threadpool
@app.post("/detect")
def detect_objects(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Read uploaded image frame
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference
        results = model(img)
        annotated_img = draw_detections(img.copy(), results)
        
        # Convert frame back to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_bytes = buffer.tobytes()
        
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model_info")
async def model_info():
    """
    Get model information
    """
    global model
    
    if model is None:
        logger.warning("Model not loaded")
        return {
            "model_type": "Not loaded",
            "status": "error"
        }
    
    # Check if custom model is loaded
    if os.path.exists('runs/train/custom_model/weights/best.pt'):
        return {
            "model_type": "Custom trained YOLO model",
            "model_path": "runs/train/custom_model/weights/best.pt",
            "classes": CUSTOM_CLASSES,
            "num_classes": len(CUSTOM_CLASSES),
            "description": "Custom object detection for 5 classes: backpack, pen, person, mobile phone, notebook"
        }
    else:
        return {
            "model_type": "Base YOLOv8n model (for transfer learning)",
            "model_path": "yolov8n.pt",
            "classes": CUSTOM_CLASSES,
            "num_classes": len(CUSTOM_CLASSES),
            "description": "Base model for transfer learning on custom dataset"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
