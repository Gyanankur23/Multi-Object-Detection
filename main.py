"""
FastAPI Application for Custom YOLO Object Detection
Server-side implementation using locally trained custom YOLO model
Supports custom object classes: person, backpack, toothbrush, bottle, book
Supports both PyTorch (.pt) and ONNX (.onnx) model formats
For deployment, uses ONNX runtime for optimal performance and size
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from contextlib import asynccontextmanager
import os
import io
import logging

# Try to import ML dependencies (optional for deployment)
try:
    import cv2
    import numpy as np
    import onnxruntime as ort
    ML_DEPENDENCIES_AVAILABLE = True
except ImportError:
    ML_DEPENDENCIES_AVAILABLE = False
    cv2 = None
    np = None
    ort = None

# Try to import ultralytics for local development (optional)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
onnx_session = None
use_onnx = False

# Custom class names for trained model
CUSTOM_CLASSES = {
    0: "person",
    1: "backpack", 
    2: "toothbrush",
    3: "bottle",
    4: "book"
}

def get_class_name(class_id: int):
    return CUSTOM_CLASSES.get(class_id, f"class_{class_id}")

# Lifespan manager replaces deprecated startup events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, onnx_session, use_onnx
    logger.info("Loading custom YOLO model on startup...")
    
    # Try to load ONNX model first (preferred for deployment)
    if os.path.exists('runs/train/custom_model/weights/best.onnx'):
        logger.info("Loading ONNX model for optimized inference...")
        onnx_session = ort.InferenceSession('runs/train/custom_model/weights/best.onnx', providers=['CPUExecutionProvider'])
        use_onnx = True
        logger.info("ONNX model loaded successfully.")
    elif os.path.exists('yolov8n.onnx'):
        logger.info("Loading base ONNX model...")
        onnx_session = ort.InferenceSession('yolov8n.onnx', providers=['CPUExecutionProvider'])
        use_onnx = True
        logger.info("ONNX model loaded successfully.")
    # Fall back to PyTorch models (only if ultralytics is available)
    elif ULTRALYTICS_AVAILABLE:
        if os.path.exists('runs/train/custom_model/weights/best.pt'):
            logger.info("Loading custom trained PyTorch model...")
            model = YOLO('runs/train/custom_model/weights/best.pt')
        elif os.path.exists('yolov8n.pt'):
            logger.info("Custom model not found, loading base PyTorch model for transfer learning...")
            model = YOLO('yolov8n.pt')
        else:
            logger.info("No model found, downloading base model...")
            model = YOLO('yolov8n.pt')
    else:
        logger.warning("No ONNX model found and ultralytics not available. Please provide an ONNX model for deployment.")
    
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

def run_onnx_inference(img, onnx_session, confidence_threshold: float = 0.5):
    """
    Run inference using ONNX model
    Returns list of detections in format: [x1, y1, x2, y2, confidence, class_id]
    """
    # Preprocess image
    original_shape = img.shape
    input_shape = (640, 640)  # YOLO default input size
    
    # Resize and normalize
    resized = cv2.resize(img, input_shape)
    normalized = resized.astype(np.float32) / 255.0
    transposed = normalized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(transposed, axis=0)
    
    # Get input/output names
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    # Run inference
    outputs = onnx_session.run([output_name], {input_name: input_tensor})
    
    # Process outputs (YOLO format: [batch, 4 + num_classes, num_boxes])
    predictions = outputs[0]
    
    # Parse predictions
    detections = []
    for pred in predictions[0]:  # First batch
        boxes = pred[:4]  # x, y, w, h
        scores = pred[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence >= confidence_threshold:
            # Convert center/width/height to x1/y1/x2/y2
            x, y, w, h = boxes
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            # Scale to original image size
            scale_x = original_shape[1] / input_shape[1]
            scale_y = original_shape[0] / input_shape[0]
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            
            detections.append([x1, y1, x2, y2, confidence, class_id])
    
    return detections

def draw_detections_onnx(frame, detections, confidence_threshold: float = 0.5):
    """
    Draw detections from ONNX inference results
    """
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        
        if confidence < confidence_threshold:
            continue
        
        class_name = get_class_name(int(class_id))
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
    global model, onnx_session, use_onnx
    
    if not ML_DEPENDENCIES_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "error": "ML dependencies not available in this deployment",
                "message": "This deployment is running in minimal mode. For full object detection capabilities, use the Railway deployment.",
                "railway_url": "https://railway.app"
            }
        )
    
    if model is None and onnx_session is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Read uploaded image frame
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference based on available model
        if use_onnx and onnx_session is not None:
            # ONNX inference
            results = run_onnx_inference(img, onnx_session)
            annotated_img = draw_detections_onnx(img.copy(), results)
        else:
            # PyTorch inference
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
            "description": "Custom object detection for 5 classes: person, backpack, toothbrush, bottle, book"
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
