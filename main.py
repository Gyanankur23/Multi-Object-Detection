"""YOLO Object Detection API"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

model = None
onnx_session = None
use_onnx = False

CUSTOM_CLASSES = {0: "bagpack", 1: "bottle", 2: "toothbrush", 3: "person", 4: "phone", 5: "book"}

def get_class_name(class_id: int):
    return CUSTOM_CLASSES.get(class_id, f"class_{class_id}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, onnx_session, use_onnx
    logger.info("Loading model...")
    
    if os.path.exists('runs/train/custom_model/weights/best.onnx'):
        try:
            onnx_session = ort.InferenceSession('runs/train/custom_model/weights/best.onnx', providers=['CPUExecutionProvider'])
            use_onnx = True
            logger.info("ONNX model loaded")
        except Exception as e:
            logger.warning(f"ONNX load failed: {e}")
            onnx_session = None
            use_onnx = False
    elif ULTRALYTICS_AVAILABLE:
        if os.path.exists('runs/train/custom_model/weights/best.pt'):
            model = YOLO('runs/train/custom_model/weights/best.pt')
        elif os.path.exists('yolov8n.pt'):
            model = YOLO('yolov8n.pt')
        else:
            model = YOLO('yolov8n.pt')
    else:
        logger.warning("No model available - demo mode")
    
    logger.info("Ready")
    yield
    logger.info("Shutting down")

app = FastAPI(title="YOLO Detector", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def draw_detections(frame, results, confidence_threshold: float = 0.5):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            if confidence < confidence_threshold:
                continue
            
            class_name = get_class_name(class_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame

def run_onnx_inference(img, onnx_session, confidence_threshold: float = 0.5):
    try:
        original_shape = img.shape
        input_shape = (640, 640)
        
        resized = cv2.resize(img, input_shape)
        normalized = resized.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(transposed, axis=0)
        
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        outputs = onnx_session.run([output_name], {input_name: input_tensor})
        predictions = outputs[0]
        
        detections = []
        for pred in predictions[0]:
            boxes = pred[:4]
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence >= confidence_threshold:
                x, y, w, h = boxes
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                scale_x = original_shape[1] / input_shape[1]
                scale_y = original_shape[0] / input_shape[0]
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y
                
                detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return detections
    except Exception as e:
        logger.error(f"ONNX error: {e}")
        return []

def draw_detections_onnx(frame, detections, confidence_threshold: float = 0.5):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        
        if confidence < confidence_threshold:
            continue
        
        class_name = get_class_name(int(class_id))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
        cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
def detect_objects(file: UploadFile = File(...)):
    global model, onnx_session, use_onnx
    
    if not ML_DEPENDENCIES_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "error": "ML dependencies not available",
                "message": "This deployment is running in minimal mode. Install cv2, numpy, and onnxruntime for full object detection capabilities."
            }
        )
    
    if model is None and onnx_session is None:
        logger.warning("Model not loaded - attempting to load")
        # Try to load model on-demand
        if os.path.exists('runs/train/custom_model/weights/best.onnx'):
            try:
                onnx_session = ort.InferenceSession('runs/train/custom_model/weights/best.onnx', providers=['CPUExecutionProvider'])
                use_onnx = True
                logger.info("ONNX model loaded on-demand")
            except Exception as e:
                logger.warning(f"ONNX load failed: {e}")
        elif ULTRALYTICS_AVAILABLE and os.path.exists('runs/train/custom_model/weights/best.pt'):
            try:
                model = YOLO('runs/train/custom_model/weights/best.pt')
                logger.info("YOLO model loaded on-demand")
            except Exception as e:
                logger.warning(f"YOLO load failed: {e}")
        
        if model is None and onnx_session is None:
            raise HTTPException(status_code=500, detail="Model not available - please ensure model files exist")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        if use_onnx and onnx_session is not None:
            results = run_onnx_inference(img, onnx_session, 0.5)
            annotated_img = draw_detections_onnx(img.copy(), results, 0.5)
        else:
            results = model(img)
            annotated_img = draw_detections(img.copy(), results, 0.5)
        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_bytes = buffer.tobytes()
        
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None or onnx_session is not None,
        "ml_dependencies": ML_DEPENDENCIES_AVAILABLE,
        "ultralytics_available": ULTRALYTICS_AVAILABLE
    }


@app.get("/model_info")
async def model_info():
    global model, onnx_session
    
    if model is None and onnx_session is None:
        return {"model_type": "Not loaded", "status": "error"}
    
    if os.path.exists('runs/train/custom_model/weights/best.pt') or os.path.exists('runs/train/custom_model/weights/best.onnx'):
        return {
            "model_type": "Custom YOLO",
            "model_path": "runs/train/custom_model/weights/best.pt or .onnx",
            "classes": CUSTOM_CLASSES,
            "num_classes": len(CUSTOM_CLASSES)
        }
    else:
        return {
            "model_type": "Base YOLOv8n",
            "model_path": "yolov8n.pt or yolov8n.onnx",
            "classes": CUSTOM_CLASSES,
            "num_classes": len(CUSTOM_CLASSES)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
