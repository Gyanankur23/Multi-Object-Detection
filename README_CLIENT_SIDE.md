# Real-Time Object Tracking - Client Side

## Overview

This is a **client-side only** version of the YOLO object tracking application. It runs entirely in the browser using ONNX Runtime Web, eliminating server latency and enabling real-time webcam object detection.

## Key Benefits

- **No Server Latency**: All processing happens in the browser
- **Real-Time Performance**: Direct webcam access with minimal delay
- **Easy Deployment**: Can be deployed to Vercel, Netlify, or any static hosting
- **No Device Restrictions**: Works on any device with a modern browser
- **Cost Effective**: No server costs - static hosting only

## Quick Start

### Local Testing

1. Ensure you have the ONNX model file (`yolov8n.onnx`) in the project directory
2. Start a local HTTP server:

```bash
# Python 3
python -m http.server 8000

# Or using Node.js
npx serve
```

3. Open your browser to `http://localhost:8000`
4. Click "Start Camera" and allow camera permissions

### Deployment to Vercel

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
vercel
```

3. Follow the prompts - Vercel will detect it as a static site

### Deployment to Netlify

1. Drag and drop the project folder to Netlify
2. Or use Netlify CLI:
```bash
npm install -g netlify-cli
netlify deploy --prod
```

## Files Required for Deployment

- `index.html` - Main application file
- `yolov8n.onnx` - ONNX model file (12.1 MB)
- `vercel.json` - Vercel configuration (optional)

## Model Conversion

If you need to convert a different YOLO model to ONNX:

```bash
python convert_to_onnx.py
```

## Browser Requirements

- Modern browser with WebGL support
- Camera permissions
- HTTPS (required for camera access on deployed sites)

## Performance

- Target FPS: 15-30 FPS depending on device
- Inference time: 50-150ms on modern devices
- Memory usage: ~200-400MB

## Technical Details

- **Framework**: ONNX Runtime Web
- **Model**: YOLOv8n (Nano) - Optimized for speed
- **Processing**: Client-side only
- **Detection**: 80 COCO classes

## Troubleshooting

**Camera not working?**
- Ensure you're using HTTPS (required for camera access)
- Check browser permissions
- Try a different browser

**Slow performance?**
- Close other browser tabs
- Try a smaller model (yolov8n is already optimized)
- Check if WebGL is enabled in your browser

**Model loading error?**
- Ensure `yolov8n.onnx` is in the same directory as `index.html`
- Check browser console for specific errors
- Verify the model file is not corrupted

## Comparison with Server-Side Version

| Feature | Server-Side (Render) | Client-Side (This) |
|---------|---------------------|-------------------|
| Latency | High (network round-trip) | Low (local processing) |
| Deployment | Complex (Docker, server) | Simple (static hosting) |
| Cost | Server costs | Free (static hosting) |
| Device Restrictions | Yes (server limitations) | No (browser-based) |
| Real-Time Performance | Limited | Excellent |
| Privacy | Frames sent to server | All processing local |

## License

Same as the original YOLO project.
