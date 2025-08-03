# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based retail security surveillance application that uses computer vision for real-time object detection and monitoring. The application leverages YOLOv8 (You Only Look Once) for object detection on video streams from security cameras, designed for retail loss prevention and security monitoring.

## Core Dependencies

- **OpenCV (cv2)**: Computer vision library for video capture, image processing, and display
- **Ultralytics YOLO**: YOLOv8 implementation for real-time object detection
- **Python 3.x**: Core runtime environment

## Common Development Commands

### Environment Setup
```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Or install manually:
# Basic dependencies
pip install opencv-python ultralytics

# For Streamlit web app with AI chat
pip install streamlit openai python-dotenv

# For FastAPI web server
pip install fastapi uvicorn pydantic

# Additional dependencies may be needed:
pip install torch torchvision  # For YOLO model inference
pip install numpy matplotlib   # Common CV dependencies
```

### Running the Application
```bash
python Basic.py                       # Run basic surveillance application
streamlit run streamlit_webrtc_app.py # Run interactive Streamlit web app with AI chat
python api.py                         # Run FastAPI web server with REST API
uvicorn api:app --reload              # Run FastAPI with auto-reload for development
```

### YOLO Model Management
```bash
# Download different YOLO models
# yolov8n.pt - Nano (fastest, least accurate)
# yolov8s.pt - Small 
# yolov8m.pt - Medium
# yolov8l.pt - Large
# yolov8x.pt - Extra Large (slowest, most accurate)
```

### Development and Testing
```bash
python -c "import cv2; print(cv2.__version__)"  # Verify OpenCV installation
python -c "from ultralytics import YOLO; print('YOLO imported successfully')"  # Verify YOLO
```

## Application Architecture

### Application Types

#### Basic Surveillance App (`app.py`)
Simple OpenCV-based application for direct video processing with YOLO detection.

#### Interactive Streamlit App (`streamlit_webrtc_app.py`)
Web-based application with AI-powered chat interface for dynamic object filtering.

#### FastAPI REST API Server (`api.py`)
Production-ready REST API server with natural language query processing for surveillance video streams.

### Core Components

#### Video Stream Management
- **Video Capture**: Uses `cv2.VideoCapture()` to connect to camera streams
- **Stream Sources**: Supports webcam (index 0) and IP camera RTSP streams
- **Frame Processing**: Real-time frame-by-frame analysis

#### Object Detection Pipeline
- **Model Loading**: YOLOv8 pretrained model initialization (`yolov8n.pt`)
- **Inference**: Real-time object detection on video frames
- **Annotation**: Automatic bounding box and label drawing
- **Display**: Live visualization of detection results

#### AI Chat Interface (Streamlit App Only)
- **OpenAI Integration**: GPT-4 for natural language processing
- **Object Filtering**: Dynamic filtering based on chat commands
- **Class Mapping**: Converts natural language to YOLO class indices
- **Filter Persistence**: Remembers filter settings between commands

#### Camera Integration
- **RTSP Support**: Designed for IP security cameras via RTSP protocol
- **Webcam Fallback**: Can use local webcam for testing/development
- **Stream Configuration**: Configurable video source via `VideoCapture` parameter

### Key Design Patterns

#### Real-time Processing Loop
The application follows a continuous processing pattern:
1. Capture frame from video stream
2. Run YOLO detection inference
3. Annotate frame with detection results
4. Display processed frame
5. Handle user input (quit on 'Q')
6. Repeat until stream ends or user exits

#### Model Management
- Uses pretrained YOLOv8 nano model for balance of speed and accuracy
- Model is loaded once at startup for efficiency
- Detection results include bounding boxes, confidence scores, and class labels

## Configuration and Customization

### Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here  # Required for AI chat and natural language queries
CAMERA_SOURCE=0                          # Camera source (0 for webcam, RTSP URL for IP camera)
CONFIDENCE_THRESHOLD=0.5                 # Object detection confidence threshold
MODEL_PATH=yolov8n.pt                    # YOLO model file path
PORT=8000                                # FastAPI server port
HOST=0.0.0.0                            # FastAPI server host
```

### Camera Stream Configuration
- **Basic App Line 9**: Update RTSP URL for your specific IP camera
- **Format**: `"rtsp://username:password@ip_address:port/stream_path"`
- **Webcam**: Use `0` for default webcam, `1` for secondary camera
- **Streamlit App**: Camera source configurable in `VideoProcessor.start_camera()`

### Model Selection
- **Line 6**: Replace `"yolov8n.pt"` with different YOLO variants
- **Custom Models**: Can substitute with custom-trained YOLO models
- **Model Path**: Supports local model files or automatic downloads

### AI Chat Commands (Streamlit App)
Natural language examples:
- `"show only people"` - Filter to person class only
- `"cars and trucks"` - Show car and truck classes
- `"show all objects"` - Reset filter to show everything
- `"bottles and cups"` - Filter to bottle and cup classes

### Detection Classes
YOLOv8 pretrained models detect 80 COCO classes including:
- person, bicycle, car, motorcycle, airplane, bus, train, truck
- bottle, wine glass, cup, fork, knife, spoon, bowl
- laptop, mouse, remote, keyboard, cell phone, book
- handbag, tie, suitcase, frisbee, skis, snowboard
- And many more retail-relevant objects

## Security and Privacy Considerations

### Camera Access
- Ensure proper network security for IP camera streams
- Use secure RTSP credentials and encrypted connections when possible
- Consider VPN access for remote camera monitoring

### Data Handling
- Video frames are processed in real-time without storage by default
- No automatic recording or data persistence in current implementation
- Consider data retention policies for any added recording features

### Performance Optimization
- **Model Selection**: Choose appropriate YOLO variant based on hardware capabilities
- **Frame Rate**: Adjust processing rate based on detection requirements vs. performance
- **Resolution**: Consider input resolution vs. processing speed trade-offs

## Retail Security Applications

### Loss Prevention
- Person detection and counting
- Unusual behavior pattern recognition
- Restricted area monitoring

### Inventory Monitoring
- Product detection and tracking
- Shelf monitoring for stock levels
- Unauthorized item removal detection

### Customer Analytics
- Foot traffic analysis
- Dwell time measurement
- Customer flow patterns

## Troubleshooting

### NumPy Version Compatibility (Intel Mac)
If you encounter NumPy 2.x compatibility errors with PyTorch/Ultralytics:

```bash
# Fix dependency conflicts
source .venv/bin/activate
pip install "numpy<2" "opencv-python<4.11"
```

**Common Error**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6`
**Solution**: Downgrade to NumPy 1.x and compatible OpenCV version

### Camera Connection Issues
```bash
# Test webcam access (use 0 for default camera)
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam works:' if cap.isOpened() else 'Webcam failed')"

# Test RTSP stream
python -c "import cv2; cap = cv2.VideoCapture('rtsp://your_camera_ip:port/stream'); print('RTSP works:' if cap.isOpened() else 'RTSP failed')"
```

**RTSP Timeout**: Check camera IP, credentials, and network connectivity
**Webcam Permission**: macOS may require camera permissions in System Preferences

### Environment Verification
```bash
# Verify all dependencies
python -c "import cv2; from ultralytics import YOLO; print('All dependencies loaded successfully')"

# Check versions
python -c "import cv2, numpy as np; print(f'OpenCV: {cv2.__version__}, NumPy: {np.__version__}')"
```

## Development Notes

### Hardware Requirements
- Camera with RTSP stream capability or USB webcam
- Adequate processing power for real-time YOLO inference
- GPU acceleration recommended for higher resolution or multiple streams

### Intel Mac Specific Notes
- Use webcam (VideoCapture(0)) for initial testing
- Camera permissions may need to be granted in System Preferences > Security & Privacy
- RTSP streams require network access to camera device

### Extension Points
- **Alert System**: Add notifications for specific detection events
- **Recording**: Implement video recording for security incidents
- **Multi-Stream**: Support multiple camera feeds simultaneously
- **Custom Training**: Train YOLO models on retail-specific datasets
- **Analytics**: Add detection counting, timing, and reporting features

## FastAPI REST API Usage

### Starting the API Server
```bash
# Start FastAPI server
python api.py

# Or with uvicorn for development
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Core Endpoints
- `GET /` - API information and available endpoints
- `GET /health` - Health check and service status
- `GET /docs` - Interactive Swagger API documentation
- `GET /redoc` - Alternative API documentation

#### Video Stream Management
- `POST /stream/start` - Start video capture from camera
- `POST /stream/stop` - Stop video capture
- `GET /stream/status` - Get current stream status and info

#### Object Detection
- `GET /detections/current` - Get current detection results
- `GET /detections/classes` - List all detectable object classes
- `GET /detections/summary` - Get detection statistics
- `POST /detections/filter` - Filter detections by class names

#### Natural Language Queries
- `POST /query` - Ask natural language questions about the video
- `GET /query/suggestions` - Get example query suggestions

#### Configuration
- `POST /config/confidence` - Update detection confidence threshold

### Example API Usage

#### Starting Video Stream
```bash
curl -X POST "http://localhost:8000/stream/start" \
     -H "Content-Type: application/json" \
     -d '{"source": 0, "confidence_threshold": 0.5}'
```

#### Natural Language Query
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Are there any dogs in the patio?"}'
```

#### Get Current Detections
```bash
curl "http://localhost:8000/detections/current"
```

#### Filter Detections
```bash
curl -X POST "http://localhost:8000/detections/filter" \
     -H "Content-Type: application/json" \
     -d '["person", "car", "dog"]'
```

### Natural Language Query Examples
- "Are there any dogs in the patio?"
- "How many people do you see?"
- "What vehicles are visible?"
- "Is there a laptop on the desk?"
- "Are there any bottles or cups?"
- "What animals do you see?"
- "How many objects are detected in total?"

### Testing the API
```bash
# Install test dependencies
pip install requests

# Run comprehensive API tests
python test_api.py

# Or test individual endpoints manually
curl http://localhost:8000/health
```

### Integration with External Applications
The FastAPI server provides a clean REST interface that external applications can use to:

1. **Query Current State**: Ask questions about what's currently visible
2. **Get Raw Detection Data**: Retrieve structured detection results
3. **Control Stream**: Start/stop video capture programmatically
4. **Monitor System Health**: Check if services are running properly

### Production Deployment Considerations
- Set appropriate environment variables for camera source and API keys
- Use proper ASGI server like uvicorn with production settings
- Implement authentication/authorization if needed
- Consider rate limiting for query endpoints
- Monitor resource usage for continuous video processing
- Set up logging and error tracking