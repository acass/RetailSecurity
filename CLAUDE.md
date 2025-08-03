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
# Basic dependencies
pip install opencv-python ultralytics

# For Streamlit web app with AI chat
pip install streamlit openai python-dotenv

# Additional dependencies may be needed:
pip install torch torchvision  # For YOLO model inference
pip install numpy matplotlib   # Common CV dependencies
```

### Running the Application
```bash
python app.py                  # Run basic surveillance application
streamlit run streamlit_app.py # Run interactive Streamlit web app with AI chat
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

#### Interactive Streamlit App (`streamlit_app.py`)
Web-based application with AI-powered chat interface for dynamic object filtering.

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

### Environment Variables (Streamlit App)
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
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