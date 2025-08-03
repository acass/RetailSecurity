# Retail Security Surveillance Application

A Python-based retail security surveillance application that uses computer vision for real-time object detection and monitoring. The application leverages YOLOv8 (You Only Look Once) for object detection on video streams from security cameras, designed for retail loss prevention and security monitoring.

## Features

- **Real-time Object Detection**: Uses YOLOv8 for accurate and fast object detection
- **Multiple Camera Support**: Supports webcam and IP camera RTSP streams
- **AI-Powered Chat Interface**: Interactive Streamlit web app with natural language filtering
- **Retail-Focused**: Optimized for loss prevention and security monitoring
- **Multiple Application Types**: Choose between basic OpenCV app or interactive web interface

## Applications

### Basic Surveillance App (`app.py`)
Simple OpenCV-based application for direct video processing with YOLO detection.

### Interactive Streamlit App (`streamlit_app.py`)
Web-based application with AI-powered chat interface for dynamic object filtering.

### WebRTC Streamlit App (`streamlit_webrtc_app.py`)
Advanced web application with WebRTC support for enhanced streaming capabilities.

## Quick Start

### Prerequisites

- Python 3.x
- Webcam or IP camera with RTSP support

### Installation

1. Clone the repository:
```bash
git clone https://github.com/acass/RetailSecurity.git
cd RetailSecurity
```

2. Install dependencies:
```bash
# Basic dependencies
pip install opencv-python ultralytics

# For Streamlit web app with AI chat
pip install streamlit openai python-dotenv

# Additional dependencies
pip install torch torchvision numpy matplotlib
```

3. For the AI chat feature, create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

```bash
# Run basic surveillance application
python app.py

# Run interactive Streamlit web app with AI chat
streamlit run streamlit_app.py

# Run WebRTC-enabled Streamlit app
streamlit run streamlit_webrtc_app.py
```

## Configuration

### Camera Setup
- **Webcam**: Use `VideoCapture(0)` for default camera
- **IP Camera**: Update RTSP URL format: `"rtsp://username:password@ip_address:port/stream_path"`

### Model Selection
The application uses YOLOv8 models. You can choose different variants:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

### AI Chat Commands (Streamlit App)
Use natural language to filter detections:
- `"show only people"` - Filter to person class only
- `"cars and trucks"` - Show car and truck classes
- `"show all objects"` - Reset filter to show everything
- `"bottles and cups"` - Filter to bottle and cup classes

## Detection Classes

YOLOv8 detects 80 COCO classes including retail-relevant objects:
- **People & Vehicles**: person, bicycle, car, motorcycle, bus, truck
- **Retail Items**: bottle, wine glass, cup, handbag, tie, suitcase
- **Electronics**: laptop, mouse, remote, keyboard, cell phone
- **And many more**

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

### NumPy Compatibility (Intel Mac)
If you encounter NumPy 2.x compatibility errors:
```bash
pip install "numpy<2" "opencv-python<4.11"
```

### Camera Connection Issues
Test your camera connection:
```bash
# Test webcam
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam works:' if cap.isOpened() else 'Webcam failed')"

# Test RTSP stream
python -c "import cv2; cap = cv2.VideoCapture('rtsp://your_camera_ip:port/stream'); print('RTSP works:' if cap.isOpened() else 'RTSP failed')"
```

### Verify Installation
```bash
python -c "import cv2; from ultralytics import YOLO; print('All dependencies loaded successfully')"
```

## Hardware Requirements

- Camera with RTSP stream capability or USB webcam
- Adequate processing power for real-time YOLO inference
- GPU acceleration recommended for higher resolution or multiple streams

## Security Considerations

- Use secure RTSP credentials and encrypted connections
- Consider VPN access for remote camera monitoring
- Video frames are processed in real-time without storage by default
- No automatic recording or data persistence in current implementation

## License

This project is open source. Please ensure compliance with any applicable regulations regarding surveillance and privacy in your jurisdiction.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.