# Retail Security Surveillance Application

A Python-based retail security surveillance application that uses computer vision for real-time object detection and monitoring. The application leverages YOLOv8 (You Only Look Once) for object detection on video streams from security cameras, designed for retail loss prevention and security monitoring.

## Features

- **Real-time Object Detection**: Uses YOLOv8 for accurate and fast object detection
- **Multiple Camera Support**: Supports webcam and IP camera RTSP streams
- **AI-Powered Chat Interface**: Interactive Streamlit web app with natural language filtering
- **REST API Backend**: FastAPI server for programmatic access and integration
- **Natural Language Queries**: Ask questions about video content using OpenAI integration
- **Retail-Focused**: Optimized for loss prevention and security monitoring
- **Multiple Application Types**: Choose between basic OpenCV app, web interface, or REST API

## Quick Start

### Prerequisites

- Python 3.8+
- Webcam or IP camera with RTSP support
- OpenAI API key (for AI features)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/acass/Retail-Security.git
cd Retail-Security
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_api_key_here
```

4. The YOLO model will be automatically downloaded on first run, or you can manually download it:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Running the Applications

```bash
# Run basic surveillance application
cd apps
python Basic.py

# Run FastAPI REST API server
cd api
python api.py
# Or with auto-reload:
uvicorn api:app --reload

# Run enhanced Streamlit UI (requires FastAPI server running)
cd apps
streamlit run enhanced_surveillance_ui.py

# Run WebRTC-enabled Streamlit app
cd apps
streamlit run streamlit_webrtc_app.py

# Run OpenAI Vision Chat
cd apps
streamlit run openai_vision_chat.py
```

## Documentation

For detailed documentation, see [docs/README.md](docs/README.md)

## Project Structure

```
Retail-Security/
├── api/                    # FastAPI backend
│   ├── api.py             # Main API server
│   ├── detection_service.py
│   ├── query_processor.py
│   └── video_manager.py
├── apps/                   # Frontend applications
│   ├── Basic.py           # Basic OpenCV app
│   ├── enhanced_surveillance_ui.py  # Advanced Streamlit UI
│   ├── streamlit_webrtc_app.py     # WebRTC Streamlit app
│   └── openai_vision_chat.py       # OpenAI Vision Chat
├── config/                 # Configuration files
│   └── requirements.txt
├── docs/                   # Documentation
│   ├── README.md          # Detailed documentation
│   └── ...
├── models/                 # YOLO model files (auto-downloaded)
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Available Applications

### 1. Basic Surveillance App (`Basic.py`)
Simple OpenCV-based application for direct video processing with YOLO detection.

### 2. FastAPI REST API Server (`api.py`)
Production-ready REST API server with natural language query processing for surveillance video streams.

**API Documentation**: http://localhost:8000/docs (when running)

### 3. Enhanced Surveillance UI (`enhanced_surveillance_ui.py`)
Advanced Streamlit web application with dual WebRTC/API integration, featuring:
- Live WebRTC video streaming with real-time detection
- FastAPI backend integration for advanced queries
- Dual-mode AI chat (local filtering + API queries)
- System health monitoring and controls

### 4. WebRTC Streamlit App (`streamlit_webrtc_app.py`)
Web application with WebRTC support for enhanced streaming capabilities and AI-powered filtering.

### 5. OpenAI Vision Chat (`openai_vision_chat.py`)
Interactive chat interface using OpenAI's GPT-4 Vision to analyze video frames and answer questions about what's visible.

## Configuration

### Camera Setup
- **Webcam**: Use `0` for default camera, `1` for second camera, etc.
- **IP Camera**: Use RTSP URL format: `"rtsp://username:password@ip_address:port/stream_path"`

### Environment Variables
See `.env.example` for all available configuration options:
- `OPENAI_API_KEY` - Required for AI features
- `CAMERA_SOURCE` - Camera source (0 for webcam or RTSP URL)
- `CONFIDENCE_THRESHOLD` - Object detection confidence (0.0 to 1.0)
- `MODEL_PATH` - Path to YOLO model file
- `HOST` / `PORT` - API server configuration

## Hardware Requirements

- Camera with RTSP stream capability or USB webcam
- Adequate processing power for real-time YOLO inference
- GPU acceleration recommended for higher resolution or multiple streams

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

## License

This project is open source. Please ensure compliance with any applicable regulations regarding surveillance and privacy in your jurisdiction.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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

For more troubleshooting help, see [docs/README.md](docs/README.md).
