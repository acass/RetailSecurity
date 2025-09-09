# Usage Guide

This guide explains how to run and interact with the different components of the Retail Security Surveillance Application.

Before proceeding, make sure you have completed all the steps in the [Installation Guide](INSTALL.md).

## Application Overview

The project includes several applications, each serving a different purpose:

1.  **Basic Surveillance App (`apps/Basic.py`)**: A simple, command-line based application that displays a video feed with object detection overlays.
2.  **FastAPI REST API Server (`api/api.py`)**: A backend server that provides programmatic access to the surveillance system.
3.  **Enhanced Surveillance UI (`apps/enhanced_surveillance_ui.py`)**: An advanced web-based interface with live streaming, AI chat, and system monitoring. **This is the recommended application for most users.**
4.  **WebRTC Streamlit App (`apps/streamlit_webrtc_app.py`)**: The original web application with WebRTC support.

## Running the Applications

You can run each application from the root directory of the project. Ensure your virtual environment is activated.

### 1. Basic Surveillance App

This application is useful for quick tests and simple monitoring. It will open a window showing your camera feed with detected objects boxed and labeled.

To run it, execute the following command in your terminal:

```bash
python apps/Basic.py
```

Press the 'q' key to close the video window and stop the application.

### 2. FastAPI REST API Server

The API server is the core of the system, handling video processing and AI queries. It is required for the *Enhanced Surveillance UI*.

To start the server:

```bash
python api/api.py
```

For development, it's often useful to run the server with auto-reload, which automatically restarts the server when you make code changes:

```bash
uvicorn api.api:app --reload
```

Once running, the API is accessible at `http://localhost:8000`. You can explore the interactive API documentation at `http://localhost:8000/docs`.

### 3. Enhanced Surveillance UI (Recommended)

This is the main graphical interface for the application. It provides a rich user experience with live video, detection controls, and an AI-powered chat.

**Prerequisite**: The FastAPI REST API server must be running before you start the Enhanced Surveillance UI.

To launch the web interface:

```bash
streamlit run apps/enhanced_surveillance_ui.py
```

This will open a new tab in your web browser with the application's UI.

### 4. WebRTC Streamlit App

This is an alternative web interface. It can be run independently of the FastAPI server.

```bash
streamlit run apps/streamlit_webrtc_app.py
```

## Interacting with the Applications

### AI Chat Commands (in Web UIs)

The Streamlit web applications (`enhanced_surveillance_ui.py` and `streamlit_webrtc_app.py`) include an AI chat interface. You can use natural language to filter the objects being detected in real-time.

*   **"show only people"**: Filters the display to only show objects classified as 'person'.
*   **"cars and trucks"**: Shows only 'car' and 'truck' detections.
*   **"show all objects"**: Resets the filter to display all detected object classes.
*   **"bottles and cups"**: Filters for 'bottle' and 'cup' detections.

### Using the FastAPI REST API

If you are a developer, you can integrate other systems with the surveillance application using its REST API. The API provides endpoints for controlling the video stream, querying detections, and more.

Here are a few examples using `curl`.

**Start a video stream:**

```bash
curl -X POST "http://localhost:8000/stream/start" \
     -H "Content-Type: application/json" \
     -d '{"source": 0, "confidence_threshold": 0.5}'
```

**Ask a natural language question about the video feed:**

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "How many people do you see?"}'
```

**Get the list of currently detected objects:**

```bash
curl "http://localhost:8000/detections/current"
```

**Filter detections by class:**

```bash
curl -X POST "http://localhost:8000/detections/filter" \
     -H "Content-Type: application/json" \
     -d '["person", "car"]'
```

For a full list of available endpoints and their parameters, please refer to the interactive API documentation at `http://localhost:8000/docs` while the API server is running.

## Configuration

### Camera Source

You can change the camera source by editing the `CAMERA_SOURCE` variable in your `.env` file.
*   For a webcam, use its index (e.g., `0` for the first one, `1` for the second).
*   For an IP camera, use its RTSP URL, like `"rtsp://username:password@ip_address:port/stream_path"`.

### Detection Model

The application uses YOLOv8. You can change the model file used in the code to balance speed and accuracy. The models are located in the `ultralytics` library.
*   `yolov8n.pt` (Nano): Fastest, lowest accuracy.
*   `yolov8x.pt` (Extra Large): Slowest, highest accuracy.

This concludes the Usage Guide. If you encounter any issues, please refer to the [Troubleshooting Guide](TROUBLESHOOTING.md).
