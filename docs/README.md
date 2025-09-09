# Retail Security Surveillance Application

This repository contains a Python-based retail security surveillance application that uses computer vision for real-time object detection and monitoring. It leverages the YOLOv8 object detection model and provides several interfaces for interaction, including a simple command-line viewer, a web-based UI, and a REST API.

## Documentation

For detailed information on how to install, use, and troubleshoot the application, please refer to the following documents:

*   **[Installation Guide](INSTALL.md)**: Step-by-step instructions to set up the project on your local machine.
*   **[Usage Guide](USAGE.md)**: Detailed explanation of how to run the different application components and interact with them.
*   **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Solutions to common problems and errors.

## Features

*   **Real-time Object Detection**: Uses YOLOv8 for accurate and fast object detection.
*   **Multiple Interfaces**: Choose between a basic OpenCV app, an advanced Streamlit web UI, or a FastAPI REST API.
*   **AI-Powered Chat**: Use natural language to filter and query object detections in the web interface.
*   **Flexible Camera Support**: Works with both USB webcams and IP cameras via RTSP streams.
*   **Retail-Focused**: Optimized for loss prevention, inventory monitoring, and customer analytics.

## Quick Overview of Applications

*   **`apps/Basic.py`**: A minimal command-line viewer for the video feed with detections.
*   **`api/api.py`**: A robust FastAPI backend that serves the video stream and handles API requests.
*   **`apps/enhanced_surveillance_ui.py`**: The main user-facing application; a rich web interface that requires the API to be running.
*   **`apps/streamlit_webrtc_app.py`**: An alternative, self-contained web application.

For instructions on how to run these, please see the [Usage Guide](USAGE.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source. Please ensure compliance with any applicable regulations regarding surveillance and privacy in your jurisdiction.