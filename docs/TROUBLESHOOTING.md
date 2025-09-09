# Troubleshooting Guide

This guide helps you resolve common issues you might encounter while installing or running the Retail Security Surveillance Application.

## Installation Issues

### `pip install` fails

If the `pip install -r config/requirements.txt` command fails, it could be due to several reasons:

*   **Network issues**: Ensure you have a stable internet connection.
*   **Python/pip version**: Make sure you are using a supported version of Python (3.7+) and that `pip` is up to date (`pip install --upgrade pip`).
*   **System dependencies**: Some packages may require system-level libraries. Check the error messages for clues. For example, on a fresh Debian/Ubuntu system, you might need to install `build-essential` or `python3-dev`.

### NumPy Compatibility on Intel Macs

If you are using an Intel-based Mac and encounter errors related to `numpy` after installation, it might be a compatibility issue with `opencv-python`. To fix this, install specific versions of these packages:

```bash
pip install "numpy<2" "opencv-python<4.11"
```

## Application Runtime Issues

### Camera Not Working

If the application starts but you don't see a video feed, follow these steps to diagnose the problem.

#### 1. Check `CAMERA_SOURCE` in `.env` file

*   **For a webcam**: Ensure `CAMERA_SOURCE` is set to the correct index. `0` is usually the default built-in webcam. If you have multiple webcams, try `1`, `2`, etc.
*   **For an RTSP stream**: Double-check the URL for typos. Ensure the camera is on the same network and that your computer can reach it.

#### 2. Test the Camera Independently

You can use a small Python script to test your camera connection without running the full application.

**Test a webcam:**

```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam works!' if cap.isOpened() else 'Webcam failed')"
```
(Replace `0` with your camera's index if it's not the default.)

**Test an RTSP stream:**

```bash
python -c "import cv2; cap = cv2.VideoCapture('rtsp://your_camera_ip:port/stream'); print('RTSP stream works!' if cap.isOpened() else 'RTSP stream failed')"
```
(Replace with your actual RTSP URL.)

If these scripts fail, the issue is with the camera or its connection, not the application itself.

### "No module named '...' " Error

If you get an error like `ModuleNotFoundError: No module named 'ultralytics'`, it means a required dependency is not installed in the Python environment you are using.

*   **Activate the virtual environment**: Make sure you have activated the virtual environment where you installed the dependencies (`source venv/bin/activate` or `.\venv\Scripts\activate`).
*   **Re-install dependencies**: Try running `pip install -r config/requirements.txt` again.

### Poor Performance or Low FPS

Real-time object detection is computationally intensive. If you are experiencing low frames-per-second (FPS):

*   **Use a smaller model**: In the application code, you can switch to a smaller YOLOv8 model like `yolov8n.pt` (nano) or `yolov8s.pt` (small). This will be faster but less accurate.
*   **Reduce camera resolution**: If your camera supports it, lower the resolution of the video stream.
*   **Use a GPU**: The application will run significantly faster on a computer with a dedicated NVIDIA GPU that supports CUDA. `torch` will automatically use the GPU if it's properly configured. On Macs, it will use the MPS backend.
*   **Close other applications**: Free up system resources by closing other programs.

### OpenAI API Key Issues

If you see errors related to the OpenAI API key when using the AI chat features:

*   **Check the `.env` file**: Ensure the `OPENAI_API_KEY` is set correctly in the `.env` file in the project's root directory.
*   **Verify the key**: Make sure the key itself is valid and has not been revoked. You can check your API key status on the [OpenAI Platform](https://platform.openai.com/).
*   **Check your billing status**: An expired trial or billing issues can cause API key errors.

## General Tips

*   **Read the terminal output**: Error messages in the terminal are your most important tool for diagnosing problems. Read them carefully.
*   **Consult the documentation**: Review the [Installation Guide](INSTALL.md) and [Usage Guide](USAGE.md) to ensure you haven't missed a step.
*   **Start fresh**: If you are completely stuck, you can try starting over. Delete the project folder, re-clone the repository, and follow the installation steps again carefully. This can often resolve complex configuration issues.
