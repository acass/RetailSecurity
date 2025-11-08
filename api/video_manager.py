import cv2
import threading
import time
from typing import Optional, Dict, List
import numpy as np

from api.config_loader import get_camera_config


class VideoManager:
    """Manages multiple video streams for the surveillance system."""
    
    def __init__(self):
        """Initialize video manager with camera sources from config."""
        self.camera_configs = get_camera_config()
        if not self.camera_configs:
            raise ValueError("No camera sources found in the configuration.")
        
        self.streams: Dict[str, cv2.VideoCapture] = {}
        self.frames: Dict[str, Optional[np.ndarray]] = {}
        self.locks: Dict[str, threading.Lock] = {}
        self.capture_threads: Dict[str, threading.Thread] = {}
        self.is_running = False
        
        for cam in self.camera_configs:
            cam_name = cam.get('name')
            self.locks[cam_name] = threading.Lock()

    def start_stream(self, camera_name: Optional[str] = None) -> bool:
        """Start a specific or all video streams.

        Args:
            camera_name: Name of the camera to start. If None, starts all.
        
        Returns:
            bool: True if stream(s) started successfully.
        """
        self.is_running = True
        cameras_to_start = [c for c in self.camera_configs if c['name'] == camera_name] if camera_name else self.camera_configs
        
        for cam in cameras_to_start:
            name = cam['name']
            uri = cam['uri']
            
            cap = cv2.VideoCapture(uri)
            if not cap.isOpened():
                print(f"Error opening camera: {name}")
                continue
            
            self.streams[name] = cap
            thread = threading.Thread(target=self._capture_frames, args=(name,), daemon=True)
            self.capture_threads[name] = thread
            thread.start()
        
        time.sleep(1.0) # Allow time for frames to be captured
        return any(self.frames.values())

    def stop_stream(self, camera_name: Optional[str] = None):
        """Stop a specific or all video streams."""
        self.is_running = False
        cameras_to_stop = [camera_name] if camera_name else list(self.streams.keys())
        
        for name in cameras_to_stop:
            if name in self.capture_threads:
                self.capture_threads[name].join(timeout=2.0)

            if name in self.streams:
                self.streams[name].release()
        
        if camera_name:
            self.streams.pop(camera_name, None)
            self.frames[camera_name] = None
        else:
            self.streams.clear()
            for name in self.frames:
                self.frames[name] = None

    def get_current_frame(self, camera_name: str) -> Optional[np.ndarray]:
        """Get the current frame from a specific camera."""
        with self.locks.get(camera_name, threading.Lock()):
            frame = self.frames.get(camera_name)
            return frame.copy() if frame is not None else None

    def get_all_current_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Get all current frames from all active cameras."""
        all_frames = {}
        for name in self.streams.keys():
            all_frames[name] = self.get_current_frame(name)
        return all_frames

    def is_stream_active(self, camera_name: str) -> bool:
        """Check if a specific video stream is active."""
        return camera_name in self.streams and self.frames.get(camera_name) is not None
        
    def get_stream_info(self, camera_name: str) -> dict:
        """Get information about a specific stream."""
        if camera_name not in self.streams:
            return {"active": False}

        cap = self.streams[camera_name]
        return {
            "active": True,
            "source": self.camera_configs[[c['name'] for c in self.camera_configs].index(camera_name)]['uri'],
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "has_current_frame": self.frames.get(camera_name) is not None
        }

    def get_all_stream_info(self) -> Dict[str, dict]:
        """Get information for all active streams."""
        return {name: self.get_stream_info(name) for name in self.streams.keys()}
        
    def get_available_cameras(self) -> List[Dict]:
        """Return the list of configured cameras."""
        return self.camera_configs

    def _capture_frames(self, camera_name: str):
        """Internal method to continuously capture frames for a camera."""
        cap = self.streams.get(camera_name)
        lock = self.locks.get(camera_name)
        if not cap or not lock:
            return
            
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                with lock:
                    self.frames[camera_name] = frame
            time.sleep(0.03)

    def __del__(self):
        """Cleanup on object destruction."""
        self.stop_stream()