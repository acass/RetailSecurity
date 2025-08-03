import cv2
import threading
import time
from typing import Optional, Tuple
import numpy as np


class VideoManager:
    """Manages video stream capture and frame access for the surveillance system."""
    
    def __init__(self, source: int | str = 0):
        """Initialize video manager with camera source.
        
        Args:
            source: Camera source (0 for webcam, RTSP URL for IP camera)
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        
    def start_stream(self) -> bool:
        """Start video stream capture.
        
        Returns:
            bool: True if stream started successfully, False otherwise
        """
        if self.is_running:
            return True
            
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Wait a moment for first frame
        time.sleep(0.5)
        return self.current_frame is not None
        
    def stop_stream(self):
        """Stop video stream capture."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        with self.frame_lock:
            self.current_frame = None
            
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current video frame.
        
        Returns:
            numpy.ndarray or None: Current frame if available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
            
    def is_stream_active(self) -> bool:
        """Check if video stream is active.
        
        Returns:
            bool: True if stream is running and has frames
        """
        return self.is_running and self.current_frame is not None
        
    def get_stream_info(self) -> dict:
        """Get information about the current stream.
        
        Returns:
            dict: Stream information including resolution, FPS, etc.
        """
        if not self.cap or not self.cap.isOpened():
            return {"active": False}
            
        return {
            "active": self.is_running,
            "source": str(self.source),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "has_current_frame": self.current_frame is not None
        }
        
    def _capture_frames(self):
        """Internal method to continuously capture frames in background thread."""
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                with self.frame_lock:
                    self.current_frame = frame
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"Video capture failed {max_failures} times, stopping stream")
                    break
                    
            time.sleep(0.03)  # ~30 FPS max
            
        self.is_running = False
        
    def __del__(self):
        """Cleanup on object destruction."""
        self.stop_stream()