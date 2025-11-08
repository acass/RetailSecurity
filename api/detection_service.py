import cv2
import time
import threading
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
import numpy as np
from dataclasses import dataclass

from api.config_loader import get_model_config


@dataclass
class Detection:
    """Data class for object detection results."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class DetectionService:
    """Service for running YOLO object detection on video frames."""
    
    # YOLO class names (COCO dataset)
    CLASS_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(self):
        """Initialize detection service using settings from config."""
        model_config = get_model_config()
        model_path = model_config.get("path", "../models/yolov8n.pt")
        self.confidence_threshold = model_config.get("confidence_threshold", 0.5)
        
        self.model = YOLO(model_path)
        self.current_detections: Dict[str, List[Detection]] = {}
        self.detection_lock = threading.Lock()
        self.last_detection_time: Dict[str, float] = {}
        
    def detect_objects(self, frame: np.ndarray, camera_name: str) -> List[Detection]:
        """Run object detection on a single frame for a specific camera.
        
        Args:
            frame: Input image frame
            camera_name: The name of the camera source
            
        Returns:
            List of Detection objects
        """
        if frame is None:
            return []
            
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        detections = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=self.CLASS_NAMES[class_id],
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    )
                    detections.append(detection)
        
        with self.detection_lock:
            self.current_detections[camera_name] = detections
            self.last_detection_time[camera_name] = time.time()
            
        return detections
        
    def get_current_detections(self, camera_name: str) -> List[Detection]:
        """Get the most recent detection results for a specific camera.
        
        Returns:
            List of current Detection objects
        """
        with self.detection_lock:
            return self.current_detections.get(camera_name, []).copy()
            
    def get_detections_by_class(self, class_names: List[str], camera_name: str) -> List[Detection]:
        """Get current detections for a camera, filtered by class names.
        
        Args:
            class_names: List of class names to filter by
            camera_name: The name of the camera source
            
        Returns:
            List of filtered Detection objects
        """
        class_names_lower = [name.lower() for name in class_names]
        
        with self.detection_lock:
            detections = self.current_detections.get(camera_name, [])
            filtered_detections = [
                detection for detection in detections
                if detection.class_name.lower() in class_names_lower
            ]
            
        return filtered_detections
        
    def has_object(self, class_name: str, camera_name: str) -> bool:
        """Check if a specific object class is currently detected on a camera.
        
        Args:
            class_name: Name of the object class to check for
            camera_name: The name of the camera source
            
        Returns:
            bool: True if object is detected, False otherwise
        """
        class_name_lower = class_name.lower()
        
        with self.detection_lock:
            detections = self.current_detections.get(camera_name, [])
            return any(
                detection.class_name.lower() == class_name_lower 
                for detection in detections
            )
            
    def count_objects(self, class_name: str, camera_name: str) -> int:
        """Count instances of an object class detected on a specific camera.
        
        Args:
            class_name: Name of the object class to count
            camera_name: The name of the camera source
            
        Returns:
            int: Number of detected instances
        """
        class_name_lower = class_name.lower()
        
        with self.detection_lock:
            detections = self.current_detections.get(camera_name, [])
            return sum(
                1 for detection in detections
                if detection.class_name.lower() == class_name_lower
            )
            
    def get_detection_summary(self, camera_name: str) -> Dict:
        """Get a summary of current detections for a specific camera.
        
        Returns:
            Dict with detection statistics and class counts
        """
        with self.detection_lock:
            detections = self.current_detections.get(camera_name, [])
            if not detections:
                return {
                    "total_detections": 0,
                    "unique_classes": 0,
                    "class_counts": {},
                    "last_updated": self.last_detection_time.get(camera_name)
                }
                
            class_counts = {}
            for detection in detections:
                class_name = detection.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
            return {
                "total_detections": len(detections),
                "unique_classes": len(class_counts),
                "class_counts": class_counts,
                "last_updated": self.last_detection_time.get(camera_name)
            }
            
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold for detections.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        
    def get_available_classes(self) -> List[str]:
        """Get list of all available object classes.
        
        Returns:
            List of class names that can be detected
        """
        return self.CLASS_NAMES.copy()
        
    def annotate_frame(self, frame: np.ndarray, detections: Optional[List[Detection]] = None) -> np.ndarray:
        """Draw detection annotations on a frame.
        
        Args:
            frame: Input frame to annotate
            detections: List of detections to draw (uses current if None)
            
        Returns:
            Annotated frame
        """
        if frame is None:
            return frame
            
        if detections is None:
            detections = self.get_current_detections()
            
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                       
        return annotated_frame