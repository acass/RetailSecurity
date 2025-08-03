import cv2
import time
import threading
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
import numpy as np
from dataclasses import dataclass


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
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """Initialize detection service.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.current_detections: List[Detection] = []
        self.detection_lock = threading.Lock()
        self.last_detection_time = 0
        
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Run object detection on a single frame.
        
        Args:
            frame: Input image frame
            
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
            self.current_detections = detections
            self.last_detection_time = time.time()
            
        return detections
        
    def get_current_detections(self) -> List[Detection]:
        """Get the most recent detection results.
        
        Returns:
            List of current Detection objects
        """
        with self.detection_lock:
            return self.current_detections.copy()
            
    def get_detections_by_class(self, class_names: List[str]) -> List[Detection]:
        """Get current detections filtered by class names.
        
        Args:
            class_names: List of class names to filter by
            
        Returns:
            List of filtered Detection objects
        """
        class_names_lower = [name.lower() for name in class_names]
        
        with self.detection_lock:
            filtered_detections = [
                detection for detection in self.current_detections
                if detection.class_name.lower() in class_names_lower
            ]
            
        return filtered_detections
        
    def has_object(self, class_name: str) -> bool:
        """Check if a specific object class is currently detected.
        
        Args:
            class_name: Name of the object class to check for
            
        Returns:
            bool: True if object is detected, False otherwise
        """
        class_name_lower = class_name.lower()
        
        with self.detection_lock:
            return any(
                detection.class_name.lower() == class_name_lower 
                for detection in self.current_detections
            )
            
    def count_objects(self, class_name: str) -> int:
        """Count how many instances of an object class are detected.
        
        Args:
            class_name: Name of the object class to count
            
        Returns:
            int: Number of detected instances
        """
        class_name_lower = class_name.lower()
        
        with self.detection_lock:
            return sum(
                1 for detection in self.current_detections
                if detection.class_name.lower() == class_name_lower
            )
            
    def get_detection_summary(self) -> Dict:
        """Get a summary of current detections.
        
        Returns:
            Dict with detection statistics and class counts
        """
        with self.detection_lock:
            if not self.current_detections:
                return {
                    "total_detections": 0,
                    "unique_classes": 0,
                    "class_counts": {},
                    "last_updated": self.last_detection_time
                }
                
            class_counts = {}
            for detection in self.current_detections:
                class_name = detection.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
            return {
                "total_detections": len(self.current_detections),
                "unique_classes": len(class_counts),
                "class_counts": class_counts,
                "last_updated": self.last_detection_time
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