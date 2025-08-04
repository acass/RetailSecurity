import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import threading
from PIL import Image
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Load environment variables
load_dotenv()

# YOLO class names (COCO dataset)
YOLO_CLASS_NAMES = [
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

class YOLOVideoProcessor(VideoProcessorBase):
    """Video processor that applies YOLO object detection to WebRTC video frames.
    
    This class handles real-time object detection on video streams using YOLOv8,
    with support for filtering specific object classes and adjustable confidence thresholds.
    """
    def __init__(self):
        """Initialize the YOLO video processor with default settings."""
        self.model = YOLO("../models/yolov8n.pt")
        self.filtered_classes = None  # None means show all objects
        self.confidence_threshold = 0.5
        
    def set_filter(self, class_names):
        """Set which object classes to show"""
        if class_names is None or len(class_names) == 0:
            self.filtered_classes = None
        else:
            # Convert class names to indices
            self.filtered_classes = []
            for name in class_names:
                name_lower = name.lower().strip()
                for i, yolo_name in enumerate(YOLO_CLASS_NAMES):
                    if name_lower == yolo_name.lower():
                        self.filtered_classes.append(i)
                        break
    
    def set_confidence(self, confidence):
        """Set confidence threshold"""
        self.confidence_threshold = confidence
        
    def recv(self, frame):
        """Transform video frame with YOLO detection"""
        # Convert av.VideoFrame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO detection with optimized settings
        results = self.model(img, verbose=False, conf=self.confidence_threshold)
        
        # Filter detections if needed
        if self.filtered_classes is not None:
            # Create filtered results
            filtered_boxes = []
            filtered_classes = []
            filtered_confidences = []
            
            boxes = results[0].boxes
            if boxes is not None:
                for i, cls in enumerate(boxes.cls):
                    if int(cls) in self.filtered_classes:
                        filtered_boxes.append(boxes.xyxy[i])
                        filtered_classes.append(boxes.cls[i])
                        filtered_confidences.append(boxes.conf[i])
            
            # Create new result with filtered detections
            if filtered_boxes:
                # Manually draw filtered boxes
                annotated_frame = img.copy()
                for box, cls, conf in zip(filtered_boxes, filtered_classes, filtered_confidences):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = YOLO_CLASS_NAMES[int(cls)]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                annotated_frame = img
        else:
            # Show all detections
            annotated_frame = results[0].plot()
            
        # Convert back to av.VideoFrame
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

class OpenAIProcessor:
    """Processes natural language commands to filter object detection results.
    
    Uses OpenAI's GPT-4 to interpret user commands and convert them into
    specific object class filters for the YOLO detection system.
    """
    def __init__(self):
        """Initialize the OpenAI processor with API key from environment variables.
        
        Raises:
            Streamlit error and stops execution if OPENAI_API_KEY is not found.
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            st.stop()
        self.client = OpenAI(api_key=api_key)
        
    def process_command(self, user_input):
        """Process user command and return filtered object classes"""
        
        available_objects = ", ".join(YOLO_CLASS_NAMES)
        
        system_prompt = f"""You are an assistant for a video surveillance system. 
Your job is to interpret user requests about which objects to show in the video stream.

Available object types: {available_objects}

Rules:
1. If user says "show all" or "reset" or similar, return an empty list []
2. If user requests specific objects, return a list of the exact object names from the available list
3. Handle natural language like "show only people" -> ["person"], "cars and trucks" -> ["car", "truck"]
4. Be flexible with synonyms (e.g., "people" = "person", "bikes" = "bicycle")
5. Return ONLY a valid JSON list, nothing else

Examples:
- "show only people" -> ["person"]
- "cars and trucks" -> ["car", "truck"] 
- "show all objects" -> []
- "people and dogs" -> ["person", "dog"]
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                class_list = json.loads(result)
                if isinstance(class_list, list):
                    return class_list
            except json.JSONDecodeError:
                # Fallback: try to extract list from text
                match = re.search(r'\[(.*?)\]', result)
                if match:
                    items = match.group(1).split(',')
                    class_list = [
                        item.strip().strip('"').strip("'") 
                        for item in items if item.strip()
                    ]
                    return class_list
            
            return []
            
        except Exception as e:
            st.error(f"Error processing command: {e}")
            return []

def main():
    """Main Streamlit application function for the AI surveillance system.
    
    Sets up the WebRTC video stream with YOLO object detection and 
    AI-powered chat interface for dynamic object filtering.
    """
    st.set_page_config(
        page_title="AI Retail Surveillance - WebRTC",
        page_icon="📹",
        layout="wide"
    )
    
    st.title("AI Surveillance System")
    st.markdown("**Real-time object detection with smooth WebRTC streaming**")
    
    # Initialize session state
    if 'openai_processor' not in st.session_state:
        st.session_state.openai_processor = OpenAIProcessor()
    if 'current_filter' not in st.session_state:
        st.session_state.current_filter = []
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.5
        
    # WebRTC Configuration with STUN servers
    rtc_configuration = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Confidence threshold
    st.session_state.confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1, 
        max_value=1.0, 
        value=st.session_state.confidence_threshold,
        step=0.1,
        help="Minimum confidence for object detection"
    )
    
    # Current filter display
    st.sidebar.subheader("Current Filter")
    if st.session_state.current_filter:
        filter_text = ", ".join(st.session_state.current_filter)
        st.sidebar.info(f"Showing: {filter_text}")
    else:
        st.sidebar.info("Showing: All objects")
        
    # Quick reset button
    if st.sidebar.button("Show All Objects"):
        st.session_state.current_filter = []
        st.sidebar.success("Filter reset - showing all objects!")
    
    # Available objects
    st.sidebar.subheader("Available Objects")
    with st.sidebar.expander("View all detectable objects"):
        for i, obj in enumerate(YOLO_CLASS_NAMES):
            if i % 3 == 0:
                cols = st.columns(3)
            cols[i % 3].write(f"• {obj}")
    
    # Main layout: Video and Chat side by side
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live WebRTC Video Stream")
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="yolo-detection",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Update processor settings when available
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_filter(st.session_state.current_filter)
            webrtc_ctx.video_processor.set_confidence(st.session_state.confidence_threshold)
    
    with col2:
        st.subheader("AI Chat Interface")
        st.markdown("Which objects you want to see?")
        
        # Chat input
        user_input = st.text_input(
            "Command:",
            placeholder="e.g., 'show only people', 'cars and trucks', 'show all'",
            key="chat_input"
        )
        
        if st.button("Process Command", type="primary"):
            if user_input.strip():
                with st.spinner("Processing command..."):
                    # Process with OpenAI
                    filtered_classes = st.session_state.openai_processor.process_command(user_input)
                    
                    # Update filter
                    st.session_state.current_filter = filtered_classes
                    
                    # Update WebRTC processor if active
                    if webrtc_ctx.video_processor:
                        webrtc_ctx.video_processor.set_filter(filtered_classes)
                    
                    # Show result
                    if filtered_classes:
                        filter_text = ", ".join(filtered_classes)
                        st.success(f"Now showing: {filter_text}")
                    else:
                        st.success("Now showing: All objects")
            else:
                st.warning("Please enter a command")
        
        # Command examples
        st.subheader("Example Commands")
        examples = [
            "show only people",
            "cars and trucks",
            "show all objects",
            "person and dog",
            "bottles and cups",
            "reset filter"
        ]
        
        for example in examples:
            if st.button(f"{example}", key=f"example_{example}"):
                # Auto-process example
                filtered_classes = st.session_state.openai_processor.process_command(example)
                st.session_state.current_filter = filtered_classes
                
                # Update WebRTC processor if active
                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.set_filter(filtered_classes)
                
                if filtered_classes:
                    filter_text = ", ".join(filtered_classes)
                    st.success(f"Now showing: {filter_text}")
                else:
                    st.success("Now showing: All objects")
        
        # Performance info
        st.subheader("Performance Info")
        if webrtc_ctx.state.playing:
            st.success("WebRTC streaming active")
            st.info("Real-time processing with minimal latency")
        else:
            st.info("Click 'START' to begin streaming")

if __name__ == "__main__":
    main()