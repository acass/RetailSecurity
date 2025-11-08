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
import requests
import time
from typing import Dict, List, Optional, Any
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


class APIClient:
    """Client for communicating with the FastAPI backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 10

    def get_cameras(self) -> Dict[str, Any]:
        """Get list of available cameras."""
        try:
            response = self.session.get(f"{self.base_url}/cameras")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_stream(self, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """Start a video stream on the backend."""
        params = {"camera_name": camera_name} if camera_name else {}
        try:
            response = self.session.post(f"{self.base_url}/stream/start", params=params)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_stream(self, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """Stop a video stream on the backend."""
        params = {"camera_name": camera_name} if camera_name else {}
        try:
            response = self.session.post(f"{self.base_url}/stream/stop", params=params)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_stream_status(self) -> Dict[str, Any]:
        """Get stream statuses from the backend."""
        try:
            response = self.session.get(f"{self.base_url}/stream/status")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def query(self, question: str, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """Send a natural language query to the backend."""
        payload = {"query": question, "camera_name": camera_name}
        try:
            response = self.session.post(f"{self.base_url}/query", json=payload)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check if API is healthy and accessible."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_current_detections(self, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current detections from backend."""
        params = {"camera_name": camera_name} if camera_name else {}
        try:
            response = self.session.get(f"{self.base_url}/detections/current", params=params)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_confidence_threshold(self, threshold: float) -> Dict[str, Any]:
        """Update confidence threshold on backend."""
        try:
            response = self.session.post(f"{self.base_url}/config/confidence?threshold={threshold}")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}


class YOLOVideoProcessor(VideoProcessorBase):
    """Video processor that applies YOLO object detection to WebRTC video frames."""
    
    def __init__(self):
        self.model = YOLO("../models/yolov8n.pt")
        self.filtered_classes = None
        self.confidence_threshold = 0.5
        
    def set_filter(self, class_names):
        """Set which object classes to show"""
        if class_names is None or len(class_names) == 0:
            self.filtered_classes = None
        else:
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
        img = frame.to_ndarray(format="bgr24")
        
        results = self.model(img, verbose=False, conf=self.confidence_threshold)
        
        if self.filtered_classes is not None:
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
            
            if filtered_boxes:
                annotated_frame = img.copy()
                for box, cls, conf in zip(filtered_boxes, filtered_classes, filtered_confidences):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = YOLO_CLASS_NAMES[int(cls)]
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{class_name}: {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                annotated_frame = img
        else:
            annotated_frame = results[0].plot()
            
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


class OpenAIProcessor:
    """Processes natural language commands for local filtering."""
    
    def __init__(self):
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
            
            try:
                class_list = json.loads(result)
                if isinstance(class_list, list):
                    return class_list
            except json.JSONDecodeError:
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


def render_status_dashboard(api_client: APIClient):
    """Render the status dashboard showing system health and camera statuses."""
    st.subheader("System Status")
    
    health = api_client.health_check()
    if health["success"]:
        st.success("✅ API Backend Connected")
    else:
        st.error(f"❌ API Backend Disconnected: {health.get('error')}")
        return

    stream_status = api_client.get_stream_status()
    if stream_status["success"]:
        st.subheader("Camera Status")
        for cam_name, status in stream_status["data"].items():
            if status.get('active'):
                st.success(f"**{cam_name}**: Active - {status.get('width')}x{status.get('height')}")
            else:
                st.info(f"**{cam_name}**: Inactive")
    else:
        st.warning("⚠️ Could not retrieve stream statuses.")


def render_api_controls(api_client: APIClient):
    """Render API stream control buttons for multiple cameras."""
    st.subheader("API Stream Controls")
    
    cameras_res = api_client.get_cameras()
    if not cameras_res["success"]:
        st.error("Could not fetch camera list from API.")
        return

    camera_names = [cam['name'] for cam in cameras_res["data"].get("cameras", [])]

    selected_camera = st.selectbox("Select Camera", ["All"] + camera_names)

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎥 Start Stream", type="primary"):
            cam_to_start = selected_camera if selected_camera != "All" else None
            with st.spinner(f"Starting stream for {selected_camera}..."):
                result = api_client.start_stream(cam_to_start)
                if result["success"]:
                    st.success(f"Stream started for {selected_camera}!")
                else:
                    st.error(f"Failed to start stream: {result.get('error')}")
    
    with col2:
        if st.button("⏹️ Stop Stream"):
            cam_to_stop = selected_camera if selected_camera != "All" else None
            with st.spinner(f"Stopping stream for {selected_camera}..."):
                result = api_client.stop_stream(cam_to_stop)
                if result["success"]:
                    st.success(f"Stream stopped for {selected_camera}!")
                else:
                    st.error(f"Failed to stop stream: {result.get('error')}")

    # API Configuration
    st.subheader("API Configuration")
    api_confidence = st.slider(
        "API Confidence Threshold",
        min_value=0.1, 
        max_value=1.0, 
        value=st.session_state.get('api_confidence', 0.5),
        step=0.1,
        help="Confidence threshold for API detection"
    )
    if st.button("Update API Confidence"):
        result = api_client.set_confidence_threshold(api_confidence)
        if result["success"]:
            st.success(f"API confidence updated to {api_confidence}")
            st.session_state.api_confidence = api_confidence
        else:
            st.error(f"Failed to update confidence: {result.get('error')}")


def render_chat_interface(api_client: APIClient, openai_processor: OpenAIProcessor, webrtc_ctx):
    """Render the dual-mode chat interface."""
    st.subheader("AI Chat Interface")
    
    chat_mode = st.radio("Chat Mode:", ["Local Filtering", "API Queries"], help="Local: Filter WebRTC stream | API: Query backend detections")
    
    if chat_mode == "Local Filtering":
        st.markdown("**Local Mode**: Filter objects in the WebRTC stream")
        user_input = st.text_input("Filter Command:", placeholder="e.g., 'show only people', 'cars and trucks'", key="local_chat_input")
        
        if st.button("Apply Filter", type="primary"):
            if user_input.strip():
                with st.spinner("Processing filter command..."):
                    filtered_classes = openai_processor.process_command(user_input)
                    st.session_state.current_filter = filtered_classes
                    if webrtc_ctx.video_processor:
                        webrtc_ctx.video_processor.set_filter(filtered_classes)
                    st.success(f"Filter updated to: {filtered_classes or 'All'}")
            else:
                st.warning("Please enter a filter command")
    else:  # API Queries
        st.markdown("**API Mode**: Ask questions about backend detections")

        cameras_res = api_client.get_cameras()
        camera_names = [cam['name'] for cam in cameras_res.get("data", {}).get("cameras", [])]

        selected_camera = st.selectbox("Query Target Camera", camera_names, key="query_camera")

        user_input = st.text_input("Question:", placeholder="e.g., 'How many people do you see?'", key="api_chat_input")
        
        if st.button("Ask Question", type="primary"):
            if user_input.strip() and selected_camera:
                with st.spinner(f"Querying {selected_camera}..."):
                    result = api_client.query(user_input, selected_camera)
                    if result["success"]:
                        data = result["data"]
                        st.success(f"**Q:** {data['query']}")
                        st.info(f"**A:** {data['answer']}")
                        if data.get('detection_context'):
                            with st.expander("Detection Details"):
                                st.json(data['detection_context'])
                    else:
                        st.error(f"Query failed: {result.get('error')}")
            else:
                st.warning("Please select a camera and enter a question.")
    
    # Quick action buttons could be adapted similarly


def main():
    """Main Streamlit application function."""
    st.set_page_config(
        page_title="Enhanced AI Surveillance System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Enhanced AI Surveillance System")
    st.markdown("**Unified WebRTC streaming with FastAPI backend integration**")
    
    # Initialize session state
    if 'api_client' not in st.session_state:
        api_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
        st.session_state.api_client = APIClient(api_url)
    
    if 'openai_processor' not in st.session_state:
        st.session_state.openai_processor = OpenAIProcessor()
    
    if 'current_filter' not in st.session_state:
        st.session_state.current_filter = []
    
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.5
    
    # WebRTC Configuration
    rtc_configuration = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })
    
    # Main layout
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.subheader("🎥 Live WebRTC Video Stream")
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="enhanced-yolo-detection",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Update processor settings
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_filter(st.session_state.current_filter)
            webrtc_ctx.video_processor.set_confidence(st.session_state.confidence_threshold)
        
        # WebRTC Controls
        st.subheader("WebRTC Controls")
        
        st.session_state.confidence_threshold = st.slider(
            "WebRTC Confidence Threshold",
            min_value=0.1, 
            max_value=1.0, 
            value=st.session_state.confidence_threshold,
            step=0.1,
            help="Confidence threshold for WebRTC stream"
        )
        
        # Current filter display
        if st.session_state.current_filter:
            filter_text = ", ".join(st.session_state.current_filter)
            st.info(f"**Current Filter**: {filter_text}")
        else:
            st.info("**Current Filter**: All objects")
        
        if st.button("Reset WebRTC Filter"):
            st.session_state.current_filter = []
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.set_filter([])
            st.success("Filter reset!")
    
    with col2:
        # Status Dashboard
        render_status_dashboard(st.session_state.api_client)
        
        # API Controls
        render_api_controls(st.session_state.api_client)
    
    with col3:
        # Chat Interface
        render_chat_interface(
            st.session_state.api_client, 
            st.session_state.openai_processor, 
            webrtc_ctx
        )
    
    # Footer info
    st.markdown("---")
    st.markdown("**💡 Usage Tips:**")
    st.markdown("- Start the FastAPI server separately: `python api.py`")
    st.markdown("- Use **Local Filtering** to filter WebRTC stream objects")
    st.markdown("- Use **API Queries** to ask questions about backend detections")
    st.markdown("- Both systems can run simultaneously for comprehensive monitoring")


if __name__ == "__main__":
    main()