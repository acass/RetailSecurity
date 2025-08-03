import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from video_manager import VideoManager
from detection_service import DetectionService, Detection
from query_processor import QueryProcessor

# Load environment variables
load_dotenv()

# Global service instances
video_manager: Optional[VideoManager] = None
detection_service: Optional[DetectionService] = None
query_processor: Optional[QueryProcessor] = None
detection_thread: Optional[threading.Thread] = None
is_processing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global video_manager, detection_service, query_processor
    
    # Default configuration
    source = int(os.getenv("CAMERA_SOURCE", "0")) if os.getenv("CAMERA_SOURCE", "0").isdigit() else os.getenv("CAMERA_SOURCE", "0")
    confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
    
    # Initialize services
    video_manager = VideoManager(source=source)
    detection_service = DetectionService(model_path=model_path, confidence_threshold=confidence)
    query_processor = QueryProcessor(detection_service)
    
    print(f"Services initialized with camera source: {source}")
    
    yield
    
    # Shutdown
    await stop_stream()


app = FastAPI(
    title="Retail Security Surveillance API",
    description="AI-powered surveillance system with natural language query capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    detection_context: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class StreamConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    source: Optional[int | str] = 0
    confidence_threshold: Optional[float] = 0.5
    model_path: Optional[str] = "yolov8n.pt"

class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]

class StreamStatus(BaseModel):
    active: bool
    source: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    has_current_frame: bool = False


def run_continuous_detection():
    """Background task to continuously run object detection."""
    global is_processing
    
    while is_processing and video_manager and detection_service:
        frame = video_manager.get_current_frame()
        if frame is not None:
            detection_service.detect_objects(frame)
        time.sleep(0.1)  # Process at ~10 FPS




@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Retail Security Surveillance API",
        "version": "1.0.0",
        "endpoints": {
            "query": "POST /query - Ask natural language questions about the video",
            "stream_status": "GET /stream/status - Check video stream status",
            "start_stream": "POST /stream/start - Start video capture",
            "stop_stream": "POST /stream/stop - Stop video capture",
            "current_detections": "GET /detections/current - Get current detections",
            "detection_classes": "GET /detections/classes - List available classes"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language query about current detections.
    
    Example queries:
    - "Are there any dogs in the patio?"
    - "How many people do you see?"
    - "What vehicles are visible?"
    - "Is there a laptop on the desk?"
    """
    if not query_processor:
        raise HTTPException(status_code=503, detail="Query processor not initialized")
    
    if not video_manager or not video_manager.is_stream_active():
        raise HTTPException(status_code=400, detail="Video stream is not active. Start the stream first.")
    
    result = query_processor.process_query(request.query)
    
    return QueryResponse(
        success=result["success"],
        query=result["query"],
        answer=result["answer"],
        detection_context=result.get("detection_context"),
        error=result.get("error")
    )


@app.get("/stream/status", response_model=StreamStatus)
async def get_stream_status():
    """Get current video stream status and information."""
    if not video_manager:
        return StreamStatus(active=False)
    
    info = video_manager.get_stream_info()
    return StreamStatus(
        active=info["active"],
        source=info.get("source"),
        width=info.get("width"),
        height=info.get("height"),
        fps=info.get("fps"),
        has_current_frame=info.get("has_current_frame", False)
    )


@app.post("/stream/start")
async def start_stream(config: Optional[StreamConfig] = None):
    """Start video stream capture with optional configuration."""
    global video_manager, detection_service, detection_thread, is_processing
    
    if config:
        # Reinitialize with new config
        if video_manager:
            video_manager.stop_stream()
        
        video_manager = VideoManager(source=config.source)
        
        if detection_service and config.confidence_threshold:
            detection_service.set_confidence_threshold(config.confidence_threshold)
    
    if not video_manager:
        raise HTTPException(status_code=503, detail="Video manager not initialized")
    
    success = video_manager.start_stream()
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start video stream. Check camera source.")
    
    # Start background detection processing
    if not is_processing:
        is_processing = True
        detection_thread = threading.Thread(target=run_continuous_detection, daemon=True)
        detection_thread.start()
    
    return {"message": "Video stream started successfully", "source": str(video_manager.source)}


@app.post("/stream/stop")
async def stop_stream():
    """Stop video stream capture."""
    global is_processing, detection_thread
    
    is_processing = False
    
    if detection_thread:
        detection_thread.join(timeout=2.0)
        detection_thread = None
    
    if video_manager:
        video_manager.stop_stream()
    
    return {"message": "Video stream stopped"}


@app.get("/detections/current")
async def get_current_detections():
    """Get current object detection results."""
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    if not video_manager or not video_manager.is_stream_active():
        raise HTTPException(status_code=400, detail="Video stream is not active")
    
    detections = detection_service.get_current_detections()
    detection_results = [
        DetectionResult(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=list(d.bbox)
        )
        for d in detections
    ]
    
    summary = detection_service.get_detection_summary()
    
    return {
        "detections": detection_results,
        "summary": summary
    }


@app.get("/detections/classes")
async def get_detection_classes():
    """Get list of all available object classes that can be detected."""
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    classes = detection_service.get_available_classes()
    return {"classes": classes, "total": len(classes)}


@app.get("/detections/summary")
async def get_detection_summary():
    """Get summary statistics of current detections."""
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    return detection_service.get_detection_summary()


@app.post("/detections/filter")
async def filter_detections(class_names: List[str]):
    """Get current detections filtered by specific class names."""
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    filtered_detections = detection_service.get_detections_by_class(class_names)
    detection_results = [
        DetectionResult(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=list(d.bbox)
        )
        for d in filtered_detections
    ]
    
    return {
        "filter": class_names,
        "detections": detection_results,
        "count": len(detection_results)
    }


@app.get("/query/suggestions")
async def get_query_suggestions():
    """Get example queries based on current detections."""
    if not query_processor:
        raise HTTPException(status_code=503, detail="Query processor not initialized")
    
    suggestions = query_processor.get_query_suggestions()
    return {"suggestions": suggestions}


@app.post("/config/confidence")
async def set_confidence_threshold(threshold: float):
    """Update confidence threshold for object detection."""
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")
    
    detection_service.set_confidence_threshold(threshold)
    return {"message": f"Confidence threshold updated to {threshold}"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "video_manager": video_manager is not None,
            "detection_service": detection_service is not None,
            "query_processor": query_processor is not None,
            "stream_active": video_manager.is_stream_active() if video_manager else False,
            "processing_active": is_processing
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Retail Security Surveillance API on {host}:{port}")
    print("Make sure to set OPENAI_API_KEY in your environment for query processing")
    
    uvicorn.run(app, host=host, port=port)