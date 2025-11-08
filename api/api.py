import threading
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config_loader import get_api_config, get_camera_config
from video_manager import VideoManager
from detection_service import DetectionService, Detection
from query_processor import QueryProcessor

# Global service instances
video_manager: Optional[VideoManager] = None
detection_service: Optional[DetectionService] = None
query_processor: Optional[QueryProcessor] = None
detection_threads: Dict[str, threading.Thread] = {}
is_processing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global video_manager, detection_service, query_processor
    video_manager = VideoManager()
    detection_service = DetectionService()
    query_processor = QueryProcessor(detection_service)
    print(f"Services initialized with {len(get_camera_config())} cameras.")
    yield
    await stop_stream(None)


app = FastAPI(
    title="Retail Security Surveillance API",
    description="AI-powered surveillance system with multi-camera support",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    camera_name: Optional[str] = None


class QueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    detection_context: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]


class StreamStatus(BaseModel):
    active: bool
    source: Optional[Any] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    has_current_frame: bool = False


def run_continuous_detection(camera_name: str):
    """Background task for continuous object detection on a specific camera."""
    while is_processing and video_manager and detection_service:
        frame = video_manager.get_current_frame(camera_name)
        if frame is not None:
            detection_service.detect_objects(frame, camera_name)
        time.sleep(0.1)


@app.get("/")
async def root():
    return {"message": "Retail Security Surveillance API v1.1.0"}


@app.get("/cameras")
async def get_cameras():
    """Get the list of available cameras from the configuration."""
    if not video_manager:
        raise HTTPException(status_code=503, detail="Video manager not initialized")
    return {"cameras": video_manager.get_available_cameras()}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language query about detections."""
    if not query_processor or not video_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    camera_name = request.camera_name
    if not camera_name or not video_manager.is_stream_active(camera_name):
        raise HTTPException(status_code=400, detail=f"Camera '{camera_name}' is not active.")

    result = query_processor.process_query(request.query, camera_name)
    return QueryResponse(
        success=result["success"],
        query=result["query"],
        answer=result["answer"],
        detection_context=result.get("detection_context"),
        error=result.get("error")
    )


@app.get("/stream/status", response_model=Dict[str, StreamStatus])
async def get_stream_status():
    """Get the status of all video streams."""
    if not video_manager:
        raise HTTPException(status_code=503, detail="Video manager not initialized")
    return video_manager.get_all_stream_info()


@app.post("/stream/start")
async def start_stream(camera_name: Optional[str] = None):
    """Start a specific video stream or all streams."""
    global is_processing
    if not video_manager:
        raise HTTPException(status_code=503, detail="Video manager not initialized")
    
    success = video_manager.start_stream(camera_name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to start stream(s).")
    
    is_processing = True
    cameras_to_process = [camera_name] if camera_name else [c['name'] for c in video_manager.get_available_cameras()]
    for cam_name in cameras_to_process:
        if cam_name not in detection_threads or not detection_threads[cam_name].is_alive():
            thread = threading.Thread(target=run_continuous_detection, args=(cam_name,), daemon=True)
            detection_threads[cam_name] = thread
            thread.start()

    return {"message": f"Stream started for {'all cameras' if not camera_name else camera_name}."}


@app.post("/stream/stop")
async def stop_stream(camera_name: Optional[str] = None):
    """Stop a specific video stream or all streams."""
    global is_processing
    if not video_manager:
        raise HTTPException(status_code=503, detail="Video manager not initialized")
    
    video_manager.stop_stream(camera_name)
    
    # If all streams are stopped, stop processing
    if not any(s['active'] for s in video_manager.get_all_stream_info().values()):
        is_processing = False
        for thread in detection_threads.values():
            thread.join(timeout=1.0)
        detection_threads.clear()

    return {"message": f"Stream stopped for {'all cameras' if not camera_name else camera_name}."}


@app.get("/detections/current", response_model=Dict)
async def get_current_detections(camera_name: str):
    """Get current detections for a specific camera."""
    if not detection_service or not video_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    if not video_manager.is_stream_active(camera_name):
        raise HTTPException(status_code=400, detail=f"Stream for {camera_name} is not active.")

    detections = detection_service.get_current_detections(camera_name)
    detection_results = [
        DetectionResult(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=list(d.bbox)
        ) for d in detections
    ]
    return {
        "camera": camera_name,
        "detections": detection_results,
        "summary": detection_service.get_detection_summary(camera_name)
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
    
    api_config = get_api_config()
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    
    print(f"Starting Retail Security Surveillance API on {host}:{port}")
    print("Make sure to set OPENAI_API_KEY in your environment for query processing")
    
    uvicorn.run(app, host=host, port=port)