import sys
import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.detection_service import DetectionService, Detection  # noqa: E402

@pytest.fixture
def mock_yolo_model():
    """Fixture to create a mock YOLO model."""
    mock_model = MagicMock()
    # Mock the model's __call__ method to return a predictable result
    mock_box = MagicMock()
    mock_box.cls = [0]  # class_id
    mock_box.conf = [0.9]  # confidence
    mock_box.xyxy = [[10, 20, 30, 40]]  # bbox
    mock_results = MagicMock()
    mock_results.boxes = [mock_box]
    mock_model.return_value = [mock_results]
    return mock_model

def test_detection_service_init():
    """Test the initialization of the DetectionService."""
    with patch('api.detection_service.YOLO') as mock_yolo:
        service = DetectionService(model_path='fake/path.pt', confidence_threshold=0.6)
        mock_yolo.assert_called_once_with('fake/path.pt')
        assert service.confidence_threshold == 0.6

def test_detect_objects(mock_yolo_model):
    """Test the detect_objects method."""
    with patch('api.detection_service.YOLO', return_value=mock_yolo_model):
        service = DetectionService()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = service.detect_objects(frame)

        assert len(detections) == 1
        detection = detections[0]
        assert isinstance(detection, Detection)
        assert detection.class_id == 0
        assert detection.class_name == 'person'  # Assuming class 0 is 'person'
        assert detection.confidence == 0.9
        assert detection.bbox == (10, 20, 30, 40)
