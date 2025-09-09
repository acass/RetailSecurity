import sys
import os
import time
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.video_manager import VideoManager  # noqa: E402

@pytest.fixture
def mock_video_capture():
    """Fixture to create a mock cv2.VideoCapture."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
    return mock_cap

def test_video_manager_init():
    """Test the initialization of the VideoManager."""
    manager = VideoManager(source=0)
    assert manager.source == 0

def test_start_stream(mock_video_capture):
    """Test the start_stream method."""
    with patch('api.video_manager.cv2.VideoCapture', return_value=mock_video_capture):
        manager = VideoManager(source=0)
        assert manager.start_stream()
        assert manager.is_running
        assert manager.capture_thread is not None
        manager.stop_stream()

def test_stop_stream(mock_video_capture):
    """Test the stop_stream method."""
    with patch('api.video_manager.cv2.VideoCapture', return_value=mock_video_capture):
        manager = VideoManager(source=0)
        manager.start_stream()
        time.sleep(0.1)  # allow thread to start
        manager.stop_stream()
        assert not manager.is_running
        mock_video_capture.release.assert_called_once()

def test_get_current_frame(mock_video_capture):
    """Test the get_current_frame method."""
    with patch('api.video_manager.cv2.VideoCapture', return_value=mock_video_capture):
        manager = VideoManager(source=0)
        manager.start_stream()
        time.sleep(0.1)  # allow frame to be captured
        frame = manager.get_current_frame()
        assert frame is not None
        assert frame.shape == (100, 100, 3)
        manager.stop_stream()
