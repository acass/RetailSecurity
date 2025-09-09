import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.query_processor import QueryProcessor  # noqa: E402
from api.detection_service import Detection  # noqa: E402

@pytest.fixture
def mock_detection_service():
    """Fixture to create a mock DetectionService."""
    mock_service = MagicMock()
    mock_service.get_current_detections.return_value = [
        Detection(class_id=0, class_name='person', confidence=0.9, bbox=(10, 20, 30, 40))
    ]
    mock_service.get_detection_summary.return_value = {
        "class_counts": {"person": 1},
        "total_detections": 1,
        "unique_classes": 1,
        "last_updated": "now"
    }
    mock_service.get_available_classes.return_value = ['person', 'car']
    return mock_service

@pytest.fixture
def mock_openai_client():
    """Fixture to create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "There is 1 person visible."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

def test_query_processor_init(mock_detection_service, monkeypatch):
    """Test the initialization of the QueryProcessor."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    with patch('api.query_processor.OpenAI') as mock_openai:
        processor = QueryProcessor(detection_service=mock_detection_service)
        assert processor.detection_service == mock_detection_service
        mock_openai.assert_called_once_with(api_key="test_key")

def test_process_query(mock_detection_service, mock_openai_client, monkeypatch):
    """Test the process_query method."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    with patch('api.query_processor.OpenAI', return_value=mock_openai_client):
        processor = QueryProcessor(detection_service=mock_detection_service)
        result = processor.process_query("How many people are there?")

        assert result['success']
        assert result['answer'] == "There is 1 person visible."
        mock_openai_client.chat.completions.create.assert_called_once()
