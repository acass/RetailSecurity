import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.api import app  # noqa: E402

# client = TestClient(app)

# def test_health_check():
#     """
#     Tests the /health endpoint to ensure the API is running.
#     """
#     response = client.get("/health")
#     assert response.status_code == 200
#     assert response.json() == {"status": "ok"}
