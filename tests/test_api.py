"""
API tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_quick_check_endpoint():
    payload = {
        "anchor_text": "Reset forgot password",
        "current_turn_text": "How do I reset my forgotten password?",
        "previous_turn_text": "I can help with password recovery."
    }
    res = client.post("/api/v1/quick-check", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "anchor_distance" in data
    assert "is_drifting" in data
    assert data["anchor_distance"] < 0.60


def test_analyze_endpoint():
    payload = {
        "conversation_id": "test_api_conv",
        "anchor_goal": "Troubleshoot WiFi connectivity",
        "embedder": "tfidf",
        "turns": [
            {"speaker": "User", "text": "My router is blinking red and WiFi is disconnected."},
            {"speaker": "Tech", "text": "Please unplug the power cord for 30 seconds and plug it back in."}
        ]
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["conversation_id"] == "test_api_conv"
    assert data["total_turns"] == 2
    assert "cohesion_index" in data
