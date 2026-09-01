import fakeredis
import pytest
from fastapi.testclient import TestClient

import app.dependencies as deps
import app.main as main_module
from app.alerts import AlertManager


class FakeDetector:
    pass


def fake_get_detector():
    return FakeDetector()


def fake_get_redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def client():
    main_module.get_detector = fake_get_detector
    main_module.get_redis_client = fake_get_redis_client

    main_module.app.dependency_overrides[deps.get_alert_manager] = (
        lambda: AlertManager(fakeredis.FakeRedis(decode_responses=True), cooldown_seconds=30)
    )
    main_module.app.dependency_overrides[deps.get_detector] = fake_get_detector

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recent_alerts_empty(client):
    response = client.get("/alerts/recent")
    assert len(response.json()["alerts"]) == 0