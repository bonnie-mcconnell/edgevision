import os
import secrets

import cv2
import redis
from fastapi import Depends, Header, HTTPException, WebSocket
from functools import lru_cache

from app.alerts import AlertManager
from app.detector import HogPersonDetector, OnnxDetector, build_detector


@lru_cache
def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        decode_responses=True,
    )


@lru_cache
def get_alert_manager(redis_client: redis.Redis = Depends(get_redis_client)) -> AlertManager:
    return AlertManager(
        redis_client, 
        cooldown_seconds=30
    )


@lru_cache
def get_detector() -> HogPersonDetector | OnnxDetector:
    return build_detector()


def get_frame_source() -> cv2.VideoCapture:
    source = os.environ.get("VIDEO_SOURCE", "0")
    source = int(source) if source.isdigit() else source
    return cv2.VideoCapture(source)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Dependency for HTTP routes. No-op if API_KEY isn't set in the 
    environment. Uses secrets.compare_digest to avoid timing vulnerabilities.
    """
    api_key = os.environ.get("API_KEY")
    if api_key is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def verify_api_key_ws(websocket: WebSocket, api_key: str | None = None) -> None:
    """
    Websocker dependency. Checks a query parameter instead (/ws/detections?api_key=...)
    because browser WebSocket clients can't set custom headers. Short circuits as a
    route dependency for an unauthorized request, so the unauthorized attempt doesn't
    trigger get_frame_source() before being rejected.
    """
    expected = os.environ.get("API_KEY")
    if expected is None:
        return
    if api_key is None or not secrets.compare_digest(api_key, expected):
        await websocket.close(code=1008, reason="Invalid or missing API key")
        raise HTTPException(status_code=401)