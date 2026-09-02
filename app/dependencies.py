import os

import cv2
import redis
from fastapi import Depends
from functools import lru_cache

from app.alerts import AlertManager
from app.detector import HogPersonDetector, OnnxPersonDetector, build_detector


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
def get_detector() -> HogPersonDetector | OnnxPersonDetector:
    return build_detector()


def get_frame_source() -> cv2.VideoCapture:
    source = os.environ.get("VIDEO_SOURCE", "0")
    source = int(source) if source.isdigit() else source
    return cv2.VideoCapture(source)
