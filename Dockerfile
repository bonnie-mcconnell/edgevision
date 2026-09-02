FROM python:3.12-slim AS builder
# ultralytics depends on full opencv-python (not -headless like this
# project's own requirements.txt), which needs these X11/GL libs present
# just to import successfully, even though export never opens a window
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ultralytics
RUN yolo export model=yolov8n.pt format=onnx imgsz=640

# runtime stage: what actually gets deployed2
FROM python:3.12-slim

# libgl1 + libglib2.0 are needed for opencv's video decode even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY --from=builder /yolov8n.onnx models/yolov8n.onnx

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV DETECTOR_BACKEND=onnx
ENV REDIS_HOST=redis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]