"""Dev-only websocket viewer, for recording a demo. Not part of the production app.

The websocket only ever streams detection/alert JSON, never the video frames
themselves (see app/main.py) and on most systems only one process can hold
a webcam open at a time anyway, so this doesn't try to open its own camera.
It just prints a clean, live-updating view of what the server is seeing.

Usage:
    pip install websockets rich
    # in another terminal: redis-server &  then
    # VIDEO_SOURCE=0 DETECTOR_BACKEND=onnx uvicorn app.main:app --reload
    python scripts/view_stream.py
"""

import asyncio
import json

import websockets
from rich.console import Console
from rich.live import Live
from rich.table import Table

WS_URL = "ws://localhost:8000/ws/detections"


def build_table(data: dict) -> Table:
    table = Table(title="EdgeVision - live detections")
    table.add_column("label")
    table.add_column("confidence")
    table.add_column("box (x1, y1, x2, y2)")

    for det in data.get("detections", []):
        box = ", ".join(f"{v:.0f}" for v in det["box"])
        table.add_row(det["label"], f"{det['confidence']:.2f}", box)

    inference_ms = data.get("inference_ms", 0)
    caption = f"inference: {inference_ms:.1f} ms"

    alerts = data.get("alerts", [])
    if alerts:
        fired = ", ".join(f"{a['zone_name']}/{a['label']}" for a in alerts)
        caption += f"  |  ALERT FIRED: {fired}"

    table.caption = caption
    return table


async def main() -> None:
    console = Console()
    console.print(f"Connecting to {WS_URL} ...")

    async with websockets.connect(WS_URL) as ws:
        with Live(console=console, refresh_per_second=8) as live:
            async for message in ws:
                data = json.loads(message)
                if "error" in data:
                    console.print(f"[red]server error: {data['error']}[/red]")
                    break
                live.update(build_table(data))


if __name__ == "__main__":
    asyncio.run(main())