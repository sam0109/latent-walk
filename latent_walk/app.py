from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import (
    VALID_SIZES,
    InvalidImageError,
    LatentWalk,
    WalkSettings,
    model_service,
    prepare_image,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Latent Walk", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "modelLoaded": model_service.loaded,
        "denoiserLoaded": model_service.denoiser_loaded,
    }


async def send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


@app.websocket("/ws")
async def walk_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw_size = websocket.query_params.get("size", "384")
        try:
            size = int(raw_size)
        except ValueError:
            await send_error(websocket, "Invalid output size.")
            return
        if size not in VALID_SIZES:
            await send_error(websocket, "SDXL-Turbo runs at 512 × 512.")
            return

        await websocket.send_json(
            {"type": "status", "message": "Preparing the starting image…"}
        )
        upload = await websocket.receive_bytes()
        try:
            pixels = await asyncio.to_thread(prepare_image, upload, size)
        except InvalidImageError as exc:
            await send_error(websocket, str(exc))
            return

        walk = LatentWalk(pixels)
        initial = await asyncio.to_thread(model_service.decode_jpeg, walk.image)
        await websocket.send_json(
            {"type": "ready", "step": 0, "distance": 0.0, "size": size}
        )
        await websocket.send_bytes(initial)

        while True:
            payload = await websocket.receive_text()
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                await send_error(websocket, "Invalid control message.")
                continue
            if not isinstance(message, dict) or message.get("type") != "step":
                await send_error(websocket, "Unknown control message.")
                continue

            try:
                settings = WalkSettings.from_message(message)
            except ValueError as exc:
                await send_error(websocket, str(exc))
                continue

            if not model_service.denoiser_loaded:
                await websocket.send_json(
                    {
                        "type": "status",
                        "message": "Loading SDXL-Turbo on the RTX 4090…",
                    }
                )
            change = await asyncio.to_thread(
                model_service.denoise_step, walk, settings
            )
            frame = await asyncio.to_thread(model_service.decode_jpeg, walk.image)
            await websocket.send_json(
                {
                    "type": "frame",
                    "step": walk.step_number,
                    "change": round(change, 4),
                }
            )
            await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        return
    except RuntimeError as exc:
        await send_error(websocket, f"Model error: {exc}")
