from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .auth import (
    COOKIE_NAME,
    SESSION_SECONDS,
    issue_session,
    verify_password,
    verify_session,
)
from .model import (
    VALID_SIZES,
    InvalidImageError,
    LatentWalk,
    WalkSettings,
    model_service,
    prepare_image,
)

STATIC_DIR = Path(__file__).parent / "static"
LEASE_IDLE_SECONDS = 120
MAX_LOGIN_FAILURES = 8
LOGIN_WINDOW_SECONDS = 5 * 60

app = FastAPI(title="Latent Walk", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
generation_lease = asyncio.Lock()
login_failures: dict[str, deque[float]] = defaultdict(deque)
global_login_failures: deque[float] = deque()


class LoginSubmission(BaseModel):
    password: str


def _client_key(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and peer in {"127.0.0.1", "::1"}:
        return forwarded.rsplit(",", 1)[-1].strip()
    return peer


def _trim_failures(failures: deque[float], now: float) -> None:
    while failures and failures[0] < now - LOGIN_WINDOW_SECONDS:
        failures.popleft()


def _login_is_limited(key: str) -> bool:
    now = time.monotonic()
    failures = login_failures[key]
    _trim_failures(failures, now)
    _trim_failures(global_login_failures, now)
    return len(failures) >= MAX_LOGIN_FAILURES or len(global_login_failures) >= 40


@app.middleware("http")
async def security_and_auth(request: Request, call_next) -> Response:
    path = request.url.path
    public_path = (
        path == "/login"
        or path == "/api/login"
        or path == "/api/health"
        or path.startswith("/static/")
    )
    if not public_path and not verify_session(request.cookies.get(COOKIE_NAME)):
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' blob: data:; "
        "connect-src 'self' ws: wss:; "
        "media-src 'self' blob:; "
        "frame-ancestors 'none'"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


@app.get("/login")
async def login_page(request: Request) -> Response:
    if verify_session(request.cookies.get(COOKIE_NAME)):
        return RedirectResponse("/", status_code=303)
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/api/login")
async def login(submission: LoginSubmission, request: Request) -> Response:
    key = _client_key(request)
    if _login_is_limited(key):
        return JSONResponse(
            {"detail": "Too many attempts. Try again in a few minutes."},
            status_code=429,
        )
    try:
        valid = await asyncio.to_thread(verify_password, submission.password)
    except RuntimeError:
        return JSONResponse({"detail": "Password login is not configured."}, status_code=503)
    if not valid:
        failed_at = time.monotonic()
        login_failures[key].append(failed_at)
        global_login_failures.append(failed_at)
        return JSONResponse({"detail": "Invalid password."}, status_code=401)

    login_failures.pop(key, None)
    response = JSONResponse({"ok": True})
    response.set_cookie(
        COOKIE_NAME,
        issue_session(),
        max_age=SESSION_SECONDS,
        httponly=True,
        secure=True,
        samesite="strict",
        path="/",
    )
    return response


@app.post("/api/logout")
async def logout() -> Response:
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME, path="/", secure=True, samesite="strict")
    return response


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


@app.get("/api/status")
async def status() -> dict[str, bool]:
    return {"busy": generation_lease.locked()}


async def send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


@app.websocket("/ws")
async def walk_socket(websocket: WebSocket) -> None:
    if not verify_session(websocket.cookies.get(COOKIE_NAME)):
        await websocket.close(code=4401)
        return

    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host")
    if origin and urlparse(origin).netloc != host:
        await websocket.close(code=4403)
        return

    await websocket.accept()
    lease_acquired = False
    try:
        try:
            await asyncio.wait_for(generation_lease.acquire(), timeout=0.05)
            lease_acquired = True
        except TimeoutError:
            await websocket.send_json(
                {
                    "type": "busy",
                    "message": "Another visitor is generating images. This studio allows one walk at a time.",
                }
            )
            await websocket.close(code=1013)
            return

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
        upload = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
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
            payload = await asyncio.wait_for(
                websocket.receive_text(), timeout=LEASE_IDLE_SECONDS
            )
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
    except TimeoutError:
        await websocket.send_json(
            {
                "type": "expired",
                "message": "The idle session was released for another visitor.",
            }
        )
        await websocket.close(code=1000)
    except RuntimeError as exc:
        await send_error(websocket, f"Model error: {exc}")
    finally:
        if lease_acquired:
            generation_lease.release()
