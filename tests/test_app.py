import io
import json
import zipfile

from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from PIL import Image

from latent_walk.app import app
from latent_walk.auth import COOKIE_NAME, issue_session, verify_session
from latent_walk.experiments import StepResult
from latent_walk.model import model_service


def test_login_protects_app(monkeypatch) -> None:
    monkeypatch.setenv("LATENT_WALK_PASSWORD_HASH", PasswordHasher().hash("test-password"))
    client = TestClient(app, base_url="https://testserver")

    redirect = client.get("/", follow_redirects=False)
    assert redirect.status_code == 303
    assert redirect.headers["location"] == "/login"
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401

    login = client.post("/api/login", json={"password": "test-password"})
    assert login.status_code == 200
    assert "HttpOnly" in login.headers["set-cookie"]
    assert "Secure" in login.headers["set-cookie"]
    assert "SameSite=strict" in login.headers["set-cookie"]
    assert client.get("/").status_code == 200


def test_health_is_public() -> None:
    client = TestClient(app)
    assert client.get("/api/health").json()["ok"] is True


def test_issued_sessions_always_verify() -> None:
    assert all(verify_session(issue_session()) for _ in range(1000))


def test_websocket_rejects_non_native_resolution(monkeypatch) -> None:
    monkeypatch.setenv("LATENT_WALK_PASSWORD_HASH", PasswordHasher().hash("test-password"))
    client = TestClient(app, base_url="https://testserver")
    client.post("/api/login", json={"password": "test-password"})
    cookie = f"{COOKIE_NAME}={client.cookies.get(COOKIE_NAME)}"
    with client.websocket_connect("/ws?size=123", headers={"cookie": cookie}) as socket:
        assert socket.receive_json() == {
            "type": "error",
            "message": "SDXL-Turbo runs at 512 × 512.",
        }


def test_websocket_rejects_invalid_seed(monkeypatch) -> None:
    monkeypatch.setenv("LATENT_WALK_PASSWORD_HASH", PasswordHasher().hash("test-password"))
    client = TestClient(app, base_url="https://testserver")
    client.post("/api/login", json={"password": "test-password"})
    cookie = f"{COOKIE_NAME}={client.cookies.get(COOKIE_NAME)}"
    with client.websocket_connect(
        "/ws?size=512&seed=-1",
        headers={"cookie": cookie},
    ) as socket:
        assert socket.receive_json()["message"].startswith(
            "Seed must be between"
        )


def test_second_websocket_is_blocked(monkeypatch) -> None:
    monkeypatch.setenv("LATENT_WALK_PASSWORD_HASH", PasswordHasher().hash("test-password"))
    client = TestClient(app, base_url="https://testserver")
    client.post("/api/login", json={"password": "test-password"})
    cookie = f"{COOKIE_NAME}={client.cookies.get(COOKIE_NAME)}"
    image = Image.new("RGB", (32, 32), "purple")
    upload = io.BytesIO()
    image.save(upload, "PNG")

    with client.websocket_connect("/ws?size=512", headers={"cookie": cookie}) as first:
        assert first.receive_json()["type"] == "status"
        first.send_bytes(upload.getvalue())
        assert first.receive_json()["type"] == "ready"
        first.receive_bytes()
        with client.websocket_connect(
            "/ws?size=512", headers={"cookie": cookie}
        ) as second:
            message = second.receive_json()
            assert message["type"] == "busy"


def test_websocket_records_and_exports_experiment(monkeypatch) -> None:
    monkeypatch.setenv("LATENT_WALK_PASSWORD_HASH", PasswordHasher().hash("test-password"))
    monkeypatch.setattr(model_service, "loading_message", lambda settings: None)

    def fake_step(walk, settings):
        image = Image.new("RGB", (512, 512), "orange")
        change = walk.advance(image)
        return StepResult(
            image,
            change,
            semantic_change=0.1,
            metrics={"pixelRms": change, "lpips": 0.2},
            effective_parameters={"ipAdapterWeight": 0.15},
        )

    monkeypatch.setattr(model_service, "denoise_step", fake_step)
    client = TestClient(app, base_url="https://testserver")
    client.post("/api/login", json={"password": "test-password"})
    cookie = f"{COOKIE_NAME}={client.cookies.get(COOKIE_NAME)}"
    upload = io.BytesIO()
    Image.new("RGB", (32, 32), "purple").save(upload, "PNG")

    with client.websocket_connect(
        "/ws?size=512&seed=42",
        headers={"cookie": cookie},
    ) as socket:
        assert socket.receive_json()["type"] == "status"
        socket.send_bytes(upload.getvalue())
        ready = socket.receive_json()
        assert ready["seed"] == 42
        socket.receive_bytes()

        socket.send_text(json.dumps({"type": "step"}))
        frame = socket.receive_json()
        assert frame["metrics"]["lpips"] == 0.2
        socket.receive_bytes()

        socket.send_text(json.dumps({"type": "exportManifest"}))
        assert socket.receive_json()["type"] == "manifest"
        manifest = json.loads(socket.receive_bytes())
        assert manifest["seed"] == 42
        assert manifest["steps"][0]["settings"]["noiseStrength"] == 0.45

        socket.send_text(json.dumps({"type": "exportBundle"}))
        assert socket.receive_json()["type"] == "status"
        assert socket.receive_json()["type"] == "bundle"
        with zipfile.ZipFile(io.BytesIO(socket.receive_bytes())) as bundle:
            assert "manifest.json" in bundle.namelist()
            assert "frames/000001.jpg" in bundle.namelist()
