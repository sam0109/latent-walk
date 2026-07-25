from fastapi.testclient import TestClient

from latent_walk.app import app


def test_index_and_health() -> None:
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["ok"] is True


def test_websocket_rejects_non_native_resolution() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws?size=123") as socket:
        assert socket.receive_json() == {
            "type": "error",
            "message": "SDXL-Turbo runs at 512 × 512.",
        }
