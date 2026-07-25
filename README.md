# Latent Walk

Latent Walk is an educational experiment that repeatedly passes the latest frame through
[SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) image-to-image
inference on an RTX 4090. Every decoded frame is re-encoded before the next
noise-and-denoise cycle, so approximate latent errors cannot accumulate
unchecked. There is no origin anchor or maximum drift radius.

## How it works

1. The browser uploads and center-crops a starting image to 512×512.
2. SDXL-Turbo's VAE encodes the latest image into its spatial latent.
3. The img2img scheduler adds noise at the selected strength.
4. The distilled SDXL denoiser performs 1–4 unconditional correction passes.
5. The VAE decodes a complete image, which is streamed as JPEG over WebSocket.
6. That decoded image—not an accumulating approximate latent—becomes the next
   starting point.

The request/response WebSocket protocol provides backpressure: the browser asks
for another frame only after receiving the previous one.

## Run

```bash
cd /home/sam/latent-walk
uv sync
uv run uvicorn latent_walk.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The first walk step downloads and loads
SDXL-Turbo; later runs use the local Hugging Face cache. A CUDA GPU is required.

To expose it privately to your tailnet:

```bash
tailscale serve --bg --https=10000 8000
```

The repository includes `deploy/latent-walk.service` for running the app as a
persistent systemd user service.

## Controls

- **Noise strength** selects how far up the diffusion noise schedule each cycle
  begins. Higher values permit more radical conceptual drift.
- **Denoising passes** controls how many distilled SDXL-Turbo correction steps
  are applied per frame.
- **Playback** controls how quickly the browser requests frames. Actual speed is
  limited by denoising time.
- **Resolution** trades CPU speed for output detail.

The uploaded image is sent only to the server running on this machine.
