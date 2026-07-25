# Latent Walk

Latent Walk is an educational experiment that repeatedly passes the latest frame through
[SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) image-to-image
inference on an RTX 4090. Every decoded frame is re-encoded before the next
noise-and-denoise cycle, so approximate latent errors cannot accumulate
unchecked. There is no origin anchor or maximum drift radius.

Live deployment: <https://samdesktop.tail677e53.ts.net:10000/>

## How it works

1. The browser uploads and center-crops a starting image to 512×512.
2. SDXL-Turbo's VAE encodes the latest image into its spatial latent.
3. The img2img scheduler adds noise at the selected strength.
4. Optional noise, semantic-guidance, and image-memory strategies intervene in
   the trajectory.
5. The distilled SDXL denoiser performs 1–4 unconditional correction passes.
6. The VAE decodes a complete image, which is streamed as JPEG over WebSocket.
7. That decoded image—not an accumulating approximate latent—becomes the next
   starting point.

The request/response WebSocket protocol provides backpressure: the browser asks
for another frame only after receiving the previous one.

Only one WebSocket session can hold the GPU generation lease. Other visitors
see an occupied-studio screen and retry automatically. An idle session releases
the lease after two minutes.

## Run

```bash
cd /home/sam/latent-walk
uv sync
uv run uvicorn latent_walk.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The first walk step downloads and loads
SDXL-Turbo; later runs use the local Hugging Face cache. A CUDA GPU is required.

Create a password hash without putting the plaintext password in a shell
argument:

```bash
mkdir -p ~/.config/latent-walk
uv run python -c \
  'from argon2 import PasswordHasher; from getpass import getpass; print("LATENT_WALK_PASSWORD_HASH=" + PasswordHasher().hash(getpass()))' \
  > ~/.config/latent-walk/env
chmod 600 ~/.config/latent-walk/env
```

To expose it publicly through a password-gated Tailscale Funnel:

```bash
tailscale funnel --bg --https=10000 8000
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
- **Resolution** is fixed at SDXL-Turbo's 512×512 native working size.
- **Download video** encodes the current sequence as H.264 MP4. Each generated
  image receives exactly one frame interval at the selected playback FPS,
  regardless of generation or network speed. Up to 1,200 compressed frames are
  held in memory for the active session and discarded when it disconnects.

### Experimental interventions

Each intervention can run alone or alongside the others:

- **Frequency-shaped noise** replaces the scheduler's Gaussian proposal with
  independently weighted composition, form, and texture bands. The bands have
  persistent state across frames, but their combined tensor is normalized back
  to unit RMS before it enters the scheduler.
- **CLIP semantic velocity** maintains a tangent direction on the normalized
  CLIP image-embedding sphere. A differentiable VAE-to-CLIP path nudges
  non-terminal denoising latents toward the next semantic target. Gradients are
  spatially low-pass filtered and bounded; the final scheduler step is never
  modified because it has no later denoiser pass to project it back onto the
  image manifold.
- **IP-Adapter memory** conditions SDXL-Turbo on the previous frame, a lagged or
  random history frame, or an exponential moving average of image embeddings.
  The adapter and its ViT-H image encoder load only when first enabled.

Proposal noise, semantic direction, and conditioning selection use independent
deterministic random streams derived from the walk seed. Toggling one technique
therefore does not silently change another technique's random draws.

These are intentionally experimental controls, not stability guarantees.
SDXL-Turbo was not trained with the selected SDXL IP-Adapter, and strong
frequency shaping can enter colorful striped attractors, especially in
combination. Those failures are useful observations for this art project:
lower noise strength or disable one intervention to return to a more
interpretable region.

The server includes measured pixel drift and, while CLIP is enabled, semantic
drift and target similarity in each frame message.

The uploaded image is sent only to the server running on this machine.

## Development and deployment

Do not develop in the production checkout. Use a separate checkout or directory
and a non-production port for GPU validation, then deploy a reviewed commit as
one complete change. The public Funnel service should remain on the prior stable
commit while experimental work is in progress.

## Security

- Passwords are verified with Argon2; only the hash is stored outside Git.
- Sessions use signed, expiring, `HttpOnly`, `Secure`, `SameSite=Strict`
  cookies. The signing key is ephemeral and rotates on service restart.
- Login failures are throttled per source address.
- WebSockets require authentication and reject cross-origin browser requests.
- Content Security Policy and standard browser security headers are applied.
