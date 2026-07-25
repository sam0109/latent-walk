from __future__ import annotations

import io
import math
import threading
from dataclasses import dataclass

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageChops, ImageOps, ImageStat, UnidentifiedImageError

MODEL_ID = "stabilityai/sdxl-turbo"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
VALID_SIZES = {512}


class InvalidImageError(ValueError):
    pass


@dataclass
class WalkSettings:
    noise_strength: float = 0.45
    denoise_steps: int = 2

    @classmethod
    def from_message(cls, message: dict[str, object]) -> "WalkSettings":
        return cls(
            noise_strength=_bounded_float(
                message, "noiseStrength", 0.45, 0.15, 0.8
            ),
            denoise_steps=_bounded_int(message, "denoiseSteps", 2, 1, 4),
        )


def _bounded_float(
    message: dict[str, object], key: str, default: float, minimum: float, maximum: float
) -> float:
    value = message.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return min(max(float(value), minimum), maximum)


def _bounded_int(
    message: dict[str, object], key: str, default: int, minimum: int, maximum: int
) -> int:
    value = message.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return min(max(value, minimum), maximum)


def prepare_image(data: bytes, size: int) -> Image.Image:
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("The image is larger than 12 MB.")
    if size not in VALID_SIZES:
        raise InvalidImageError("Unsupported output size.")

    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            return ImageOps.fit(image, (size, size), Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The file is not a readable image.") from exc


class LatentWalk:
    def __init__(self, image: Image.Image) -> None:
        self.image = image.copy()
        self.step_number = 0

    def advance(self, image: Image.Image) -> float:
        difference = ImageChops.difference(self.image, image)
        change = sum(ImageStat.Stat(difference).rms) / (3 * 255)
        self.image = image
        self.step_number += 1
        return change


class ModelService:
    def __init__(self) -> None:
        self._pipeline: AutoPipelineForImage2Image | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def denoiser_loaded(self) -> bool:
        return self.loaded

    def load(self) -> AutoPipelineForImage2Image:
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "SDXL-Turbo requires a CUDA GPU; run this app on samdesktop."
                        )
                    pipeline = AutoPipelineForImage2Image.from_pretrained(
                        MODEL_ID,
                        torch_dtype=torch.float16,
                        variant="fp16",
                    )
                    pipeline.set_progress_bar_config(disable=True)
                    pipeline.to("cuda")
                    pipeline.vae.to(dtype=torch.float32)
                    self._pipeline = pipeline
        return self._pipeline

    def denoise_step(self, walk: LatentWalk, settings: WalkSettings) -> float:
        with self._inference_lock, torch.inference_mode():
            pipeline = self.load()
            total_steps = max(
                settings.denoise_steps,
                math.ceil(settings.denoise_steps / settings.noise_strength),
            )
            result = pipeline(
                prompt="",
                image=walk.image,
                strength=settings.noise_strength,
                num_inference_steps=total_steps,
                guidance_scale=0.0,
                output_type="pil",
            ).images[0]
            return walk.advance(result)

    def decode_jpeg(self, image: Image.Image, quality: int = 88) -> bytes:
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()


model_service = ModelService()
