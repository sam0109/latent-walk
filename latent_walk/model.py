from __future__ import annotations

import io
import math
import os
import secrets
import threading
from dataclasses import dataclass, field

import av

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageChops, ImageOps, ImageStat, UnidentifiedImageError
from transformers import CLIPVisionModelWithProjection

from latent_walk.experiments import (
    ExperimentSettings,
    FrequencyNoiseStrategy,
    IpAdapterSettings,
    StagnationResult,
    StepResult,
    WalkState,
    gaussian_blur,
    semantic_stagnation,
)
from latent_walk.metrics import MetricSuite, image_tensor

torch.use_deterministic_algorithms(True)

MODEL_ID = "stabilityai/sdxl-turbo"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
VALID_SIZES = {512}


class InvalidImageError(ValueError):
    pass


@dataclass
class WalkSettings:
    noise_strength: float = 0.45
    denoise_steps: int = 2
    experiments: ExperimentSettings = field(default_factory=ExperimentSettings)

    @classmethod
    def from_message(cls, message: dict[str, object]) -> "WalkSettings":
        return cls(
            noise_strength=_bounded_float(
                message, "noiseStrength", 0.45, 0.15, 0.8
            ),
            denoise_steps=_bounded_int(message, "denoiseSteps", 2, 1, 4),
            experiments=ExperimentSettings.from_message(message),
        )

    def to_message(self) -> dict[str, object]:
        return {
            "noiseStrength": self.noise_strength,
            "denoiseSteps": self.denoise_steps,
            "experiments": self.experiments.to_message(),
        }


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


def encode_mp4(frames: list[bytes], fps: int) -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 512
        stream.height = 512
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}

        for jpeg in frames:
            with Image.open(io.BytesIO(jpeg)) as image:
                frame = av.VideoFrame.from_image(image.convert("RGB"))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


class LatentWalk(WalkState):
    def __init__(self, image: Image.Image, seed: int | None = None) -> None:
        super().__init__(
            image=image,
            seed=seed if seed is not None else secrets.randbits(63),
        )

    def advance(self, image: Image.Image) -> float:
        difference = ImageChops.difference(self.image, image)
        change = sum(ImageStat.Stat(difference).rms) / (3 * 255)
        super().advance(image)
        return change


class ClipGuidance:
    def __init__(
        self,
        settings: ExperimentSettings,
        vae: torch.nn.Module,
        image_encoder: CLIPVisionModelWithProjection,
        target: torch.Tensor,
    ) -> None:
        self.settings = settings
        self.vae = vae
        self.image_encoder = image_encoder
        self.target = target

    def __call__(
        self,
        pipeline: object,
        step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del timestep
        latents = callback_kwargs["latents"]
        guidance = self.settings.clip.guidance
        if guidance <= 0 or step_index >= int(pipeline.num_timesteps) - 1:
            return callback_kwargs

        self.vae.to(dtype=torch.float32)
        try:
            with torch.enable_grad():
                active = latents.detach().float().requires_grad_(True)
                pixels = self.vae.decode(
                    active / self.vae.config.scaling_factor,
                    return_dict=False,
                )[0]
                embedding = self._embed_pixels(pixels)
                loss = 1 - (embedding * self.target).sum(dim=-1).mean()
                gradient = torch.autograd.grad(loss, active)[0]
        finally:
            self.vae.to(dtype=torch.float16)

        gradient = gaussian_blur(gradient, sigma=1.5)
        gradient /= gradient.square().mean().sqrt().clamp_min(1e-6)
        remaining_steps = max(int(pipeline.num_timesteps) - step_index, 1)
        step_size = guidance / math.sqrt(remaining_steps)
        callback_kwargs["latents"] = (
            latents - step_size * gradient.to(latents.dtype)
        ).detach()
        return callback_kwargs

    def _embed_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = (pixels * 0.5 + 0.5).clamp(0, 1)
        pixels = self._resize_clip_pixels(pixels)
        mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073])
        std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711])
        pixels = (pixels - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        output = self.image_encoder(pixel_values=pixels.to(torch.float16))
        return F.normalize(output.image_embeds.float(), dim=-1)

    @staticmethod
    def _resize_clip_pixels(pixels: torch.Tensor) -> torch.Tensor:
        pixels = F.avg_pool2d(pixels, kernel_size=2, stride=2)
        return F.interpolate(
            pixels,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )


class ModelService:
    def __init__(self) -> None:
        self._pipeline: AutoPipelineForImage2Image | None = None
        self._clip_encoder: CLIPVisionModelWithProjection | None = None
        self._ip_adapter_loaded = False
        self._frequency_noise = FrequencyNoiseStrategy()
        self._metrics = MetricSuite()
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def denoiser_loaded(self) -> bool:
        return self.loaded

    def loading_message(self, settings: WalkSettings) -> str | None:
        if not self.loaded:
            return "Loading SDXL-Turbo on the RTX 4090…"
        if (
            self._tracks_semantics(settings.experiments)
            and self._clip_encoder is None
        ):
            return "Loading the CLIP semantic encoder…"
        if settings.experiments.ip_adapter.enabled and not self._ip_adapter_loaded:
            return "Loading the IP-Adapter image encoder…"
        if settings.experiments.metrics.enabled and not self._metrics.loaded:
            return "Loading perceptual metrics…"
        return None

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
                    pipeline.vae.requires_grad_(False)
                    pipeline.unet.requires_grad_(False)
                    pipeline.text_encoder.requires_grad_(False)
                    pipeline.text_encoder_2.requires_grad_(False)
                    self._pipeline = pipeline
        return self._pipeline

    def denoise_step(
        self,
        walk: LatentWalk,
        settings: WalkSettings,
    ) -> StepResult:
        with self._inference_lock:
            pipeline = self.load()
            total_steps = max(
                settings.denoise_steps,
                math.ceil(settings.denoise_steps / settings.noise_strength),
            )
            if walk.generator is None:
                walk.generator = torch.Generator(device="cuda").manual_seed(walk.seed)
                walk.semantic_generator = torch.Generator(
                    device="cuda"
                ).manual_seed(walk.seed ^ 0x434C4950)
                walk.conditioning_generator = torch.Generator(
                    device="cuda"
                ).manual_seed(walk.seed ^ 0x49504144)
                walk.escape_generator = torch.Generator(
                    device="cuda"
                ).manual_seed(walk.seed ^ 0x45534350)

            ip_kwargs, effective_ip_weight = self._prepare_ip_adapter(
                walk,
                settings.experiments,
            )
            callback, clip_target, escape_active, stagnation = (
                self._prepare_clip_guidance(
                    walk,
                    settings.experiments,
                )
            )
            if escape_active:
                result, escape_similarity = self._escape_step(
                    pipeline,
                    walk,
                    settings,
                    ip_kwargs,
                    effective_ip_weight,
                    stagnation,
                )
            else:
                custom_latents = self._prepare_custom_latents(
                    walk,
                    settings,
                    total_steps,
                )
                with torch.no_grad():
                    result = pipeline(
                        prompt="",
                        image=walk.image,
                        strength=settings.noise_strength,
                        num_inference_steps=total_steps,
                        guidance_scale=0.0,
                        generator=walk.generator,
                        latents=custom_latents,
                        callback_on_step_end=callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                        output_type="pil",
                        **ip_kwargs,
                    ).images[0]
                escape_similarity = None

            previous_image = walk.image
            change = walk.advance(result)
            semantic_change = self._update_clip_state(walk, settings.experiments)
            metrics = (
                self._metrics.compute(previous_image, result)
                if settings.experiments.metrics.enabled
                else {}
            )
            metrics["pixelRms"] = change
            if semantic_change is not None:
                metrics["semanticChange"] = semantic_change
            if clip_target is not None and walk.semantic_embedding is not None:
                target_similarity = float(
                    (walk.semantic_embedding * clip_target).sum()
                )
            else:
                target_similarity = None
            return StepResult(
                image=result,
                pixel_change=change,
                semantic_change=semantic_change,
                metrics=metrics,
                effective_parameters={
                    "frequency": settings.experiments.frequency.enabled,
                    "clip": settings.experiments.clip.enabled,
                    "escape": settings.experiments.escape.enabled,
                    "escapeActive": escape_active,
                    "semanticRadius": stagnation.radius,
                    "semanticProgress": stagnation.progress_ratio,
                    "semanticRevisit": stagnation.revisit_similarity,
                    "escapeSelectionSimilarity": escape_similarity,
                    "ipAdapter": settings.experiments.ip_adapter.enabled,
                    "ipAdapterWeight": effective_ip_weight,
                    "clipTargetSimilarity": target_similarity,
                },
            )

    def _prepare_custom_latents(
        self,
        walk: LatentWalk,
        settings: WalkSettings,
        total_steps: int,
    ) -> torch.Tensor | None:
        if not settings.experiments.frequency.enabled:
            return None

        pipeline = self._pipeline
        device = pipeline._execution_device
        image = pipeline.image_processor.preprocess(walk.image).to(
            device=device,
            dtype=torch.float32,
        )
        pipeline.vae.to(dtype=torch.float32)
        try:
            with torch.no_grad():
                encoded = pipeline.vae.encode(image).latent_dist.sample(
                    generator=walk.generator
                )
        finally:
            pipeline.vae.to(dtype=torch.float16)
        encoded = (
            encoded * pipeline.vae.config.scaling_factor
        ).to(dtype=torch.float16)

        pipeline.scheduler.set_timesteps(total_steps, device=device)
        init_timestep = min(
            int(total_steps * settings.noise_strength),
            total_steps,
        )
        timestep = pipeline.scheduler.timesteps[-init_timestep]
        latent_timestep = timestep.repeat(encoded.shape[0])
        noise = self._frequency_noise.sample(
            encoded,
            walk,
            settings.experiments.frequency,
            walk.generator,
        )
        return pipeline.scheduler.add_noise(encoded, noise, latent_timestep)

    def _prepare_ip_adapter(
        self,
        walk: LatentWalk,
        settings: ExperimentSettings,
    ) -> tuple[dict[str, object], float]:
        adapter = settings.ip_adapter
        pipeline = self._pipeline
        if not adapter.enabled:
            if self._ip_adapter_loaded:
                pipeline.unload_ip_adapter()
                self._ip_adapter_loaded = False
            return {}, 0.0

        if not self._ip_adapter_loaded:
            pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                image_encoder_folder="models/image_encoder",
            )
            pipeline.image_encoder.to(device="cuda", dtype=torch.float16)
            pipeline.image_encoder.requires_grad_(False)
            self._ip_adapter_loaded = True

        effective_weight = self._effective_ip_weight(walk, adapter)
        pipeline.set_ip_adapter_scale(effective_weight)
        if adapter.memory == "ema":
            embeds = pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image=walk.image,
                ip_adapter_image_embeds=None,
                device=pipeline._execution_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )[0]
            if walk.ip_adapter_ema is None:
                walk.ip_adapter_ema = embeds
            else:
                walk.ip_adapter_ema = (
                    adapter.decay * walk.ip_adapter_ema
                    + (1 - adapter.decay) * embeds
                )
            return {"ip_adapter_image_embeds": [walk.ip_adapter_ema]}, effective_weight

        history = list(walk.history)
        source = walk.image
        if history and adapter.memory == "lagged":
            source = history[max(0, len(history) - adapter.lag - 1)]
        elif history and adapter.memory == "random":
            index = torch.randint(
                len(history),
                (1,),
                generator=walk.conditioning_generator,
                device="cuda",
            ).item()
            source = history[index]
        return {"ip_adapter_image": source}, effective_weight

    def _effective_ip_weight(
        self,
        walk: LatentWalk,
        adapter: IpAdapterSettings,
    ) -> float:
        if adapter.modulation == "decay":
            return adapter.weight * math.exp(
                -adapter.modulation_rate * walk.step_number
            )
        if adapter.modulation == "pulse":
            phase = walk.step_number % adapter.pulse_period
            active = phase < adapter.pulse_period * adapter.pulse_duty
            return adapter.weight if active else 0.0
        if adapter.modulation == "feedback" and walk.last_semantic_change is not None:
            ratio = walk.last_semantic_change / adapter.feedback_target
            return min(adapter.weight * min(max(ratio, 0.25), 2.0), 1.5)
        return adapter.weight

    def _load_clip_encoder(self) -> CLIPVisionModelWithProjection:
        if self._clip_encoder is None:
            encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to("cuda")
            encoder.eval().requires_grad_(False)
            self._clip_encoder = encoder
        return self._clip_encoder

    def _embed_image(self, image: Image.Image) -> torch.Tensor:
        encoder = self._load_clip_encoder()
        pixels = image_tensor(image)
        pixels = ClipGuidance._resize_clip_pixels(pixels).clamp(0, 1)
        mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073])
        std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711])
        pixels = (pixels - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        with torch.no_grad():
            output = encoder(pixel_values=pixels.to(torch.float16))
        return F.normalize(output.image_embeds.float(), dim=-1)

    def _prepare_clip_guidance(
        self,
        walk: LatentWalk,
        settings: ExperimentSettings,
    ) -> tuple[
        ClipGuidance | None,
        torch.Tensor | None,
        bool,
        StagnationResult,
    ]:
        if not self._tracks_semantics(settings):
            return None, None, False, StagnationResult()

        encoder = self._load_clip_encoder()
        if (
            walk.semantic_embedding is None
            or walk.semantic_step_number != walk.step_number
        ):
            current = self._embed_image(walk.image)
            walk.semantic_embedding = current
            walk.semantic_step_number = walk.step_number
            walk.semantic_direction = None
            walk.semantic_history.clear()
            walk.semantic_history.append(current.detach())
            walk.escape_cooldown = 0

        if settings.clip.enabled and walk.semantic_direction is None:
            current = walk.semantic_embedding
            direction = torch.randn(
                current.shape,
                generator=walk.semantic_generator,
                device=current.device,
            )
            direction -= (direction * current).sum(dim=-1, keepdim=True) * current
            walk.semantic_direction = F.normalize(direction, dim=-1)

        clip_target = (
            F.normalize(
                walk.semantic_embedding
                + settings.clip.semantic_step * walk.semantic_direction,
                dim=-1,
            )
            if settings.clip.enabled
            else None
        )
        stagnation = (
            semantic_stagnation(
                walk.semantic_history,
                settings.escape.sensitivity,
            )
            if settings.escape.enabled
            else StagnationResult()
        )
        escape_active = self._should_escape(walk, settings, stagnation)
        callback = (
            ClipGuidance(
                settings,
                self._pipeline.vae,
                encoder,
                clip_target,
            )
            if clip_target is not None
            else None
        )
        return callback, clip_target, escape_active, stagnation

    @staticmethod
    def _should_escape(
        walk: LatentWalk,
        settings: ExperimentSettings,
        stagnation: StagnationResult,
    ) -> bool:
        if not settings.escape.enabled:
            walk.escape_cooldown = 0
            return False

        if walk.escape_cooldown > 0:
            walk.escape_cooldown -= 1
        if not stagnation.stuck or walk.escape_cooldown > 0:
            return False

        walk.escape_cooldown = 24
        return True

    def _escape_step(
        self,
        pipeline: AutoPipelineForImage2Image,
        walk: LatentWalk,
        settings: WalkSettings,
        ip_kwargs: dict[str, object],
        effective_ip_weight: float,
        stagnation: StagnationResult,
    ) -> tuple[Image.Image, float]:
        candidate_strength = min(
            settings.noise_strength + settings.experiments.escape.strength,
            1.0,
        )
        candidate_steps = max(
            settings.denoise_steps,
            math.ceil(settings.denoise_steps / candidate_strength),
        )
        seeds = torch.randint(
            0,
            2**31,
            (4,),
            generator=walk.escape_generator,
            device="cuda",
        ).cpu()
        candidates = []
        adapter_active = bool(ip_kwargs)
        if adapter_active:
            pipeline.set_ip_adapter_scale(0.0)
        try:
            with torch.no_grad():
                for seed in seeds:
                    generator = torch.Generator(device="cuda").manual_seed(int(seed))
                    candidate = pipeline(
                        prompt="",
                        image=walk.image,
                        strength=candidate_strength,
                        num_inference_steps=candidate_steps,
                        guidance_scale=0.0,
                        generator=generator,
                        output_type="pil",
                        **ip_kwargs,
                    ).images[0]
                    candidates.append(candidate)
        finally:
            if adapter_active:
                pipeline.set_ip_adapter_scale(effective_ip_weight)

        centroid = stagnation.centroid
        similarities = [
            float((self._embed_image(candidate) * centroid).sum())
            for candidate in candidates
        ]
        selected = min(range(len(candidates)), key=similarities.__getitem__)
        return candidates[selected], similarities[selected]

    @staticmethod
    def _tracks_semantics(settings: ExperimentSettings) -> bool:
        return (
            settings.clip.enabled
            or settings.escape.enabled
            or settings.metrics.enabled
            or (
                settings.ip_adapter.enabled
                and settings.ip_adapter.modulation == "feedback"
            )
        )

    def _update_clip_state(
        self,
        walk: LatentWalk,
        settings: ExperimentSettings,
    ) -> float | None:
        if not self._tracks_semantics(settings):
            return None

        previous = walk.semantic_embedding
        current = self._embed_image(walk.image)
        semantic_change = float(1 - (previous * current).sum())
        if settings.clip.enabled:
            observed = current - previous
            observed -= (observed * current).sum(dim=-1, keepdim=True) * current
        else:
            observed = None
        if observed is not None and observed.norm() > 1e-6:
            observed = F.normalize(observed, dim=-1)
            innovation = torch.randn(
                current.shape,
                generator=walk.semantic_generator,
                device=current.device,
            )
            innovation -= (
                innovation * current
            ).sum(dim=-1, keepdim=True) * current
            innovation = F.normalize(innovation, dim=-1)
            direction = (
                settings.clip.momentum * walk.semantic_direction
                + (1 - settings.clip.momentum)
                * (0.8 * observed + 0.2 * innovation)
            )
            direction -= (direction * current).sum(dim=-1, keepdim=True) * current
            walk.semantic_direction = F.normalize(direction, dim=-1)
        walk.semantic_embedding = current
        walk.semantic_step_number = walk.step_number
        walk.semantic_history.append(current.detach())
        walk.last_semantic_change = semantic_change
        return semantic_change

    def decode_jpeg(self, image: Image.Image, quality: int = 88) -> bytes:
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()


model_service = ModelService()
