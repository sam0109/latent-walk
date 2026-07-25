from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class FrequencySettings:
    enabled: bool = False
    low: float = 1.0
    mid: float = 1.0
    high: float = 1.0
    persistence: float = 0.5


@dataclass(frozen=True)
class ClipSettings:
    enabled: bool = False
    semantic_step: float = 0.08
    momentum: float = 0.85
    guidance: float = 0.005


@dataclass(frozen=True)
class IpAdapterSettings:
    enabled: bool = False
    weight: float = 0.2
    memory: str = "previous"
    lag: int = 4
    decay: float = 0.85


@dataclass(frozen=True)
class ExperimentSettings:
    frequency: FrequencySettings = field(default_factory=FrequencySettings)
    clip: ClipSettings = field(default_factory=ClipSettings)
    ip_adapter: IpAdapterSettings = field(default_factory=IpAdapterSettings)

    @classmethod
    def from_message(cls, message: dict[str, object]) -> "ExperimentSettings":
        raw = message.get("experiments", {})
        if not isinstance(raw, dict):
            raise ValueError("experiments must be an object")

        frequency = _section(raw, "frequency")
        clip = _section(raw, "clip")
        ip_adapter = _section(raw, "ipAdapter")
        memory = ip_adapter.get("memory", "previous")
        if memory not in {"previous", "ema", "lagged", "random"}:
            raise ValueError("ipAdapter.memory is invalid")

        return cls(
            frequency=FrequencySettings(
                enabled=_boolean(frequency, "enabled", False),
                low=_number(frequency, "low", 1.0, 0.0, 2.0),
                mid=_number(frequency, "mid", 1.0, 0.0, 2.0),
                high=_number(frequency, "high", 1.0, 0.0, 2.0),
                persistence=_number(
                    frequency, "persistence", 0.5, 0.0, 0.98
                ),
            ),
            clip=ClipSettings(
                enabled=_boolean(clip, "enabled", False),
                semantic_step=_number(
                    clip, "semanticStep", 0.08, 0.005, 0.3
                ),
                momentum=_number(clip, "momentum", 0.85, 0.0, 0.98),
                guidance=_number(clip, "guidance", 0.005, 0.0, 0.025),
            ),
            ip_adapter=IpAdapterSettings(
                enabled=_boolean(ip_adapter, "enabled", False),
                weight=_number(ip_adapter, "weight", 0.2, 0.0, 1.5),
                memory=memory,
                lag=_integer(ip_adapter, "lag", 4, 1, 24),
                decay=_number(ip_adapter, "decay", 0.85, 0.0, 0.99),
            ),
        )


@dataclass
class WalkState:
    image: Image.Image
    seed: int
    step_number: int = 0
    generator: torch.Generator | None = None
    semantic_generator: torch.Generator | None = None
    conditioning_generator: torch.Generator | None = None
    history: deque[Image.Image] = field(
        default_factory=lambda: deque(maxlen=32)
    )
    frequency_memory: dict[str, torch.Tensor] = field(default_factory=dict)
    semantic_embedding: torch.Tensor | None = None
    semantic_direction: torch.Tensor | None = None
    ip_adapter_ema: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.image = self.image.copy()
        self.history.append(self.image.copy())

    def advance(self, image: Image.Image) -> None:
        self.image = image
        self.history.append(image.copy())
        self.step_number += 1


@dataclass(frozen=True)
class StepResult:
    image: Image.Image
    pixel_change: float
    semantic_change: float | None = None
    effective_parameters: dict[str, object] = field(default_factory=dict)


class NoiseStrategy(Protocol):
    def sample(
        self,
        latent: torch.Tensor,
        state: WalkState,
        settings: FrequencySettings,
        generator: torch.Generator,
    ) -> torch.Tensor: ...


class ConditioningStrategy(Protocol):
    def prepare(
        self, state: WalkState, settings: IpAdapterSettings
    ) -> dict[str, object]: ...


class GuidanceStrategy(Protocol):
    def prepare(
        self, state: WalkState, settings: ClipSettings
    ) -> tuple[object | None, dict[str, float]]: ...


class GaussianNoiseStrategy:
    def sample(
        self,
        latent: torch.Tensor,
        state: WalkState,
        settings: FrequencySettings,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.randn(
            latent.shape,
            generator=generator,
            device=latent.device,
            dtype=latent.dtype,
        )


class FrequencyNoiseStrategy:
    def sample(
        self,
        latent: torch.Tensor,
        state: WalkState,
        settings: FrequencySettings,
        generator: torch.Generator,
    ) -> torch.Tensor:
        raw = torch.randn(
            latent.shape,
            generator=generator,
            device=latent.device,
            dtype=torch.float32,
        )
        low = gaussian_blur(raw, sigma=6.0)
        mid_smooth = gaussian_blur(raw, sigma=1.5)
        bands = {
            "low": _normalize(low),
            "mid": _normalize(mid_smooth - low),
            "high": _normalize(raw - mid_smooth),
        }

        persistence = settings.persistence
        innovation = math.sqrt(1.0 - persistence**2)
        for name, band in bands.items():
            previous = state.frequency_memory.get(name)
            if previous is not None and previous.shape == band.shape:
                band = persistence * previous + innovation * band
                band = _normalize(band)
            state.frequency_memory[name] = band.detach()
            bands[name] = band

        combined = (
            settings.low * bands["low"]
            + settings.mid * bands["mid"]
            + settings.high * bands["high"]
        )
        if float(combined.square().mean()) < 1e-8:
            combined = raw
        return _normalize(combined).to(dtype=latent.dtype)


def gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(
        -radius, radius + 1, device=tensor.device, dtype=tensor.dtype
    )
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(tensor.shape[1], 1, -1, -1)
    padded = F.pad(tensor, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel, groups=tensor.shape[1])


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    rms = tensor.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    return tensor / rms.clamp_min(1e-6)


def _section(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"experiments.{key} must be an object")
    return value


def _number(
    parent: dict[str, object],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = parent.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return min(max(float(value), minimum), maximum)


def _integer(
    parent: dict[str, object],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = parent.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return min(max(value, minimum), maximum)


def _boolean(parent: dict[str, object], key: str, default: bool) -> bool:
    value = parent.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
