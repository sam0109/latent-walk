from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, Sequence

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
class EscapeSettings:
    enabled: bool = False
    strength: float = 0.02
    sensitivity: float = 1.2


@dataclass(frozen=True)
class IpAdapterSettings:
    enabled: bool = False
    weight: float = 0.2
    memory: str = "previous"
    lag: int = 4
    decay: float = 0.85
    modulation: str = "constant"
    modulation_rate: float = 0.08
    pulse_period: int = 8
    pulse_duty: float = 0.5
    feedback_target: float = 0.08


@dataclass(frozen=True)
class MetricSettings:
    enabled: bool = False


@dataclass(frozen=True)
class ExperimentSettings:
    frequency: FrequencySettings = field(default_factory=FrequencySettings)
    clip: ClipSettings = field(default_factory=ClipSettings)
    escape: EscapeSettings = field(default_factory=EscapeSettings)
    ip_adapter: IpAdapterSettings = field(default_factory=IpAdapterSettings)
    metrics: MetricSettings = field(default_factory=MetricSettings)

    @classmethod
    def from_message(cls, message: dict[str, object]) -> "ExperimentSettings":
        raw = message.get("experiments", {})
        if not isinstance(raw, dict):
            raise ValueError("experiments must be an object")

        frequency = _section(raw, "frequency")
        clip = _section(raw, "clip")
        escape = _section(raw, "escape")
        ip_adapter = _section(raw, "ipAdapter")
        metrics = _section(raw, "metrics")
        memory = ip_adapter.get("memory", "previous")
        if memory not in {"previous", "ema", "lagged", "random"}:
            raise ValueError("ipAdapter.memory is invalid")
        modulation = ip_adapter.get("modulation", "constant")
        if modulation not in {"constant", "decay", "pulse", "feedback"}:
            raise ValueError("ipAdapter.modulation is invalid")

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
            escape=EscapeSettings(
                enabled=_boolean(escape, "enabled", False),
                strength=_number(escape, "strength", 0.02, 0.0, 0.025),
                sensitivity=_number(escape, "sensitivity", 1.2, 0.5, 1.5),
            ),
            ip_adapter=IpAdapterSettings(
                enabled=_boolean(ip_adapter, "enabled", False),
                weight=_number(ip_adapter, "weight", 0.2, 0.0, 1.5),
                memory=memory,
                lag=_integer(ip_adapter, "lag", 4, 1, 24),
                decay=_number(ip_adapter, "decay", 0.85, 0.0, 0.99),
                modulation=modulation,
                modulation_rate=_number(
                    ip_adapter, "modulationRate", 0.08, 0.0, 1.0
                ),
                pulse_period=_integer(
                    ip_adapter, "pulsePeriod", 8, 2, 64
                ),
                pulse_duty=_number(
                    ip_adapter, "pulseDuty", 0.5, 0.05, 1.0
                ),
                feedback_target=_number(
                    ip_adapter, "feedbackTarget", 0.08, 0.001, 0.5
                ),
            ),
            metrics=MetricSettings(
                enabled=_boolean(metrics, "enabled", False),
            ),
        )

    def to_message(self) -> dict[str, object]:
        return {
            "frequency": {
                "enabled": self.frequency.enabled,
                "low": self.frequency.low,
                "mid": self.frequency.mid,
                "high": self.frequency.high,
                "persistence": self.frequency.persistence,
            },
            "clip": {
                "enabled": self.clip.enabled,
                "semanticStep": self.clip.semantic_step,
                "momentum": self.clip.momentum,
                "guidance": self.clip.guidance,
            },
            "escape": {
                "enabled": self.escape.enabled,
                "strength": self.escape.strength,
                "sensitivity": self.escape.sensitivity,
            },
            "ipAdapter": {
                "enabled": self.ip_adapter.enabled,
                "weight": self.ip_adapter.weight,
                "memory": self.ip_adapter.memory,
                "lag": self.ip_adapter.lag,
                "decay": self.ip_adapter.decay,
                "modulation": self.ip_adapter.modulation,
                "modulationRate": self.ip_adapter.modulation_rate,
                "pulsePeriod": self.ip_adapter.pulse_period,
                "pulseDuty": self.ip_adapter.pulse_duty,
                "feedbackTarget": self.ip_adapter.feedback_target,
            },
            "metrics": {
                "enabled": self.metrics.enabled,
            },
        }


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
    semantic_step_number: int | None = None
    semantic_direction: torch.Tensor | None = None
    semantic_history: deque[torch.Tensor] = field(
        default_factory=lambda: deque(maxlen=64)
    )
    ip_adapter_ema: torch.Tensor | None = None
    last_semantic_change: float | None = None

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
    metrics: dict[str, float] = field(default_factory=dict)
    effective_parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StagnationResult:
    stuck: bool = False
    pressure: float = 0.0
    radius: float | None = None
    progress_ratio: float | None = None
    revisit_similarity: float | None = None
    centroid: torch.Tensor | None = None


def semantic_stagnation(
    embeddings: Sequence[torch.Tensor],
    sensitivity: float,
    window_size: int = 12,
) -> StagnationResult:
    if len(embeddings) < window_size * 2:
        return StagnationResult()

    history = torch.cat(list(embeddings), dim=0).float()
    window = history[-window_size:]
    centroid = F.normalize(window.mean(dim=0, keepdim=True), dim=-1)
    radius = torch.acos(
        (window * centroid).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    ).mean()
    step_angles = torch.acos(
        (window[:-1] * window[1:]).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    )
    net_angle = torch.acos(
        (window[0] * window[-1]).sum().clamp(-1 + 1e-6, 1 - 1e-6)
    )
    progress_ratio = net_angle / step_angles.sum().clamp_min(1e-6)
    older = history[:-window_size]
    revisit_similarity = (older * window[-1]).sum(dim=-1).max()

    radius_value = float(radius)
    progress_value = float(progress_ratio)
    revisit_value = float(revisit_similarity)
    radius_pressure = min(
        max((0.38 * sensitivity - radius_value) / (0.24 * sensitivity), 0.0),
        1.0,
    )
    progress_pressure = min(
        max(
            (0.22 * sensitivity - progress_value) / (0.14 * sensitivity),
            0.0,
        ),
        1.0,
    )
    revisit_pressure = min(
        max(
            (revisit_value - (1 - 0.18 * sensitivity)) / (0.14 * sensitivity),
            0.0,
        ),
        1.0,
    )
    pressure = (radius_pressure * progress_pressure * revisit_pressure) ** (1 / 3)
    stuck = (
        radius_value < 0.22 * sensitivity
        and progress_value < 0.13 * sensitivity
        and revisit_value > 1 - 0.06 * sensitivity
    )
    return StagnationResult(
        stuck=stuck,
        pressure=pressure,
        radius=radius_value,
        progress_ratio=progress_value,
        revisit_similarity=revisit_value,
        centroid=centroid,
    )


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
            "low": low,
            "mid": mid_smooth - low,
            "high": raw - mid_smooth,
        }

        persistence = settings.persistence
        innovation = math.sqrt(1.0 - persistence**2)
        for name, band in bands.items():
            previous = state.frequency_memory.get(name)
            if previous is not None and previous.shape == band.shape:
                band = persistence * previous + innovation * band
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
