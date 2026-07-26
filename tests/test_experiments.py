import math

import torch
from PIL import Image

from latent_walk.experiments import (
    ExperimentSettings,
    FrequencyNoiseStrategy,
    FrequencySettings,
    WalkState,
    semantic_stagnation,
)
from latent_walk.metrics import MetricSuite


def state() -> WalkState:
    return WalkState(Image.new("RGB", (512, 512), "black"), seed=7)


def test_experiment_settings_parse_and_bound_values() -> None:
    settings = ExperimentSettings.from_message(
        {
            "experiments": {
                "frequency": {
                    "enabled": True,
                    "low": 9,
                    "mid": 0.4,
                    "high": -2,
                    "persistence": 2,
                },
                "clip": {"enabled": True, "semanticStep": 0.12},
                "escape": {
                    "enabled": True,
                    "strength": 2,
                    "sensitivity": 3,
                },
                "ipAdapter": {
                    "enabled": True,
                    "weight": 0.7,
                    "memory": "lagged",
                    "lag": 8,
                    "decay": 0.7,
                },
            }
        }
    )

    assert settings.frequency.enabled
    assert settings.frequency.low == 2
    assert settings.frequency.high == 0
    assert settings.frequency.persistence == 0.98
    assert settings.clip.semantic_step == 0.12
    assert settings.escape.enabled
    assert settings.escape.strength == 0.025
    assert settings.escape.sensitivity == 1.5
    assert settings.ip_adapter.memory == "lagged"
    assert settings.ip_adapter.lag == 8
    assert settings.ip_adapter.decay == 0.7


def test_experiment_defaults_use_calibrated_safe_values() -> None:
    settings = ExperimentSettings.from_message({})

    assert settings.clip.guidance == 0.005
    assert settings.escape.strength == 0.02
    assert settings.escape.sensitivity == 1.2
    assert settings.ip_adapter.weight == 0.2


def test_experiment_settings_round_trip_through_protocol() -> None:
    original = ExperimentSettings.from_message(
        {
            "experiments": {
                "ipAdapter": {
                    "enabled": True,
                    "modulation": "pulse",
                    "pulsePeriod": 12,
                    "pulseDuty": 0.25,
                },
                "metrics": {"enabled": True},
            }
        }
    )

    restored = ExperimentSettings.from_message(
        {"experiments": original.to_message()}
    )

    assert restored == original


def test_multiscale_and_edge_metrics_detect_visual_change(monkeypatch) -> None:
    suite = MetricSuite("cpu")
    monkeypatch.setattr(suite, "_lpips_distance", lambda before, after: 0.5)
    black = Image.new("RGB", (64, 64), "black")
    split = Image.new("RGB", (64, 64), "black")
    for x in range(32, 64):
        for y in range(64):
            split.putpixel((x, y), (255, 255, 255))

    metrics = suite.compute(black, split)

    assert metrics["lpips"] == 0.5
    assert metrics["pixelMultiscale"] > 0.45
    assert metrics["edgeChange"] > 0


def test_frequency_noise_preserves_unit_variance() -> None:
    strategy = FrequencyNoiseStrategy()
    latent = torch.zeros(1, 4, 64, 64)
    generator = torch.Generator().manual_seed(5)

    noise = strategy.sample(
        latent,
        state(),
        FrequencySettings(enabled=True, low=0.3, mid=1.2, high=0.7),
        generator,
    )

    assert noise.square().mean().sqrt() == torch.tensor(1.0)


def test_equal_frequency_weights_preserve_white_noise_spectrum() -> None:
    strategy = FrequencyNoiseStrategy()
    latent = torch.zeros(1, 4, 64, 64)
    expected = torch.randn(
        latent.shape,
        generator=torch.Generator().manual_seed(7),
    )
    expected /= expected.square().mean().sqrt()

    noise = strategy.sample(
        latent,
        state(),
        FrequencySettings(persistence=0),
        torch.Generator().manual_seed(7),
    )

    torch.testing.assert_close(noise, expected, atol=2e-7, rtol=2e-7)


def test_low_frequency_noise_is_spatially_smoother() -> None:
    strategy = FrequencyNoiseStrategy()
    latent = torch.zeros(1, 4, 64, 64)
    low = strategy.sample(
        latent,
        state(),
        FrequencySettings(enabled=True, low=1, mid=0, high=0, persistence=0),
        torch.Generator().manual_seed(9),
    )
    high = strategy.sample(
        latent,
        state(),
        FrequencySettings(enabled=True, low=0, mid=0, high=1, persistence=0),
        torch.Generator().manual_seed(9),
    )

    low_gradient = (low[..., 1:] - low[..., :-1]).abs().mean()
    high_gradient = (high[..., 1:] - high[..., :-1]).abs().mean()
    assert low_gradient < high_gradient * 0.1


def test_frequency_persistence_correlates_steps() -> None:
    strategy = FrequencyNoiseStrategy()
    latent = torch.zeros(1, 4, 64, 64)
    walk = state()
    settings = FrequencySettings(enabled=True, persistence=0.95)
    generator = torch.Generator().manual_seed(3)

    first = strategy.sample(latent, walk, settings, generator)
    second = strategy.sample(latent, walk, settings, generator)
    correlation = torch.corrcoef(
        torch.stack((first.flatten(), second.flatten()))
    )[0, 1]

    assert correlation > 0.85


def test_semantic_stagnation_distinguishes_circling_from_progress() -> None:
    circling = []
    progressing = []
    for index in range(24):
        circle_angle = 0.03 * math.sin(index * math.pi / 2)
        progress_angle = index * 2 / 23
        circling.append(
            torch.tensor([[math.cos(circle_angle), math.sin(circle_angle), 0.0]])
        )
        progressing.append(
            torch.tensor([[math.cos(progress_angle), math.sin(progress_angle), 0.0]])
        )

    stuck = semantic_stagnation(circling, sensitivity=1.0)
    moving = semantic_stagnation(progressing, sensitivity=1.0)

    assert stuck.stuck
    assert stuck.radius < 0.22
    assert stuck.progress_ratio < 0.13
    assert stuck.revisit_similarity > 0.94
    assert stuck.pressure > 0.8
    assert not moving.stuck
    assert moving.pressure == 0
