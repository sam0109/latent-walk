import io

import av
import pytest
from PIL import Image

from latent_walk.model import (
    InvalidImageError,
    LatentWalk,
    WalkSettings,
    encode_mp4,
    prepare_image,
)


def image_bytes(width: int = 80, height: int = 40) -> bytes:
    image = Image.new("RGB", (width, height), (130, 70, 210))
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_prepare_image_crops_and_normalizes() -> None:
    image = prepare_image(image_bytes(), 512)
    assert image.size == (512, 512)
    assert image.mode == "RGB"


def test_prepare_image_rejects_invalid_input() -> None:
    with pytest.raises(InvalidImageError, match="readable image"):
        prepare_image(b"not an image", 512)


def test_walk_advances_without_origin_constraint() -> None:
    walk = LatentWalk(Image.new("RGB", (8, 8), "black"))
    destination = Image.new("RGB", (8, 8), "white")

    change = walk.advance(destination)

    assert walk.image is destination
    assert change == 1
    assert walk.step_number == 1


def test_settings_are_bounded() -> None:
    settings = WalkSettings.from_message(
        {"noiseStrength": 9, "denoiseSteps": 100}
    )
    assert settings == WalkSettings(noise_strength=0.8, denoise_steps=4)


def test_mp4_uses_requested_frame_rate() -> None:
    frames = []
    for color in ("red", "green", "blue"):
        output = io.BytesIO()
        Image.new("RGB", (512, 512), color).save(output, "JPEG")
        frames.append(output.getvalue())

    video = encode_mp4(frames, fps=2)

    with av.open(io.BytesIO(video)) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
        duration = float(stream.duration * stream.time_base)
    assert len(decoded) == 3
    assert float(stream.average_rate) == 2
    assert duration == pytest.approx(1.5, abs=0.05)
