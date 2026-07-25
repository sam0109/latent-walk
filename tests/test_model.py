import io

import pytest
from PIL import Image

from latent_walk.model import InvalidImageError, LatentWalk, WalkSettings, prepare_image


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
