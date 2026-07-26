import io
import json
import zipfile

from PIL import Image

from latent_walk.experiments import StepResult
from latent_walk.model import WalkSettings
from latent_walk.recording import ExperimentRecorder


def jpeg(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, "JPEG")
    return output.getvalue()


def test_recorder_exports_replayable_manifest_and_bundle() -> None:
    source = jpeg("black")
    frame = jpeg("white")
    recorder = ExperimentRecorder(42, source, 512, max_frames=10)
    settings = WalkSettings()
    result = StepResult(
        Image.new("RGB", (32, 32), "white"),
        pixel_change=1.0,
        metrics={"lpips": 0.8},
    )

    recorder.record(settings, result, frame, step=1)
    manifest = json.loads(recorder.manifest_bytes())

    assert manifest["version"] == 2
    assert manifest["seed"] == 42
    assert manifest["firstStep"] == 1
    assert manifest["truncated"] is False
    assert manifest["steps"][0]["settings"] == settings.to_message()
    with zipfile.ZipFile(io.BytesIO(recorder.bundle_bytes())) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "source.jpg",
            "frames/000000.jpg",
            "frames/000001.jpg",
        }


def test_recorder_marks_truncated_manifest_as_not_replayable() -> None:
    source = jpeg("black")
    recorder = ExperimentRecorder(42, source, 512, max_frames=2)
    settings = WalkSettings()

    for step in range(1, 4):
        result = StepResult(Image.new("RGB", (32, 32), "white"), pixel_change=1.0)
        recorder.record(settings, result, jpeg("white"), step=step)

    manifest = recorder.manifest()
    assert manifest["firstStep"] == 2
    assert manifest["truncated"] is True
    assert [record["step"] for record in manifest["steps"]] == [2, 3]
