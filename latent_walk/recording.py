from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import deque
from dataclasses import dataclass

from .experiments import StepResult
from .model import WalkSettings

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class RecordedFrame:
    step: int
    jpeg: bytes


class ExperimentRecorder:
    def __init__(
        self,
        seed: int,
        source_jpeg: bytes,
        size: int,
        max_frames: int,
    ) -> None:
        self.seed = seed
        self.source_jpeg = source_jpeg
        self.size = size
        self.frames: deque[RecordedFrame] = deque(
            [RecordedFrame(0, source_jpeg)],
            maxlen=max_frames,
        )
        self.steps: deque[dict[str, object]] = deque(maxlen=max_frames)

    def record(
        self,
        settings: WalkSettings,
        result: StepResult,
        jpeg: bytes,
        step: int,
    ) -> None:
        self.frames.append(RecordedFrame(step, jpeg))
        self.steps.append(
            {
                "step": step,
                "settings": settings.to_message(),
                "metrics": result.metrics,
                "effective": result.effective_parameters,
            }
        )

    def manifest(self) -> dict[str, object]:
        steps = list(self.steps)
        first_step = steps[0]["step"] if steps else None
        return {
            "version": MANIFEST_VERSION,
            "model": "stabilityai/sdxl-turbo",
            "seed": self.seed,
            "size": self.size,
            "sourceSha256": hashlib.sha256(self.source_jpeg).hexdigest(),
            "retainedFrameSteps": [frame.step for frame in self.frames],
            "firstStep": first_step,
            "truncated": first_step not in (None, 1),
            "steps": steps,
        }

    def manifest_bytes(self) -> bytes:
        return json.dumps(
            self.manifest(),
            indent=2,
            sort_keys=True,
        ).encode()

    def bundle_bytes(self) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr("manifest.json", self.manifest_bytes())
            archive.writestr("source.jpg", self.source_jpeg)
            for frame in self.frames:
                archive.writestr(
                    f"frames/{frame.step:06d}.jpg",
                    frame.jpeg,
                )
        return output.getvalue()
