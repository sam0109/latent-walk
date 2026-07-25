from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def image_tensor(image: Image.Image, device: str = "cuda") -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        / 255
    )


class MetricSuite:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._lpips = None

    @property
    def loaded(self) -> bool:
        return self._lpips is not None

    def compute(
        self,
        previous: Image.Image,
        current: Image.Image,
    ) -> dict[str, float]:
        before = image_tensor(previous, self.device)
        after = image_tensor(current, self.device)
        metrics = {
            "pixelMultiscale": self._multiscale_l1(before, after),
            "edgeChange": self._edge_change(before, after),
            "lpips": self._lpips_distance(before, after),
        }
        return metrics

    def _multiscale_l1(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> float:
        distances = []
        for scale in (1, 2, 4, 8):
            if scale == 1:
                before_scale, after_scale = before, after
            else:
                before_scale = F.avg_pool2d(before, scale)
                after_scale = F.avg_pool2d(after, scale)
            distances.append((before_scale - after_scale).abs().mean())
        return float(torch.stack(distances).mean())

    def _edge_change(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> float:
        kernel_x = before.new_tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ).view(1, 1, 3, 3)
        kernel_y = kernel_x.transpose(-1, -2)
        before_gray = before.mean(dim=1, keepdim=True)
        after_gray = after.mean(dim=1, keepdim=True)
        before_edges = torch.sqrt(
            F.conv2d(before_gray, kernel_x, padding=1).square()
            + F.conv2d(before_gray, kernel_y, padding=1).square()
            + 1e-8
        )
        after_edges = torch.sqrt(
            F.conv2d(after_gray, kernel_x, padding=1).square()
            + F.conv2d(after_gray, kernel_y, padding=1).square()
            + 1e-8
        )
        return float((before_edges - after_edges).abs().mean())

    def _lpips_distance(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> float:
        if self._lpips is None:
            import lpips

            self._lpips = lpips.LPIPS(net="alex").eval().to(self.device)
            self._lpips.requires_grad_(False)
        with torch.no_grad():
            distance = self._lpips(before * 2 - 1, after * 2 - 1)
        return float(distance.mean())
