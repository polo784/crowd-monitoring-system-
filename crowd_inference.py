"""Reusable CSRNet checkpoint inference, independent of datasets and training."""

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from csrnet import CSRNet


@dataclass
class DensityPrediction:
    density: np.ndarray
    count: float
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class CSRNetPredictor:
    """Load any compatible CSRNet checkpoint once and predict one BGR frame.

    Normalization comes from checkpoint metadata, or an explicit override.
    The complete trained weights are loaded; VGG weights are never downloaded.
    """

    def __init__(
        self, checkpoint: str | Path, device: str = "auto",
        normalization: str = "auto", input_size: tuple[int, int] | None = None,
    ) -> None:
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        if input_size is not None and any(n < 8 or n % 8 for n in input_size):
            raise ValueError("--input-size HEIGHT WIDTH must use positive multiples of 8")
        self.input_size = input_size
        self.checkpoint_path = Path(checkpoint).resolve()
        saved = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict):
            raise ValueError("Expected a checkpoint dictionary or a model state dictionary")
        config = saved.get("training_config", {})
        if config.get("model", "CSRNet") != "CSRNet":
            raise ValueError("This predictor requires a CSRNet checkpoint")
        if normalization not in ("auto", "imagenet", "none"):
            raise ValueError("normalization must be auto, imagenet, or none")
        if normalization == "auto":
            if not isinstance(config.get("pretrained_vgg16"), bool):
                raise ValueError(
                    "Checkpoint lacks normalization metadata. Specify --normalization "
                    "imagenet or none to match its training preprocessing."
                )
            normalization = "imagenet" if config["pretrained_vgg16"] else "none"
        self.normalization = normalization
        mean = std = self.mean = self.std = None
        if normalization == "imagenet":
            # Explicit overrides for older checkpoints retain the standard values.
            # Metadata-driven inference must not guess missing training statistics.
            mean = config.get("imagenet_mean")
            std = config.get("imagenet_std")
            if mean is None or std is None:
                raise ValueError("ImageNet normalization requires checkpoint imagenet_mean and imagenet_std")
            try:
                if len(mean) != 3 or len(std) != 3 or any(isinstance(n, bool) for n in [*mean, *std]):
                    raise ValueError("Expected three numeric mean/std values")
                self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
                self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("Invalid checkpoint normalization values") from error
            if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
                raise ValueError("Invalid checkpoint normalization values")
        self.model = CSRNet(pretrained=False)
        self.model.load_state_dict(saved.get("model_state_dict", saved), strict=True)
        self.model.to(self.device).eval()
        self.metadata = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_epoch": saved.get("epoch"),
            "checkpoint_validation_mae": saved.get("validation_mae"),
            "model": "CSRNet",
            "normalization": normalization,
            "mean": mean if normalization == "imagenet" else None,
            "std": std if normalization == "imagenet" else None,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "torch_version": str(torch.__version__),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> DensityPrediction:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("Expected an HWC uint8 BGR video frame")
        started = time.perf_counter()
        frame = frame_bgr
        if self.input_size is not None:
            height, width = self.input_size
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        height, width = frame.shape[:2]
        if height < 8 or width < 8 or height % 8 or width % 8:
            raise ValueError(
                f"Native frame {height}x{width} is not divisible by CSRNet's stride 8. "
                "Use --input-size HEIGHT WIDTH with multiples of 8. "
                "No automatic crop, padding, or resize is performed."
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32).div_(255.0)
        if self.normalization == "imagenet":
            tensor = (tensor - self.mean) / self.std
        self._sync()
        prepared = time.perf_counter()
        prediction = self.model(tensor)
        self._sync()
        inferred = time.perf_counter()
        if prediction.ndim != 4 or prediction.shape[:2] != (1, 1):
            raise RuntimeError(f"Expected one density channel, got {tuple(prediction.shape)}")
        density = prediction[0, 0].cpu().numpy().copy()
        if not np.isfinite(density).all():
            raise RuntimeError("Non-finite density prediction; stopping inference")
        # Raw prediction only: do not clamp, resize, or normalize before counting.
        count = float(density.sum(dtype=np.float64))
        ended = time.perf_counter()
        return DensityPrediction(
            density, count, tuple(tensor.shape), tuple(prediction.shape),
            (prepared - started) * 1000, (inferred - prepared) * 1000,
            (ended - inferred) * 1000,
        )
