"""Single-model local runtime, with generation guards against stale video results."""

import base64
import copy
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import threading
import time
import uuid
from PIL import Image
import io
import torch

import cv2
import numpy as np

from crowd_inference import CSRNetPredictor
from video_inference import file_sha256, normalize_density_display

LOG = logging.getLogger("crowd.dashboard")
ROOT = Path(__file__).resolve().parent


class RuntimeProblem(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


def load_config(path=ROOT / "dashboard_config.json"):
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        return validate_config(config)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Configuration {path}: {error}") from error


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    integers = {"history_limit": (1, 1200), "max_model_pixels": (64, 2073600),
                "max_video_bytes": (1, 2147483648), "max_frame_bytes": (1, 33554432)}
    numbers = {"max_sample_hz": (.1, 4), "overlay_max_age_seconds": (.1, 120),
               "observation_max_age_seconds": (1, 300), "health_timeout_seconds": (1, 60),
               "inference_timeout_seconds": (1, 300), "control_timeout_seconds": (1, 60),
               "upload_timeout_seconds": (10, 900)}
    for key, (minimum, maximum) in {**integers, **numbers}.items():
        value = config.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or (key in integers and not isinstance(value, int))
                or not minimum <= value <= maximum or not math.isfinite(value)):
            kind = "integer" if key in integers else "number"
            raise ValueError(f"{key} must be a finite {kind} in [{minimum}, {maximum}]")
    thresholds = config.get("attention_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("attention_thresholds must be an object")
    levels = [thresholds.get(k) for k in ("watch", "elevated", "review")]
    if (any(isinstance(n, bool) or not isinstance(n, (float, int)) or not 0 <= n <= 1000000
            or not math.isfinite(n) for n in levels) or not levels[0] < levels[1] < levels[2]):
        raise ValueError("Attention thresholds must be finite, nonnegative and increasing")
    if config.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if not isinstance(config.get("checkpoint"), str) or not config["checkpoint"].strip():
        raise ValueError("checkpoint must be a nonempty path")
    digest = config.get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 fingerprint")
    return config


def attention_level(count, thresholds):
    if not math.isfinite(count):
        raise ValueError("Count must be finite")
    if count < 0:
        return "UNAVAILABLE"
    for key in ("review", "elevated", "watch"):
        if count >= thresholds[key]:
            return key.upper()
    return "NORMAL"


def attention_reason(count, thresholds):
    level = attention_level(count, thresholds)
    if level == "UNAVAILABLE":
        return "Negative raw count retained; attention assessment unavailable"
    if level == "NORMAL":
        return f"Raw count < WATCH threshold {thresholds['watch']}"
    return f"Raw count ≥ {level} threshold {thresholds[level.lower()]}"


def compatible_size(height, width, max_pixels):
    """Preserve the whole scene; bound pixel budget, then round UP to stride 8.

    No crop/padding/count compensation. Very slight aspect change from rounding.
    """
    if height < 8 or width < 8 or height > 16384 or width > 16384:
        raise RuntimeProblem(422, "Video dimensions must be between 8 and 16384 pixels")
    scale = min(1.0, math.sqrt(max_pixels / (height * width)))
    h = max(8, math.ceil(height * scale / 8) * 8)
    w = max(8, math.ceil(width * scale / 8) * 8)
    # Rounding may exceed the cap by one stride: reduce along the larger axis.
    while h * w > max_pixels:
        if h >= w and h > 8:
            h -= 8
        elif w > 8:
            w -= 8
        else:
            h -= 8
    return h, w


def heatmap_png(density):
    """Existing red-positive/blue-negative display rule, transparent at zero."""
    display = normalize_density_display(density)
    bgra = np.zeros((*display.shape, 4), dtype=np.uint8)
    bgra[:, :, 2] = np.where(display >= 0, 255, 0)
    bgra[:, :, 0] = np.where(display < 0, 255, 0)
    bgra[:, :, 3] = np.rint(np.abs(display) * 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", bgra)
    if not ok:
        raise RuntimeError("Cannot encode density visualization")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


@dataclass
class VideoSession:
    id: str
    path: Path
    filename: str
    metadata: dict
    scratch: object = None
    generation: int = 0
    playing: bool = False
    timestamp: float = 0
    latest: dict | None = None
    latest_density: np.ndarray | None = None
    latest_frame: np.ndarray | None = None
    history: deque = field(default_factory=deque)
    completions: deque = field(default_factory=lambda: deque(maxlen=20))
    segment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    observations_total: int = 0


class DashboardRuntime:
    def __init__(self, config, predictor_factory=CSRNetPredictor):
        self.config, self.predictor_factory = validate_config(config), predictor_factory
        self.lock, self.inference_lock = threading.RLock(), threading.Lock()
        self.predictor = None
        self.model_state, self.error = "loading", None
        self.session = None

    def load_model(self):
        try:
            checkpoint = (ROOT / self.config["checkpoint"]).resolve()
            digest = file_sha256(checkpoint)
            if digest != self.config["checkpoint_sha256"]:
                raise ValueError("Checkpoint SHA-256 mismatch; use the trusted epoch-16 artifact")
            predictor = self.predictor_factory(checkpoint, device=self.config["device"])
            if predictor.metadata.get("checkpoint_epoch") != 16:
                raise ValueError("This milestone requires the preserved epoch-16 checkpoint")
            predictor.metadata["checkpoint_sha256"] = digest
            with self.lock:
                self.predictor, self.model_state = predictor, "ready"
            LOG.info("CSRNet epoch 16 loaded once | device=%s | normalization=%s",
                     predictor.metadata["device_name"], predictor.metadata["normalization"])
        except Exception as error:
            with self.lock:
                self.model_state, self.error = "error", str(error)
            LOG.exception("Model could not load")

    def select_video(self, path, filename, metadata, scratch=None):
        with self.lock:
            old = self.session
            self.session = VideoSession(uuid.uuid4().hex, Path(path), filename, metadata, scratch,
                                        history=deque(maxlen=self.config["history_limit"]))
            result = self.session_info()
        if old and old.scratch:
            old.scratch.cleanup()
        LOG.info("Video opened: %s | %.2f FPS | %sx%s | model H,W=%s",
                 filename, metadata["fps"], metadata["width"], metadata["height"], metadata["model_size_hw"])
        return result

    def session_info(self):
        s = self.session
        return None if not s else {"id": s.id, "filename": s.filename, "generation": s.generation,
                                  "playing": s.playing, "timestamp": s.timestamp,
                                  "video": s.metadata, "segment_id": s.segment_id,
                                  "observations_total": s.observations_total,
                                  "latest": s.latest, "history": list(s.history)}

    def status(self):
        with self.lock:
            return {"model_state": self.model_state, "error": self.error,
                    "model": self.predictor.metadata if self.predictor else None,
                    "config": {k: v for k, v in self.config.items() if k not in {"checkpoint", "checkpoint_sha256"}},
                    "session": None if not self.session else {"id": self.session.id,
                                "generation": self.session.generation, "segment_id": self.session.segment_id},
                    "inference_busy": self.inference_lock.locked()}

    def export_history(self):
        with self.lock:
            if not self.session:
                raise RuntimeProblem(409, "No video selected")
            s = self.session
            observations = copy.deepcopy(list(s.history))
            metadata = self.export_metadata()
            metadata.update(observations_total=s.observations_total,
                            discarded_observations=max(0, s.observations_total - len(observations)),
                            retained_range_seconds=[observations[0]["timestamp"], observations[-1]["timestamp"]] if observations else None,
                            history_status="historical accepted backend observations; browser latency is unavailable here")
            return {"metadata": metadata, "observations": observations}

    def require_session(self, session_id, generation=None):
        s = self.session
        if not s or s.id != session_id:
            raise RuntimeProblem(409, "This video session was replaced; choose the video again")
        if generation is not None and generation != s.generation:
            raise RuntimeProblem(409, "Stale frame discarded after a playback change")
        return s

    def control(self, session_id, generation, action, timestamp, playing):
        with self.lock:
            s = self.require_session(session_id)
            if generation <= s.generation:
                raise RuntimeProblem(409, "Old playback command discarded")
            if action not in {"play", "pause", "seek", "restart", "ended", "invalidate"}:
                raise RuntimeProblem(422, "Unknown playback command")
            self.check_timestamp(s, timestamp)
            s.generation, s.playing, s.timestamp = generation, playing, timestamp
            s.completions.clear()
            s.latest = s.latest_density = s.latest_frame = None
            if action in {"seek", "restart"}:
                s.history.clear()
                s.segment_id = uuid.uuid4().hex
                s.observations_total = 0
            result = {"generation": s.generation, "playing": s.playing,
                      "segment_id": s.segment_id, "history_length": len(s.history)}
        LOG.info("Playback %s at %.3fs | generation=%s", action, timestamp, generation)
        return result

    @staticmethod
    def check_timestamp(session, timestamp):
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp > session.metadata["duration"] + .5:
            raise RuntimeProblem(422, "Invalid video timestamp")

    def infer(self, session_id, generation, timestamp, image_bytes, snapshot=False, capture_time=None):
        # Never queue unbounded old video frames or run concurrent GPU predictions.
        if not self.inference_lock.acquire(blocking=False):
            raise RuntimeProblem(429, "Model is finishing the previous frame; retry the current frame")
        started = time.perf_counter()
        try:
            with self.lock:
                s = self.require_session(session_id, generation)
                if self.model_state != "ready":
                    raise RuntimeProblem(503, self.error or "Model is still loading")
                if not s.playing and not snapshot:
                    raise RuntimeProblem(409, "Video is paused; pending inference discarded")
                self.check_timestamp(s, timestamp)
                if s.latest and timestamp <= s.latest["timestamp"]:
                    raise RuntimeProblem(409, "Repeated or out-of-order sample discarded")
                metadata = s.metadata.copy()
            if len(image_bytes) > self.config["max_frame_bytes"]:
                raise RuntimeProblem(413, "Sampled image is too large")
            # Read JPEG header before allocating the decoded image. This is a local
            # resource guard, not an internet-ready media sandbox.
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.format != "JPEG" or image.size != (metadata["width"], metadata["height"]):
                        raise ValueError("dimensions/format")
            except Exception as error:
                raise RuntimeProblem(422, "Sampled frame is invalid or has unexpected dimensions") from error
            frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (metadata["height"], metadata["width"]):
                raise RuntimeProblem(422, "Sampled frame is invalid or has unexpected dimensions")
            h, w = metadata["model_size_hw"]
            prepared = frame if frame.shape[:2] == (h, w) else cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            try:
                prediction = self.predictor.predict(prepared)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise RuntimeProblem(503, "CUDA out of memory. Reduce max_model_pixels in dashboard_config.json and restart. No automatic resize retry was performed.") from error
            # Predictor counts the raw density in float64 before visualization.
            if not math.isfinite(prediction.count) or not np.isfinite(prediction.density).all():
                raise RuntimeProblem(503, "Model produced non-finite values; inference stopped")
            png = heatmap_png(prediction.density)
            completed = time.perf_counter()
            with self.lock:
                s = self.require_session(session_id, generation)  # Reject work invalidated while CUDA ran.
                s.completions.append(completed)
                rate = ((len(s.completions) - 1) / (s.completions[-1] - s.completions[0])) if len(s.completions) > 1 else None
                observation = {"timestamp": timestamp, "count": prediction.count,
                               "capture_time_utc": capture_time,
                               "observation_time_utc": datetime.now(timezone.utc).isoformat(),
                               "segment_id": s.segment_id}
                result = {"session_id": session_id, "generation": generation, **observation,
                          "attention": attention_level(prediction.count, self.config["attention_thresholds"]),
                          "attention_reason": attention_reason(prediction.count, self.config["attention_thresholds"]),
                          "preprocess_ms": prediction.preprocess_ms, "postprocess_ms": prediction.postprocess_ms,
                          "inference_ms": prediction.inference_ms,
                          "model_fps": 1000 / max(prediction.inference_ms, 1e-9),
                          "sample_fps": rate, "service_ms": (completed - started) * 1000,
                          "input_shape": prediction.input_shape, "output_shape": prediction.output_shape,
                          "heatmap": png}
                s.latest, s.latest_density, s.latest_frame = result, prediction.density, frame
                s.timestamp = timestamp
                s.history.append({k: v for k, v in result.items() if k != "heatmap"})
                s.observations_total += 1
            return result
        finally:
            self.inference_lock.release()

    def export_metadata(self):
        with self.lock:
            s = self.session
            return {"model": copy.deepcopy(self.predictor.metadata) if self.predictor else None,
                    "configuration": copy.deepcopy(self.config),
                    "video_filename": s.filename if s else None,
                    "video": copy.deepcopy(s.metadata) if s else None,
                    "session_id": s.id if s else None, "segment_id": s.segment_id if s else None,
                    "history_capacity": self.config["history_limit"],
                    "count_definition": "float64 sum of raw signed density; no clipping or calibration",
                    "sampling": "browser decoded playback frame encoded as JPEG quality 0.92",
                    "reset_rule": "seek/restart/video change resets segment; pause retains history",
                    "timing_definition": "Browser capture-to-result uses performance.now and includes JPEG, upload, service, download; service excludes HTTP transfer",
                    "limitations": ["Prototype attention thresholds; not validated safety limits.",
                                     "Heatmap shows predicted image-space density, not people/m²."]}

    def close(self):
        with self.lock:
            s, self.session = self.session, None
        if s and s.scratch:
            s.scratch.cleanup()
