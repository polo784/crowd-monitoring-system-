"""Run crowd-density inference on a recorded video; save MP4, CSV, and JSON.

Example: python video_inference.py --video clip.mp4 --checkpoint best.pt \
    --output-dir outputs/video_run
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from crowd_inference import CSRNetPredictor


def normalize_density_display(density: np.ndarray, heatmap_max: float | None = None) -> np.ndarray:
    """Signed relative density for DISPLAY ONLY; never use this to count people."""
    scale = heatmap_max if heatmap_max is not None else float(np.percentile(np.abs(density), 99))
    scale = max(scale, np.finfo(np.float32).tiny)
    return np.clip(density / scale, -1, 1)


def draw_overlay(frame: np.ndarray, density: np.ndarray, count: float,
                 timestamp: float, alpha: float, heatmap_max: float | None) -> np.ndarray:
    """Signed relative heatmap; its scaling never changes the predicted count."""
    normalized = normalize_density_display(density, heatmap_max)
    display = cv2.resize(normalized, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = np.zeros_like(frame, dtype=np.float32)
    color[:, :, 2] = np.where(display >= 0, 255, 0)  # Positive = red (BGR).
    color[:, :, 0] = np.where(display < 0, 255, 0)   # Negative = blue.
    opacity = (np.abs(display) * alpha)[:, :, None]
    result = np.clip(frame * (1 - opacity) + color * opacity, 0, 255).astype(np.uint8)
    label = "relative/frame" if heatmap_max is None else "fixed scale"
    lines = [f"Estimated count: {count:.2f} | Video: {timestamp:.3f}s",
             f"Density: red + / blue - | {label}"]
    base_width = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0] for text in lines)
    font_scale = max(0.1, min(0.65, (frame.shape[1] - 16) / base_width))
    sizes = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0] for text in lines]
    line_height = max(size[1] for size in sizes) + 9
    panel_width = min(frame.shape[1], max(size[0] for size in sizes) + 16)
    panel_height = min(frame.shape[0], line_height * len(lines) + 6)
    result[:panel_height, :panel_width] = (result[:panel_height, :panel_width] * 0.2).astype(np.uint8)
    for i, text in enumerate(lines):
        position = (8, (i + 1) * line_height)
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_video(predictor, video: Path, output_dir: Path, *, alpha: float = 0.5,
              heatmap_max: float | None = None, max_frames: int | None = None) -> dict:
    """Video infrastructure accepts any predictor implementing predict(frame).

    Every decoded frame is processed once. Output uses the source's nominal FPS.
    Timings include the first (cold) forward pass; there is no hidden warmup.
    """
    video, output_dir = Path(video).resolve(), Path(output_dir).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    if not 0 <= alpha <= 1 or (heatmap_max is not None and (not math.isfinite(heatmap_max) or heatmap_max <= 0)):
        raise ValueError("alpha must be in [0,1]; heatmap-max must be finite and positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be positive")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV cannot open video: {video}")
    writer = None
    created = False
    summary = {"status": "running", "model": predictor.metadata, "warnings": []}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable nominal FPS; cannot preserve playback timing")
        output_dir.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing run.
        created = True
        partial_path = output_dir / "annotated.partial.mp4"
        video_path = output_dir / "annotated.mp4"
        csv_path = output_dir / "measurements.csv"
        fields = ["frame_index", "timestamp_seconds", "timestamp_source", "predicted_count",
                  "decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
                  "overlay_encode_ms", "frame_processing_ms", "cumulative_processing_fps"]
        totals = {key: 0.0 for key in fields if key.endswith("_ms")}
        processed = 0
        first_timestamp = None
        previous_timestamp = -1.0
        fallback_count = 0
        input_shape = output_shape = None
        native_shape = None
        first_inference_ms = None
        started = time.perf_counter()
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            table = csv.DictWriter(stream, fieldnames=fields)
            table.writeheader()
            while max_frames is None or processed < max_frames:
                frame_started = time.perf_counter()
                ok, frame = cap.read()
                decoded = time.perf_counter()
                if not ok:
                    break
                if native_shape is None:
                    native_shape = tuple(frame.shape[:2])
                    height, width = native_shape
                    if height % 2 or width % 2:
                        raise ValueError("MP4 encoding requires even source dimensions; refusing silent cropping")
                    writer = cv2.VideoWriter(str(partial_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot initialize the mp4v video encoder")
                    summary["input_video"] = {
                        "path": str(video), "width": width, "height": height,
                        "nominal_fps": fps, "reported_frame_count": declared_frames,
                        "nominal_duration_seconds": declared_frames / fps if declared_frames > 0 else None,
                    }
                elif tuple(frame.shape[:2]) != native_shape:
                    raise RuntimeError("Frame resolution changed during the video")
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                timestamp_source = "decoder_pts"
                if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_timestamp:
                    timestamp = processed / fps
                    # Once falling back, keep timestamps monotonic even after variable PTS.
                    timestamp = max(timestamp, previous_timestamp + 1 / fps)
                    timestamp_source = "nominal_fps_fallback"
                    fallback_count += 1
                previous_timestamp = timestamp
                prediction = predictor.predict(frame)
                if not math.isfinite(prediction.count) or not np.isfinite(prediction.density).all():
                    raise RuntimeError("Predictor returned non-finite measurements")
                input_shape, output_shape = prediction.input_shape, prediction.output_shape
                if first_inference_ms is None:
                    first_inference_ms = prediction.inference_ms
                    print(f"Model input: {input_shape} -> output: {output_shape}", flush=True)
                rendered_start = time.perf_counter()
                annotated = draw_overlay(frame, prediction.density, prediction.count, timestamp, alpha, heatmap_max)
                writer.write(annotated)
                ended = time.perf_counter()
                processed += 1
                if first_timestamp is None:
                    first_timestamp = timestamp
                row = {
                    "frame_index": processed - 1, "timestamp_seconds": timestamp,
                    "timestamp_source": timestamp_source, "predicted_count": prediction.count,
                    "decode_ms": (decoded - frame_started) * 1000,
                    "preprocess_ms": prediction.preprocess_ms,
                    "inference_ms": prediction.inference_ms,
                    "postprocess_ms": prediction.postprocess_ms,
                    "overlay_encode_ms": (ended - rendered_start) * 1000,
                    "frame_processing_ms": (ended - frame_started) * 1000,
                    "cumulative_processing_fps": processed / (ended - started),
                }
                table.writerow(row)
                for key in totals:
                    totals[key] += row[key]
                if processed == 1 or processed % 50 == 0:
                    stream.flush()
                    print(f"Frame {processed}/{declared_frames or '?'} | count {prediction.count:.2f} | "
                          f"processing {row['cumulative_processing_fps']:.2f} FPS", flush=True)
        if processed == 0:
            raise RuntimeError("No decodable frames in the input video")
        writer.release()
        writer = None
        elapsed = time.perf_counter() - started
        check = cv2.VideoCapture(str(partial_path))
        try:
            readable, first_frame = check.read()
            encoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
            if not readable or tuple(first_frame.shape[:2]) != native_shape or encoded_frames != processed:
                raise RuntimeError("Encoded video failed frame-count, dimensions, or readability verification")
        finally:
            check.release()
        partial_path.rename(video_path)
        if fallback_count:
            summary["warnings"].append(f"{fallback_count} timestamps used nominal FPS because decoder PTS was unavailable/non-monotonic")
        limited = max_frames is not None and processed == max_frames
        if declared_frames > 0 and processed != declared_frames and not limited:
            summary["warnings"].append("Decoded frame count differs from metadata: input may be truncated or metadata inaccurate")
        summary.update({
            "status": "completed_with_warnings" if summary["warnings"] else "complete",
            "scope": "limited_run" if limited else "full_video",
            "processed_frames": processed,
            "model_input_shape": input_shape, "model_output_shape": output_shape,
            "frame_size_handling": "native" if predictor.input_size is None else f"explicit resize to H,W={predictor.input_size}",
            "count_method": "float64 sum of raw predicted density; no clamping or resizing",
            "heatmap": {"alpha": alpha, "scale": heatmap_max if heatmap_max else "per-frame 99th percentile of absolute density",
                        "meaning": "relative model density: red positive, blue negative; not people per square metre"},
            "timing": {
                "first_inference_ms": first_inference_ms,
                "average_inference_ms": totals["inference_ms"] / processed,
                "model_only_fps": processed * 1000 / max(totals["inference_ms"], 1e-9),
                "effective_inference_fps": processed / elapsed,
                "average_frame_processing_latency_ms": totals["frame_processing_ms"] / processed,
                "processing_seconds": elapsed,
                "average_stage_ms": {key: value / processed for key, value in totals.items()},
                "definition": "Cold first pass included. CUDA synchronized. Latency is decode through encode submission, not live capture-to-display. Effective FPS includes CSV and encoder finalization; excludes model loading and final verification.",
            },
            "timestamps": {"first_seconds": first_timestamp, "last_seconds": previous_timestamp, "fallback_frames": fallback_count},
            "output_video": str(video_path), "measurements": str(csv_path),
            "output_encoding": {"codec": "mp4v", "fps": fps, "audio": False,
                                "timing": "constant nominal source FPS; variable-frame-rate source timing is approximated"},
        })
        return summary
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if created:
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--normalization", choices=("auto", "imagenet", "none"), default="auto")
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="Explicit model resize, multiples of 8; default native resolution")
    parser.add_argument("--heatmap-alpha", type=float, default=0.5)
    parser.add_argument("--heatmap-max", type=float, help="Fixed raw density display scale; default relative per frame")
    parser.add_argument("--max-frames", type=int, help="Optional short verification run; default entire video")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Input video not found: {args.video}")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory")
    predictor = CSRNetPredictor(args.checkpoint, args.device, args.normalization,
                                tuple(args.input_size) if args.input_size else None)
    predictor.metadata["checkpoint_sha256"] = file_sha256(args.checkpoint)
    predictor.metadata["command"] = [sys.executable, *sys.argv]
    print(f"Device: {predictor.metadata['device_name']} | Checkpoint epoch: {predictor.metadata['checkpoint_epoch']} | "
          f"Normalization: {predictor.normalization}", flush=True)
    result = run_video(predictor, args.video, args.output_dir, alpha=args.heatmap_alpha,
                       heatmap_max=args.heatmap_max, max_frames=args.max_frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
