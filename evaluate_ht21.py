"""Compare saved video counts with CroHD/HT21 human head annotations.

This is an external-scene counting diagnostic, not the official tracking benchmark.
Reads gt/gt.txt, never detector-generated det/det.txt. No training or inference.
"""

import argparse
import configparser
import csv
import io
import json
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np

from video_inference import file_sha256


def load_ground_truth(archive_path, sequence):
    if sequence not in {f"HT21-{i:02d}" for i in range(1, 5)}:
        raise ValueError("Use a public annotated HT21 training sequence (01-04)")
    base = f"HT21Labels/train/{sequence}/"
    with zipfile.ZipFile(archive_path) as archive:
        config = configparser.ConfigParser()
        config.read_string(archive.read(base + "seqinfo.ini").decode())
        metadata = dict(config["Sequence"])
        rows = np.loadtxt(io.StringIO(archive.read(base + "gt/gt.txt").decode()), delimiter=",", ndmin=2)
    validate_ground_truth(rows, int(metadata["seqlength"]))
    return rows, metadata


def validate_ground_truth(rows, length):
    if rows.ndim != 2 or rows.shape[1] != 9 or not np.isfinite(rows).all():
        raise ValueError("Expected finite nine-column MOT head ground truth")
    for column in (0, 1, 6, 7, 8):
        if not np.equal(rows[:, column], np.floor(rows[:, column])).all():
            raise ValueError("Frame, ID, confidence, class and visibility must be integers")
    if not np.isin(rows[:, 7], [1, 2, 3, 4]).all() or not np.isin(rows[:, 8], [0, 1]).all():
        raise ValueError("Unknown CroHD class or visibility")
    if not np.isin(rows[:, 6], [0, 1]).all() or (rows[:, 4:6] <= 0).any():
        raise ValueError("Invalid annotation flag or head box")
    if len(np.unique(rows[:, :2], axis=0)) != len(rows):
        raise ValueError("Duplicate frame/person annotation")
    if set(rows[:, 0].astype(int)) != set(range(1, length + 1)):
        raise ValueError("Ground truth does not cover every expected one-based frame")


def human_masks(rows):
    # CroHD's classes differ from MOT20: 1 pedestrian, 2 static, 3 ignore,
    # 4 person on vehicle. Include stationary people; exclude statues/reflections.
    people = (rows[:, 6] == 1) & np.isin(rows[:, 7], [1, 2, 4])
    return people & (rows[:, 8] == 1), people


def count_metrics(predictions, targets):
    difference = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    return {"mae": float(np.abs(difference).mean()),
            "rmse": float(np.sqrt(np.square(difference).mean())),
            "mean_signed_error": float(difference.mean()),
            "ground_truth_mean": float(np.mean(targets)),
            "ground_truth_min": int(np.min(targets)), "ground_truth_max": int(np.max(targets)),
            "predicted_mean": float(np.mean(predictions)),
            "predicted_min": float(np.min(predictions)), "predicted_max": float(np.max(predictions))}


def evaluate(labels, sequence, run_dir, output_dir):
    gt, metadata = load_ground_truth(labels, sequence)
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary["status"] != "complete" or summary["scope"] != "full_video":
        raise ValueError("Require a successful full-video inference run with no warnings")
    with (run_dir / "measurements.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    length, fps = int(metadata["seqlength"]), float(metadata["framerate"])
    if len(rows) != length or summary["processed_frames"] != length:
        raise ValueError("Video prediction length does not match annotated sequence")
    if [int(r["frame_index"]) for r in rows] != list(range(length)):
        raise ValueError("Predictions must cover every zero-based frame exactly once, in order")
    if not math.isclose(summary["input_video"]["nominal_fps"], fps):
        raise ValueError("Video and annotation FPS differ")
    timestamps = [float(r["timestamp_seconds"]) for r in rows]
    if not np.allclose(timestamps, np.arange(length) / fps, atol=0.002, rtol=0):
        raise ValueError("Video timestamps differ from one-to-one annotation frame timing")
    predictions = np.array([float(r["predicted_count"]) for r in rows])
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite prediction")
    visible, all_people = human_masks(gt)
    visible_counts = np.bincount(gt[visible, 0].astype(int), minlength=length + 1)[1:]
    all_counts = np.bincount(gt[all_people, 0].astype(int), minlength=length + 1)[1:]
    comparison_rows = [dict(frame_index=i, annotation_frame=i + 1, timestamp_seconds=timestamps[i],
                            visible_head_count=int(visible_counts[i]),
                            annotated_people_including_occluded=int(all_counts[i]),
                            predicted_count=float(predictions[i]),
                            signed_error_vs_visible=float(predictions[i] - visible_counts[i]),
                            absolute_error_vs_visible=float(abs(predictions[i] - visible_counts[i])))
                       for i in range(length)]
    report = {
        "status": "complete", "sequence": sequence, "evaluated_frames": length,
        "primary_count_definition": "Visible human heads: valid flag=1, class in (1 pedestrian, 2 static, 4 person on vehicle), visibility=1; exclude class 3 ignore",
        "primary_metrics_visible_heads": count_metrics(predictions, visible_counts),
        "secondary_metrics_including_occluded_tracks": count_metrics(predictions, all_counts),
        "annotation_statistics": {"rows": len(gt), "visible_human_rows": int(visible.sum()),
                                  "all_human_rows": int(all_people.sum()),
                                  "ignored_rows": int((gt[:, 7] == 3).sum()),
                                  "classes": {str(int(c)): int((gt[:, 7] == c).sum()) for c in np.unique(gt[:, 7])}},
        "alignment": {"video_index_0_matches_annotation_frame": 1, "fps": fps,
                      "annotation_resolution_wh": [int(metadata["imwidth"]), int(metadata["imheight"])],
                      "video_resolution_wh": [summary["input_video"]["width"], summary["input_video"]["height"]],
                      "sample_frames_zero_based": [0, length // 2, length - 1]},
        "inference": summary, "labels_sha256": file_sha256(labels),
        "video_sha256": file_sha256(Path(summary["input_video"]["path"])),
        "sources": ["https://motchallenge.net/data/Head_Tracking_21/", "https://arxiv.org/html/2103.13516v1"],
        "limitations": [
            "External evaluation on a public CroHD training clip; our checkpoint was trained on Mall, not CroHD. Not an official CroHD test/leaderboard result.",
            "Downloaded raw preview MP4 is half-resolution, compressed and has a sequence-name watermark. It is not the original image sequence.",
            "Model input is explicitly resized only for stride compatibility; no scale was selected based on resulting accuracy.",
            "Primary count excludes fully occluded tracks; the secondary count includes them. This is a counting adaptation of a head-tracking dataset.",
            "Adjacent video frames are strongly correlated; one short clip cannot establish deployment reliability.",
            "Frame numbers, lengths and timing are checked; sampled overlays provide a visual alignment check, not pixel-exact proof against the original images."
        ]}
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    source = cv2.VideoCapture(summary["input_video"]["path"])
    annotated = cv2.VideoCapture(str(run_dir / "annotated.mp4"))
    width, height = summary["input_video"]["width"], summary["input_video"]["height"]
    writer = cv2.VideoWriter(str(output_dir / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not source.isOpened() or not annotated.isOpened() or not writer.isOpened():
        source.release(); annotated.release(); writer.release()
        raise RuntimeError("Cannot open comparison video input/output")
    sx, sy = width / int(metadata["imwidth"]), height / int(metadata["imheight"])
    try:
        for i in range(length):
            ok, original = source.read()
            ok2, overlay = annotated.read()
            if not ok or not ok2:
                raise RuntimeError(f"Video ended before annotated frame {i + 1}")
            label = f"Visible GT: {visible_counts[i]} | Model: {predictions[i]:.2f} | Error: {predictions[i]-visible_counts[i]:+.2f}"
            cv2.rectangle(overlay, (0, height - 38), (width, height), (0, 0, 0), -1)
            cv2.putText(overlay, label, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(overlay)
            if i in report["alignment"]["sample_frames_zero_based"]:
                for box in gt[(gt[:, 0] == i + 1) & visible]:
                    x, y, w, h = box[2:6]
                    cv2.rectangle(original, (round(x*sx), round(y*sy)), (round((x+w)*sx), round((y+h)*sy)), (0, 255, 0), 1)
                # Left: human head boxes; right: model heatmap and signed count error.
                preview = np.concatenate([original, overlay], axis=1)
                if not cv2.imwrite(str(output_dir / f"frame_{i:06d}_comparison.jpg"), preview):
                    raise RuntimeError("Could not save sampled comparison")
        if source.read()[0] or annotated.read()[0]:
            raise RuntimeError("Video contains extra frames beyond annotation length")
    finally:
        source.release(); annotated.release(); writer.release()
    verify = cv2.VideoCapture(str(output_dir / "comparison.mp4"))
    decoded = 0
    try:
        while True:
            ok, frame = verify.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Comparison video dimensions differ")
            decoded += 1
    finally:
        verify.release()
    if decoded != length:
        raise RuntimeError("Comparison video frame count differs")
    report["comparison_video_verified_frames"] = decoded
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequence", default="HT21-01")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.labels, args.sequence, args.run_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
