"""Validate Mall Dataset images, annotations, and generated density maps.

This script does not modify the dataset. It reports structural/numerical problems
and writes a few visual overlays to the Git-ignored outputs directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "cv_project" / "mall_dataset"
IMAGES_DIR = DATASET_DIR / "frames" / "frames"
DENSITY_DIR = DATASET_DIR / "density_maps"
ANNOTATIONS_PATH = DATASET_DIR / "mall_gt.mat"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"
EXPECTED_IMAGE_SIZE = (640, 480)  # Pillow uses (width, height).


def image_path(frame_number: int) -> Path:
    return IMAGES_DIR / f"seq_{frame_number:06d}.jpg"


def density_path(frame_number: int) -> Path:
    return DENSITY_DIR / f"frame_{frame_number:04d}.h5"


def load_annotations() -> list[np.ndarray]:
    """Return one Nx2 array of (x, y) point annotations per frame."""
    mat_data = sio.loadmat(ANNOTATIONS_PATH)
    if "frame" not in mat_data:
        raise ValueError(f"{ANNOTATIONS_PATH} does not contain a 'frame' field")

    annotations: list[np.ndarray] = []
    for frame in mat_data["frame"][0]:
        points = np.asarray(frame["loc"][0][0], dtype=np.float32)
        annotations.append(points.reshape(-1, 2))
    return annotations


def save_overlay(
    frame_number: int,
    points: np.ndarray,
    density: np.ndarray,
    output_dir: Path,
) -> Path:
    """Overlay density in red and annotation points in cyan on one frame."""
    base = Image.open(image_path(frame_number)).convert("RGBA")

    maximum = float(density.max())
    normalized = density / maximum if maximum > 0 else np.zeros_like(density)
    strength = np.sqrt(np.clip(normalized, 0.0, 1.0))

    heat = np.zeros((*density.shape, 4), dtype=np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = (100 * (1.0 - strength)).astype(np.uint8)
    heat[..., 3] = (190 * strength).astype(np.uint8)
    overlay = Image.fromarray(heat, mode="RGBA")
    combined = Image.alpha_composite(base, overlay)

    draw = ImageDraw.Draw(combined)
    for x, y in points:
        radius = 4
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(box, outline=(0, 255, 255, 255), width=2)

    label = f"Frame {frame_number} | annotations={len(points)} | density sum={density.sum():.2f}"
    draw.rectangle((8, 8, 8 + 8 * len(label), 30), fill=(0, 0, 0, 190))
    draw.text((12, 11), label, fill=(255, 255, 255, 255))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_number:04d}_overlay.png"
    combined.convert("RGB").save(output_path)
    return output_path


def validate_dataset(
    overlay_frames: list[int],
    output_dir: Path,
    count_tolerance: float,
) -> int:
    required_paths = [ANNOTATIONS_PATH, IMAGES_DIR, DENSITY_DIR]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"ERROR: Required path does not exist: {path}")
        return 1

    annotations = load_annotations()
    total_frames = len(annotations)
    expected_images = {image_path(i) for i in range(1, total_frames + 1)}
    expected_maps = {density_path(i) for i in range(1, total_frames + 1)}
    actual_images = set(IMAGES_DIR.glob("seq_*.jpg"))
    actual_maps = set(DENSITY_DIR.glob("frame_*.h5"))

    errors: list[str] = []
    for path in sorted(expected_images - actual_images):
        errors.append(f"Missing image: {path.name}")
    for path in sorted(expected_maps - actual_maps):
        errors.append(f"Missing density map: {path.name}")
    for path in sorted(actual_images - expected_images):
        errors.append(f"Unexpected image: {path.name}")
    for path in sorted(actual_maps - expected_maps):
        errors.append(f"Unexpected density map: {path.name}")

    annotation_counts: list[int] = []
    density_counts: list[float] = []
    count_errors: list[float] = []
    overlays = set(overlay_frames)

    print(f"Validating {total_frames} annotated frames...")
    for frame_number, points in enumerate(annotations, start=1):
        current_image = image_path(frame_number)
        current_map = density_path(frame_number)
        if not current_image.exists() or not current_map.exists():
            continue

        with Image.open(current_image) as image:
            if image.size != EXPECTED_IMAGE_SIZE:
                errors.append(
                    f"Frame {frame_number}: image size {image.size}, expected {EXPECTED_IMAGE_SIZE}"
                )

        if not np.isfinite(points).all():
            errors.append(f"Frame {frame_number}: annotations contain NaN or infinity")
        elif len(points):
            x, y = points[:, 0], points[:, 1]
            in_bounds = (x >= 0) & (x < 640) & (y >= 0) & (y < 480)
            if not in_bounds.all():
                errors.append(
                    f"Frame {frame_number}: {int((~in_bounds).sum())} annotation(s) outside the image"
                )

        with h5py.File(current_map, "r") as h5_file:
            if "density" not in h5_file:
                errors.append(f"Frame {frame_number}: HDF5 file has no 'density' dataset")
                continue
            density = h5_file["density"][:]

        if density.shape != (480, 640):
            errors.append(
                f"Frame {frame_number}: density shape {density.shape}, expected (480, 640)"
            )
            continue
        if not np.isfinite(density).all():
            errors.append(f"Frame {frame_number}: density contains NaN or infinity")
            continue
        if np.any(density < 0):
            errors.append(f"Frame {frame_number}: density contains negative values")

        annotation_count = len(points)
        density_count = float(density.sum(dtype=np.float64))
        difference = abs(density_count - annotation_count)
        annotation_counts.append(annotation_count)
        density_counts.append(density_count)
        count_errors.append(difference)

        if difference > count_tolerance:
            errors.append(
                f"Frame {frame_number}: density sum {density_count:.3f} differs from "
                f"annotation count {annotation_count} by {difference:.3f}"
            )

        if frame_number in overlays:
            saved_path = save_overlay(frame_number, points, density, output_dir)
            print(f"Saved overlay: {saved_path.relative_to(PROJECT_ROOT)}")

        if frame_number % 250 == 0 or frame_number == total_frames:
            print(f"Checked {frame_number}/{total_frames} frames")

    print("\nDataset summary")
    print(f"  Annotation frames: {total_frames}")
    print(f"  Image files:       {len(actual_images)}")
    print(f"  Density maps:      {len(actual_maps)}")
    if annotation_counts:
        print(
            "  People per frame:  "
            f"min={min(annotation_counts)}, "
            f"mean={np.mean(annotation_counts):.2f}, "
            f"max={max(annotation_counts)}"
        )
        print(f"  Total annotations: {sum(annotation_counts)}")
        print(f"  Largest count gap: {max(count_errors):.4f}")

    if errors:
        print(f"\nValidation found {len(errors)} problem(s). First 20:")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 1

    print("\nValidation passed: images, annotations, and density maps are consistent.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-frames",
        type=int,
        nargs="*",
        default=[1, 800, 2000],
        help="One-based frame numbers for which overlays should be generated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated overlays.",
    )
    parser.add_argument(
        "--count-tolerance",
        type=float,
        default=0.05,
        help="Maximum allowed difference between density sum and annotation count.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        validate_dataset(
            overlay_frames=arguments.overlay_frames,
            output_dir=arguments.output_dir,
            count_tolerance=arguments.count_tolerance,
        )
    )
