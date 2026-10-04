"""Package only CSRNet source files and existing Mall data for a Colab upload.

No preprocessing is run. JPG and HDF5 file bytes are copied unchanged.
The archive lives under the Git-ignored outputs directory, not in the repository.
"""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "csrnet.py", "density_utils.py", "train_csrnet.py",
    "mall_torch_dataset.py", "mall_dataloader.py", "test_csrnet.py",
)
FRAMES_DIR = Path("cv_project/mall_dataset/frames/frames")
DENSITY_DIR = Path("cv_project/mall_dataset/density_maps")


def bundle_files() -> list[Path]:
    frames = [FRAMES_DIR / f"seq_{number:06d}.jpg" for number in range(1, 2001)]
    densities = [DENSITY_DIR / f"frame_{number:04d}.h5" for number in range(1, 2001)]
    if {path.name for path in (PROJECT_ROOT / FRAMES_DIR).glob("*.jpg")} != {
        path.name for path in frames
    }:
        raise RuntimeError("Expected exactly Mall JPG frames 1-2000")
    if {path.name for path in (PROJECT_ROOT / DENSITY_DIR).glob("*.h5")} != {
        path.name for path in densities
    }:
        raise RuntimeError("Expected exactly Mall density maps 1-2000")
    paths = [Path(name) for name in SOURCE_FILES] + frames + densities
    for relative in paths:
        if not (PROJECT_ROOT / relative).is_file():
            raise FileNotFoundError(PROJECT_ROOT / relative)
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_notebook_sources() -> None:
    """Embed current Python sources without rebuilding or uploading the dataset ZIP."""
    notebook_path = PROJECT_ROOT / "CSRNet_Mall_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "source-snapshot")
    sources = {name: (PROJECT_ROOT / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    # JSON object/string syntax is valid Python for this string-only dictionary.
    cell["source"] = ("# Current source snapshot; refresh after editing the Python scripts.\n"
                      + "NOTEBOOK_SOURCES = " + json.dumps(sources, indent=2)
                      + "\n").splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Refreshed {len(sources)} source files in {notebook_path}")


def build_bundle(output: Path) -> None:
    paths = bundle_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    # Exclusive creation protects an earlier upload bundle from accidental overwrite.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for index, relative in enumerate(paths, 1):
            digest = hashlib.sha256()
            size = 0
            with (PROJECT_ROOT / relative).open("rb") as source:
                with archive.open(relative.as_posix(), "w", force_zip64=True) as destination:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        destination.write(chunk)
                        size += len(chunk)
            records[relative.as_posix()] = {"bytes": size, "sha256": digest.hexdigest()}
            if index % 500 == 0 or index == len(paths):
                print(f"Packed {index}/{len(paths)} files", flush=True)

        manifest = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "frame_count": 2000,
            "density_map_count": 2000,
            "batch_size": 4,
            "splits": {"train": [1, 1400], "validation": [1401, 1700], "test": [1701, 2000]},
            "source_files": list(SOURCE_FILES),
            "files": records,
        }
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))

    checksum = sha256_file(output)
    print(f"Bundle: {output}")
    print(f"Size: {output.stat().st_size / (1024 ** 2):.1f} MiB")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-notebook", action="store_true",
                        help="Update embedded code only; do not rebuild the large data ZIP")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs" / "colab" / "mall_csrnet_bundle.zip")
    args = parser.parse_args()
    if args.refresh_notebook:
        refresh_notebook_sources()
    else:
        build_bundle(args.output.resolve())
