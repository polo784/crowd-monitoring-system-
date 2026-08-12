# Crowd Density Learning Project

This project explores computer vision and machine learning through a practical
crowd-analysis problem: estimating how people are distributed across a scene.
The longer-term goal is to adapt the same ideas to platform zones in systems
such as Kolkata Metro, where passengers could be directed toward less crowded
boarding areas.

The current dataset is the Mall Dataset: 2,000 frames from one fixed CCTV
camera, point annotations marking people, and density maps generated from those
points. It is a learning dataset, not yet a direct model of metro compartments.

## Current pipeline

```text
Mall CCTV frame
    -> annotated (x, y) person locations
    -> Gaussian density-map target
    -> future neural-network prediction
    -> total and regional crowd estimates
```

## Project files

- `preprocess.py` converts MATLAB point annotations into HDF5 density maps.
- `validate_dataset.py` checks image/annotation/map consistency and generates
  visual overlays.
- `pyproject.toml` declares the Python version and direct dependencies.
- `uv.lock` records exact dependency versions for reproducible environments.

The downloaded dataset and generated density maps are intentionally excluded
from Git. They remain on the local computer under `cv_project/mall_dataset/`.

## Environment

The project uses Python 3.11 and `uv` for dependency management.

In PowerShell, activate the existing environment and verify the main packages:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -c "import numpy, scipy, h5py, torch, torchvision; print('All imports successful')"
```

## Validate the dataset

Run the complete validation before model training:

```powershell
python validate_dataset.py
```

The validator checks that:

- every annotated frame has the expected image and density-map file;
- images are 640 x 480 pixels and density maps have shape `(480, 640)`;
- annotation coordinates stay inside image boundaries;
- density values are finite, non-negative, and sum approximately to the number
  of annotated people;
- selected density maps visually align with their source images.

By default, overlays for frames 1, 800, and 2000 are written to
`outputs/validation/`. Red regions represent density, while cyan circles mark
the original point annotations. This output directory is ignored by Git.

A validation failure is information, not a disaster. It means the data pipeline
needs correction before a model is asked to learn from it.

### First validation result

The first complete validation found all 2,000 images, annotation frames, and
density maps with the expected dimensions. Visual overlays for frames 1, 800,
and 2000 showed that annotation coordinates and density maps align with people.

However, 1,327 density maps failed the strict count-preservation check. Their
pixel sums were lower than their annotation counts by up to 1.4492 people. The
current Gaussian preprocessing uses constant padding, so density near an image
edge can spill outside the stored matrix and be lost. This must be corrected and
the maps regenerated before training.

## Current status

- [x] Acquire and extract the Mall Dataset.
- [x] Convert point annotations into density maps.
- [x] Protect local data and generated artifacts with `.gitignore`.
- [x] Verify the Python virtual environment and core imports.
- [x] Run and review full dataset validation.
- [ ] Correct any preprocessing issues revealed by validation.
- [ ] Create the first reviewed Git commit.
- [ ] Build a PyTorch `Dataset` and baseline model.

## Data provenance note

Before publishing or sharing the repository, record the exact Kaggle dataset
page, original research source, license, and download instructions here. That
allows another person to reproduce the project without committing the dataset
itself.
