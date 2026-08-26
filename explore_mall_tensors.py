"""Load one Mall Dataset frame and inspect it as PyTorch tensors."""

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "cv_project" / "mall_dataset"
IMAGE_PATH = DATASET_DIR / "frames" / "frames" / "seq_000001.jpg"
DENSITY_PATH = DATASET_DIR / "density_maps" / "frame_0001.h5"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "exploration" / "frame_0001_image_and_density.png"


# Load the image as RGB, then copy its pixel values into a NumPy array.
with Image.open(IMAGE_PATH) as opened_image:
    image = opened_image.convert("RGB")
    image_array = np.array(image, dtype=np.uint8, copy=True)

# Load the matching two-dimensional density map from the HDF5 file.
with h5py.File(DENSITY_PATH, "r") as h5_file:
    density_map = h5_file["density"][:]

# Convert HWC to CHW, change uint8 to float32, and scale 0-255 to 0.0-1.0.
image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(torch.float32) / 255.0
density_tensor = torch.from_numpy(density_map)

print(f"Original image shape: {image_array.shape}")
print(f"Image tensor shape:   {tuple(image_tensor.shape)}")
print(f"Image tensor dtype:   {image_tensor.dtype}")
print(f"Image tensor minimum: {image_tensor.min().item():.6f}")
print(f"Image tensor maximum: {image_tensor.max().item():.6f}")
print(f"Density-map shape:    {density_map.shape}")
print(f"Density tensor shape: {tuple(density_tensor.shape)}")
print(f"Density tensor dtype: {density_tensor.dtype}")
print(f"Density-map sum:      {density_map.sum(dtype=np.float64):.6f}")

# Make a visible grayscale COPY of the density map. The real map is unchanged.
display_max = float(density_map.max())
if display_max > 0:
    density_for_display = (density_map / display_max * 255).astype(np.uint8)
else:
    density_for_display = np.zeros_like(density_map, dtype=np.uint8)

density_image = Image.fromarray(density_for_display, mode="L").convert("RGB")
side_by_side = Image.new("RGB", (image.width * 2, image.height))
side_by_side.paste(image, (0, 0))
side_by_side.paste(density_image, (image.width, 0))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
side_by_side.save(OUTPUT_PATH)
print(f"Saved side-by-side preview: {OUTPUT_PATH}")
