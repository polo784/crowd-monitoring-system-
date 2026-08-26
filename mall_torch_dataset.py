"""Minimal PyTorch Dataset for paired Mall images and density maps."""

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "cv_project" / "mall_dataset"


class MallDataset(Dataset):
    def __init__(self, dataset_dir: Path = DEFAULT_DATASET_DIR) -> None:
        images_dir = Path(dataset_dir) / "frames" / "frames"
        density_dir = Path(dataset_dir) / "density_maps"

        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(images_dir.glob("seq_*.jpg")):
            frame_number = int(image_path.stem.removeprefix("seq_"))
            density_path = density_dir / f"frame_{frame_number:04d}.h5"
            if not density_path.exists():
                raise FileNotFoundError(f"Missing density map for {image_path.name}")
            self.samples.append((image_path, density_path))

        if not self.samples:
            raise FileNotFoundError(f"No Mall JPG frames found in {images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, density_path = self.samples[index]

        with Image.open(image_path) as opened_image:
            image_array = np.array(opened_image.convert("RGB"), copy=True)

        with h5py.File(density_path, "r") as h5_file:
            density_array = h5_file["density"][:]

        image_tensor = (
            torch.from_numpy(image_array).permute(2, 0, 1).to(torch.float32) / 255.0
        )
        density_tensor = torch.from_numpy(density_array).to(torch.float32).unsqueeze(0)
        return image_tensor, density_tensor


if __name__ == "__main__":
    dataset = MallDataset()
    image_tensor, density_tensor = dataset[0]

    print(f"Dataset length:       {len(dataset)}")
    print(f"Image tensor shape:   {tuple(image_tensor.shape)}")
    print(f"Image dtype:          {image_tensor.dtype}")
    print(f"Image min/max:        {image_tensor.min().item():.6f} / {image_tensor.max().item():.6f}")
    print(f"Density tensor shape: {tuple(density_tensor.shape)}")
    print(f"Density dtype:        {density_tensor.dtype}")
    print(f"Density sum:          {density_tensor.sum().item():.6f}")
