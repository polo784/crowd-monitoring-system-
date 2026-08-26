"""A very small baseline CNN for Mall crowd-density estimation."""

import torch
from torch import nn


class CrowdDensityCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


if __name__ == "__main__":
    from mall_dataloader import train_loader

    image_batch, density_batch = next(iter(train_loader))
    model = CrowdDensityCNN()

    with torch.no_grad():
        predicted_density_maps = model(image_batch)

    assert predicted_density_maps.shape == density_batch.shape
    print(f"Input batch shape:            {tuple(image_batch.shape)}")
    print(f"Target density-map shape:    {tuple(density_batch.shape)}")
    print(f"Predicted density-map shape: {tuple(predicted_density_maps.shape)}")
