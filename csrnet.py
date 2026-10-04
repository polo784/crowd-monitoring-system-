"""CSRNet: VGG16 conv1_1 through conv4_3, then a dilated density backend.

Architecture reference: https://github.com/leeyeehoo/CSRNet-pytorch
The classification head, pool4, and pool5 are NOT part of this model.
"""

import torch
from torch import nn
from torchvision.models import VGG16_Weights


def make_layers(channels: list[int | str], in_channels: int, dilation: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    for out_channels in channels:
        if out_channels == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend([
                nn.Conv2d(in_channels, int(out_channels), 3, padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True),
            ])
            in_channels = int(out_channels)
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.frontend = make_layers(
            [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512], 3
        )
        self.backend = make_layers([512, 512, 512, 256, 128, 64], 512, dilation=2)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.pretrained_frontend = pretrained

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

        if pretrained:
            # Load only matching frontend parameters. No classifier is instantiated.
            weights = VGG16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            frontend_weights = {
                name: weights[f"features.{name}"] for name in self.frontend.state_dict()
            }
            self.frontend.load_state_dict(frontend_weights, strict=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Standard CSRNet has a linear output, not a final ReLU or classifier.
        return self.output_layer(self.backend(self.frontend(images)))
