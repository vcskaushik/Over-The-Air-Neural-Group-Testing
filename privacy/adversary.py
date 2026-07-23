"""AdversaryHead: same architecture as the OTA-NGT receiver (layer3 + layer4 + avgpool + fc),
but with a configurable-width final classifier head. Used for both the in-loop privacy adversary
during Stage B and the from-scratch honest-evaluation adversary in Stage C.
"""
import torch
import torch.nn as nn

import resnet_design2 as models


class AdversaryHead(nn.Module):
    """Decoder-only twin of `ResNet_GT`. Consumes post-channel features.

    Args:
        arch_name: name of a model factory in `resnet_design2` (e.g. "resnet18", "resnext101_32x8d").
        num_classes: output dimension of the final fc head.
    """

    def __init__(self, arch_name: str, num_classes: int = 1000, pretrained: bool = False):
        super().__init__()
        if not hasattr(models, arch_name):
            raise ValueError(f"Unknown arch_name {arch_name!r}; not found in resnet_design2")
        backbone = getattr(models, arch_name)(pretrained=pretrained, gt=True, phase=False)
        # Borrow only the decoder slice. Drop encoder layers.
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        in_features = backbone.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)
        del backbone

    def forward(self, post_channel: torch.Tensor) -> torch.Tensor:
        x = self.layer3(post_channel)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
