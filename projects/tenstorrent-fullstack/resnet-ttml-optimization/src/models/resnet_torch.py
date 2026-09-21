from pathlib import Path
import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class TorchResNet18(ResNet):
    def __init__(self, num_classes=None):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes or 1000)
        if num_classes is None:
            self.fc = nn.Identity()


class TorchResNet50(ResNet):
    def __init__(self, num_classes=None):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes or 1000)
        if num_classes is None:
            self.fc = nn.Identity()


class TorchResNet101(ResNet):
    def __init__(self, num_classes=None):
        super().__init__(Bottleneck, [3, 4, 23, 3], num_classes=num_classes or 1000)
        if num_classes is None:
            self.fc = nn.Identity()


def load_reference_weights(model, depth, weights_path=None, *, pretrained=False):
    if weights_path is not None:
        state = torch.load(Path(weights_path).expanduser(), map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
    elif pretrained:
        from torchvision import models
        weights = getattr(models, f'ResNet{depth}_Weights').IMAGENET1K_V1
        state = weights.get_state_dict(progress=True, check_hash=True)
    else:
        return
    if isinstance(model.fc, nn.Identity):
        state = {key: value for key, value in state.items() if key not in {'fc.weight','fc.bias'}}
    model.load_state_dict(state, strict=True)
