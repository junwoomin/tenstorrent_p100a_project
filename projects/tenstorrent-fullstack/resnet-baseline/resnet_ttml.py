import numpy as np
import torch
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

from optimization_config import CLASSIFIER_DTYPE
from models.resnet_torch import TorchResNet18, TorchResNet50, TorchResNet101, load_reference_weights
from models.resnet_ttnn import TTNNResNet18, TTNNResNet50, TTNNResNet101
from compat.model_compare import compare_models
from compat.weight_mapper import load_torch_weights_into_ttnn


TORCH_MODELS = {18: TorchResNet18, 50: TorchResNet50, 101: TorchResNet101}
TTNN_MODELS = {18: TTNNResNet18, 50: TTNNResNet50, 101: TTNNResNet101}


class FrozenResNet:
    def __init__(self, device, depth=18, image_size=112, weights_path=None, *,
                 batch_size=None, cnn_defaults=None, pool_defaults=None, layer_overrides=None):
        if depth not in TTNN_MODELS:
            raise ValueError('depth must be 18, 50 or 101')
        self.device, self.depth, self.image_size = device, depth, image_size

        self.backbone = TTNN_MODELS[depth](device=device, image_size=image_size,
            batch_size=batch_size, cnn_defaults=cnn_defaults,
            pool_defaults=pool_defaults, layer_overrides=layer_overrides)
        model = TORCH_MODELS[depth]().eval().requires_grad_(False)
        load_reference_weights(model, depth, weights_path, pretrained=weights_path is None)
        report = compare_models(model, self.backbone)
        print(report)
        if not report.compatible:
            raise RuntimeError(str(report))
        load_torch_weights_into_ttnn(model, self.backbone)
        self.out_features = self.backbone.out_features

        self.state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}

    def __call__(self, images):
        features = self.backbone(images)
        return ttml.autograd.create_tensor(features, requires_grad=False)


class FrozenResNet18(FrozenResNet):


    def __init__(self, device, image_size=112, weights_path=None, **kwargs):
        super().__init__(device, 18, image_size, weights_path, **kwargs)


class ResNetClassifier(AbstractModuleBase):

    def __init__(self, in_features, num_classes=37, hidden_size=512):
        super().__init__()
        self.fc1 = LinearLayer(in_features, hidden_size)
        self.fc2 = LinearLayer(hidden_size, hidden_size)
        self.fc3 = LinearLayer(hidden_size, num_classes)

    def forward(self, x):
        x = ttml.ops.unary.relu(self.fc1(x))
        x = ttml.ops.unary.relu(self.fc2(x))
        return self.fc3(x)

    def cpu_state_dict(self):

        state = {}
        for name, parameter in self.parameters().items():
            array = parameter.to_numpy(ttnn.DataType.FLOAT32)
            state[name] = torch.from_numpy(np.array(array, copy=True))
        return state

    def load_cpu_state_dict(self, state):

        parameters = self.parameters()
        if parameters.keys() != state.keys():
            raise ValueError("Classifier checkpoint keys do not match")
        for name, parameter in parameters.items():
            if tuple(parameter.shape) != tuple(state[name].shape):
                raise ValueError(f"Classifier checkpoint shape mismatch: {name}")
            value = ttml.autograd.Tensor.from_numpy(
                state[name].float().numpy(), new_type=CLASSIFIER_DTYPE,
            )
            parameter.set_value(value.get_value())
