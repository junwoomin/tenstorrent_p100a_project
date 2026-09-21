import numpy as np
import torch
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

from dtype_config import MODEL_TORCH_DTYPE
from vgg import (
    ACTIVATION_DTYPE,
    ACTIVATION_LAYOUT,
    CLASSIFIER_DTYPE,
    HOST_DTYPE,
    WEIGHT_DTYPE,
    TTConv2d,
    tt_max_pool,
)


class FrozenVGG11:
    def __init__(
        self,
        device,
        image_size=112,
        weights_path=None,
    ):
        if image_size < 32:
            raise ValueError("VGG11 requires image_size >= 32")

        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
        else:
            from torchvision.models import VGG11_Weights
            state = VGG11_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        self.device = device
        self.image_size = image_size

        self.out_features = 512 * (image_size // 32) ** 2
        self.state = {}

        self.conv1 = self._make_conv(
            state, feature_index=0, in_channels=3, out_channels=64,
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=64,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=True,
            deallocate_activation=True,
            output_layout=ACTIVATION_LAYOUT,

            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self.conv2 = self._make_conv(
            state, feature_index=3, in_channels=64, out_channels=128,
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=0,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=False,
            deallocate_activation=False,
            output_layout=ACTIVATION_LAYOUT,

            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        )
        self.conv3 = self._make_conv(
            state, feature_index=6, in_channels=128, out_channels=256,
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=0,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=False,
            deallocate_activation=False,
            output_layout=ACTIVATION_LAYOUT,

        )
        self.conv4 = self._make_conv(
            state, feature_index=8, in_channels=256, out_channels=256,
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=0,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=False,
            deallocate_activation=False,
            output_layout=ACTIVATION_LAYOUT,

            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        )
        self.conv5 = self._make_conv(
            state, feature_index=11, in_channels=256, out_channels=512,
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=0,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=False,
            deallocate_activation=False,
            output_layout=ACTIVATION_LAYOUT,

        )
        self.conv6 = self._make_conv(
            state, feature_index=13, in_channels=512, out_channels=512,
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=0,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=False,
            deallocate_activation=False,
            output_layout=ACTIVATION_LAYOUT,

        )
        self.conv7 = self._make_conv(
            state, feature_index=16, in_channels=512, out_channels=512,
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=0,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=False,
            deallocate_activation=False,
            output_layout=ACTIVATION_LAYOUT,

        )
        self.conv8 = self._make_conv(
            state, feature_index=18, in_channels=512, out_channels=512,
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h_override=0,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=False,
            deallocate_activation=False,
            output_layout=ACTIVATION_LAYOUT,
        )

    def _make_conv(
        self,
        state,
        feature_index,
        in_channels,
        out_channels,
        weights_dtype,
        activation,
        act_block_h_override,
        enable_act_double_buffer,
        enable_weights_double_buffer,
        reshard_if_not_optimal,
        deallocate_activation,
        output_layout,
        shard_layout=None,
    ):
        layer = TTConv2d(
            in_channels,
            out_channels,
            self.device,
            weights_dtype=weights_dtype,
            activation=activation,
            act_block_h_override=act_block_h_override,
            enable_act_double_buffer=enable_act_double_buffer,
            enable_weights_double_buffer=enable_weights_double_buffer,
            reshard_if_not_optimal=reshard_if_not_optimal,
            deallocate_activation=deallocate_activation,
            output_layout=output_layout,
            shard_layout=shard_layout,
        )
        prefix = f"features.{feature_index}"
        weight = state[f"{prefix}.weight"].detach().cpu().contiguous()
        bias = state[f"{prefix}.bias"].detach().cpu().contiguous()
        expected = (out_channels, in_channels, 3, 3)
        if tuple(weight.shape) != expected or tuple(bias.shape) != (out_channels,):
            raise ValueError(f"Invalid VGG11 weights at {prefix}")
        self.state[f"{prefix}.weight"] = weight
        self.state[f"{prefix}.bias"] = bias
        layer.weight = ttnn.from_torch(
            weight.to(MODEL_TORCH_DTYPE), dtype=HOST_DTYPE,
        )
        layer.bias = ttnn.from_torch(
            bias.to(MODEL_TORCH_DTYPE).reshape(1, 1, 1, -1),
            dtype=HOST_DTYPE,
        )
        return layer

    def __call__(self, images):
        batch = images.shape[0]

        h = w = self.image_size


        x, h, w = self.conv1(images, batch, h, w)
        x, h, w = tt_max_pool(x, batch, h, w, 64)


        x, h, w = self.conv2(x, batch, h, w)
        x, h, w = tt_max_pool(x, batch, h, w, 128)


        x, h, w = self.conv3(x, batch, h, w)
        x, h, w = self.conv4(x, batch, h, w)
        x, h, w = tt_max_pool(x, batch, h, w, 256)


        x, h, w = self.conv5(x, batch, h, w)
        x, h, w = self.conv6(x, batch, h, w)
        x, h, w = tt_max_pool(x, batch, h, w, 512)


        x, h, w = self.conv7(x, batch, h, w)
        x, h, w = self.conv8(x, batch, h, w)
        x, h, w = tt_max_pool(x, batch, h, w, 512)


        x = ttnn.reshape(x, (1, 1, batch, self.out_features))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if ACTIVATION_DTYPE != CLASSIFIER_DTYPE:
            x = ttnn.typecast(x, CLASSIFIER_DTYPE)
        return ttml.autograd.create_tensor(x, requires_grad=False)


class VGGClassifier(AbstractModuleBase):

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
