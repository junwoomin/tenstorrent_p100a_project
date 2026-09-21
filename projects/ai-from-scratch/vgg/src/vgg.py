import math
import torch
import ttnn

from dtype_config import MODEL_TORCH_DTYPE, get_ttnn_dtype, get_ttnn_layout


ACTIVATION_DTYPE = get_ttnn_dtype(ttnn, "activation")
WEIGHT_DTYPE = get_ttnn_dtype(ttnn, "weight")
CLASSIFIER_DTYPE = get_ttnn_dtype(ttnn, "classifier")
HOST_DTYPE = get_ttnn_dtype(ttnn, "host")
ACTIVATION_LAYOUT = get_ttnn_layout(ttnn)
DEFAULT_CONV_ACTIVATION = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)


class TTConv2d:
    def __init__(
        self,
        in_channels,
        out_channels,
        device,
        kernel_size=3,
        stride=1,
        padding=1,
        weights_dtype=WEIGHT_DTYPE,
        activation=DEFAULT_CONV_ACTIVATION,
        act_block_h_override=0,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        reshard_if_not_optimal=False,
        deallocate_activation=False,
        output_layout=ACTIVATION_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        activation_dtype=ACTIVATION_DTYPE,
        dilation=(1, 1),
        groups=1,
        shard_layout=None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.device = device

        weight = torch.empty(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            dtype=MODEL_TORCH_DTYPE,
        )

        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)

        weight.normal_(
            mean=0.0,
            std=std,
        )

        bias = torch.zeros(
            (1, 1, 1, out_channels),
            dtype=MODEL_TORCH_DTYPE,
        )

        self.weight = ttnn.from_torch(
            weight,

            dtype=HOST_DTYPE,
        )

        self.bias = ttnn.from_torch(
            bias,
            dtype=HOST_DTYPE,
        )

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            activation=activation,

            act_block_h_override=act_block_h_override,
            enable_act_double_buffer=enable_act_double_buffer,
            enable_weights_double_buffer=enable_weights_double_buffer,
            reshard_if_not_optimal=reshard_if_not_optimal,

            deallocate_activation=deallocate_activation,
            output_layout=output_layout,
            shard_layout=shard_layout,
            config_tensors_in_dram=True,

        )
        self.weights_prepared = False


        self.common_args = dict(
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=dilation,
            groups=groups,
            dtype=activation_dtype,
            conv_config=self.conv_config,
            return_output_dim=True,
            memory_config=memory_config,
        )

    def __call__(self, x, batch_size, height, width):
        if not self.weights_prepared:
            x, output_dim, prepared = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.weight,
                bias_tensor=self.bias,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                **self.common_args,
                return_weights_and_bias=True,

            )

            self.weight, self.bias = prepared
            self.weights_prepared = True

        else:
            x, output_dim = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.weight,
                bias_tensor=self.bias,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                **self.common_args,
                return_weights_and_bias=False,
            )

        return x, output_dim[0], output_dim[1]


def tt_max_pool(
    x,
    batch_size,
    height,
    width,
    channels,
):

    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=height,
        input_w=width,
        channels=channels,
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
        ceil_mode=False,

        config_tensor_in_dram=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        applied_shard_scheme=None,

        deallocate_input=False,

        dtype=ACTIVATION_DTYPE,
        output_layout=ACTIVATION_LAYOUT,
    )

    height = height // 2
    width = width // 2


    return x, height, width


class TTLinear:
    def __init__(
        self,
        in_features,
        out_features,
        device,
        relu=False,
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.relu = relu

        weight = torch.empty(
            in_features,
            out_features,
            dtype=MODEL_TORCH_DTYPE,
        )

        bound = math.sqrt(2.0 / in_features)

        weight.normal_(
            mean=0.0,
            std=bound,
        )

        bias = torch.zeros(
            (1, 1, 1, out_features),
            dtype=MODEL_TORCH_DTYPE,
        )

        self.weight = ttnn.from_torch(
            weight,
            device=device,
            dtype=WEIGHT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
        )

        self.bias = ttnn.from_torch(
            bias,
            device=device,
            dtype=ACTIVATION_DTYPE,
            layout=ttnn.TILE_LAYOUT,
        )

    def __call__(self, x):

        x = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            dtype=ACTIVATION_DTYPE,
        )

        if self.relu:
            x = ttnn.relu(x)

        return x


class TTVGG11:

    def __init__(
        self,
        device,
        image_size=112,
        num_classes=37,
    ):

        self.device = device
        self.image_size = image_size
        self.num_classes = num_classes


        self.conv1 = TTConv2d(
            3,
            64,
            device,
        )

        self.conv2 = TTConv2d(
            64,
            128,
            device,
        )

        self.conv3 = TTConv2d(
            128,
            256,
            device,
        )

        self.conv4 = TTConv2d(
            256,
            256,
            device,
        )

        self.conv5 = TTConv2d(
            256,
            512,
            device,
        )

        self.conv6 = TTConv2d(
            512,
            512,
            device,
        )

        self.conv7 = TTConv2d(
            512,
            512,
            device,
        )

        self.conv8 = TTConv2d(
            512,
            512,
            device,
        )


        final_size = image_size

        for _ in range(5):
            final_size //= 2

        self.final_size = final_size

        flatten_size = (
            512
            * final_size
            * final_size
        )

        print(
            f"[VGG11] Final feature map: "
            f"512 x {final_size} x {final_size}"
        )

        print(
            f"[VGG11] Flatten size: "
            f"{flatten_size}"
        )


        self.fc1 = TTLinear(
            flatten_size,
            4096,
            device,
            relu=True,
        )

        self.fc2 = TTLinear(
            4096,
            4096,
            device,
            relu=True,
        )

        self.fc3 = TTLinear(
            4096,
            num_classes,
            device,
            relu=False,
        )


    def __call__(
        self,
        x,
        batch_size,
    ):

        h = self.image_size
        w = self.image_size


        x, h, w = self.conv1(
            x,
            batch_size,
            h,
            w,
        )

        print(
            f"conv1 : "
            f"{h} x {w} x 64"
        )

        x, h, w = tt_max_pool(
            x,
            batch_size,
            h,
            w,
            64,
        )

        print(
            f"pool1 : "
            f"{h} x {w} x 64"
        )


        x, h, w = self.conv2(
            x,
            batch_size,
            h,
            w,
        )

        print(
            f"conv2 : "
            f"{h} x {w} x 128"
        )

        x, h, w = tt_max_pool(
            x,
            batch_size,
            h,
            w,
            128,
        )

        print(
            f"pool2 : "
            f"{h} x {w} x 128"
        )


        x, h, w = self.conv3(
            x,
            batch_size,
            h,
            w,
        )

        x, h, w = self.conv4(
            x,
            batch_size,
            h,
            w,
        )

        print(
            f"conv4 : "
            f"{h} x {w} x 256"
        )

        x, h, w = tt_max_pool(
            x,
            batch_size,
            h,
            w,
            256,
        )

        print(
            f"pool3 : "
            f"{h} x {w} x 256"
        )


        x, h, w = self.conv5(
            x,
            batch_size,
            h,
            w,
        )

        x, h, w = self.conv6(
            x,
            batch_size,
            h,
            w,
        )

        print(
            f"conv6 : "
            f"{h} x {w} x 512"
        )

        x, h, w = tt_max_pool(
            x,
            batch_size,
            h,
            w,
            512,
        )

        print(
            f"pool4 : "
            f"{h} x {w} x 512"
        )


        x, h, w = self.conv7(
            x,
            batch_size,
            h,
            w,
        )

        x, h, w = self.conv8(
            x,
            batch_size,
            h,
            w,
        )

        print(
            f"conv8 : "
            f"{h} x {w} x 512"
        )

        x, h, w = tt_max_pool(
            x,
            batch_size,
            h,
            w,
            512,
        )

        print(
            f"pool5 : "
            f"{h} x {w} x 512"
        )


        flatten_size = (
            h
            * w
            * 512
        )

        x = ttnn.reshape(
            x,
            (
                batch_size,
                1,
                1,
                flatten_size,
            ),
        )


        x = ttnn.to_layout(
            x,
            ttnn.TILE_LAYOUT,
        )

        print(
            f"flatten : "
            f"{batch_size} x {flatten_size}"
        )


        x = self.fc1(x)

        print(
            f"fc1     : "
            f"{batch_size} x 4096"
        )

        x = self.fc2(x)

        print(
            f"fc2     : "
            f"{batch_size} x 4096"
        )

        x = self.fc3(x)

        print(
            f"fc3     : "
            f"{batch_size} x {self.num_classes}"
        )

        return x
