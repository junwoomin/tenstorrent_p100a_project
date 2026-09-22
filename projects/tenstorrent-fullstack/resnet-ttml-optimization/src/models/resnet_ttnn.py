from __future__ import annotations

import math
import random

import ttnn

from optimization_config import ACTIVATION_LAYOUT, HOST_DTYPE, WEIGHT_DTYPE
from .ttnn_config import apply_conv_options, pool_options
from .residual import add_relu


class TTConv2d:


    def __init__(self, in_channels: int, out_channels: int, device, *,
                 image_size: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, batch_size: int | None = None,
                 relu: bool = False, cnn_defaults=None, overrides=None,
                 act_block_h_override: int = 0,
                 deallocate_activation: bool = False, shard_layout=None,
                 bn_eps: float = 1e-5, initialize: bool = True):
        for name, value in (
            ('in_channels', in_channels), ('out_channels', out_channels),
            ('image_size', image_size), ('kernel_size', kernel_size),
            ('stride', stride),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
            raise ValueError('padding must be a nonnegative integer')
        if batch_size is not None and (
            not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0
        ):
            raise ValueError('batch_size must be a positive integer or None')
        if not math.isfinite(bn_eps) or bn_eps < 0:
            raise ValueError('bn_eps must be finite and nonnegative')

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.input_height = image_size
        self.input_width = image_size
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (1, 1)
        self.groups = 1
        self.batch_size = batch_size
        self.output_size = (image_size + 2 * padding - kernel_size) // stride + 1
        if self.output_size <= 0:
            raise ValueError('Convolution output size must be positive')
        self.bn_eps = bn_eps
        self.weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = None
        self.bias = None


        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=WEIGHT_DTYPE,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None,
            act_block_h_override=act_block_h_override,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=False,
            deallocate_activation=deallocate_activation,
            shard_layout=shard_layout,
            output_layout=ACTIVATION_LAYOUT,
            config_tensors_in_dram=True,
        )

        self.dtype, self.memory_config = apply_conv_options(
            self.conv_config, in_channels, out_channels, kernel_size, stride, padding,
            image_size=image_size, batch_size=batch_size,
            defaults=cnn_defaults, overrides=overrides,
        )
        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:

        kh, kw = self.kernel_size
        weight = ttnn.randn(
            (1, 1, self.out_channels, self.in_channels * kh * kw),
            device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, seed=random.getrandbits(32),
        )
        weight = ttnn.multiply(
            weight,
            math.sqrt(2.0 / (self.out_channels * kh * kw * (1.0 + self.bn_eps))),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        weight = ttnn.typecast(weight, HOST_DTYPE)
        weight = ttnn.to_layout(ttnn.from_device(weight), ttnn.ROW_MAJOR_LAYOUT)
        self.weight = ttnn.reshape(weight, self.weight_shape)
        self.bias = ttnn.zeros(
            (1, 1, 1, self.out_channels),
            dtype=HOST_DTYPE, layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def set_parameters(self, weight: ttnn.Tensor,
                       bias: ttnn.Tensor | None = None) -> None:

        if weight is None or tuple(weight.shape) != self.weight_shape:
            raise ValueError(f'Expected OIHW weight shape {self.weight_shape}')
        if bias is not None and tuple(bias.shape) != (1, 1, 1, self.out_channels):
            raise ValueError(f'Expected bias shape {(1, 1, 1, self.out_channels)}')
        self.weight = weight
        self.bias = bias

    def __call__(self, x: ttnn.Tensor, batch_size: int | None = None) -> ttnn.Tensor:
        if self.weight is None:
            raise RuntimeError('TTConv2d weight is not loaded; call set_parameters() or reset_parameters()')
        if batch_size is None:
            batch_size = self.batch_size
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size <= 0):
            raise ValueError('Pass a positive integer batch_size or set it in the constructor')


        out, (self.weight, self.bias) = ttnn.conv2d(
            input_tensor=x, weight_tensor=self.weight, bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels, out_channels=self.out_channels,
            input_height=self.input_height, input_width=self.input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
            dtype=self.dtype, conv_config=self.conv_config,
            memory_config=self.memory_config,
            return_output_dim=False, return_weights_and_bias=True,
        )
        return out


class TTBasicBlock:
    def __init__(self, in_channels, channels, device, *, image_size, stride=1,
                 batch_size=None, cnn_defaults=None, overrides=None, initialize=True):
        self.device = device
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = channels
        self.stride = stride
        self.image_size = image_size
        self.output_size = (image_size + stride - 1) // stride
        self.bn_eps = 1e-5
        overrides = overrides or {}
        settings = dict(batch_size=batch_size, cnn_defaults=cnn_defaults,
                        bn_eps=self.bn_eps, initialize=initialize)

        self.conv1 = TTConv2d(
            in_channels, channels, device, image_size=image_size,
            kernel_size=3, stride=stride, padding=1, relu=True,
            overrides=overrides.get('conv1'), **settings,
        )
        self.conv2 = TTConv2d(
            channels, channels, device, image_size=self.output_size,
            kernel_size=3, padding=1,
            overrides=overrides.get('conv2'), **settings,
        )
        self.downsample = None
        if stride != 1 or in_channels != self.out_channels:
            self.downsample = TTConv2d(
                in_channels, self.out_channels, device, image_size=image_size,
                kernel_size=1, stride=stride,
                overrides=overrides.get('downsample'), **settings,
            )

    def __call__(self, x, batch_size):
        out = self.conv1(x, batch_size)
        out = self.conv2(out, batch_size)
        shortcut = x if self.downsample is None else self.downsample(x, batch_size)
        return add_relu(out, shortcut)


class TTBottleneck:
    def __init__(self, in_channels, channels, device, *, image_size, stride=1,
                 batch_size=None, cnn_defaults=None, overrides=None, initialize=True):
        self.device = device
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = channels * 4
        self.stride = stride
        self.image_size = image_size
        self.output_size = (image_size + stride - 1) // stride
        self.bn_eps = 1e-5
        overrides = overrides or {}
        settings = dict(batch_size=batch_size, cnn_defaults=cnn_defaults,
                        bn_eps=self.bn_eps, initialize=initialize)

        self.conv1 = TTConv2d(
            in_channels, channels, device, image_size=image_size,
            kernel_size=1, relu=True,
            overrides=overrides.get('conv1'), **settings,
        )
        self.conv2 = TTConv2d(
            channels, channels, device, image_size=image_size,
            kernel_size=3, stride=stride, padding=1, relu=True,
            overrides=overrides.get('conv2'), **settings,
        )
        self.conv3 = TTConv2d(
            channels, self.out_channels, device, image_size=self.output_size,
            kernel_size=1,
            overrides=overrides.get('conv3'), **settings,
        )
        self.downsample = None
        if stride != 1 or in_channels != self.out_channels:
            self.downsample = TTConv2d(
                in_channels, self.out_channels, device, image_size=image_size,
                kernel_size=1, stride=stride,
                overrides=overrides.get('downsample'), **settings,
            )

    def __call__(self, x, batch_size):
        out = self.conv1(x, batch_size)
        out = self.conv2(out, batch_size)
        out = self.conv3(out, batch_size)
        shortcut = x if self.downsample is None else self.downsample(x, batch_size)
        return add_relu(out, shortcut)


class _ResNetIO:
    def __init__(self, device, image_size, out_features,
                 cnn_defaults, pool_defaults, layer_overrides, batch_size,
                 *, initialize=True):
        if image_size < 32:
            raise ValueError('ResNet requires image_size >= 32')
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.out_features = out_features
        self.bn_eps = 1e-5
        self.layer_overrides = dict(layer_overrides or {})
        self.block_overrides = {}
        for key, value in self.layer_overrides.items():
            if not isinstance(key, str):
                raise ValueError('layer_overrides keys must be strings')
            if key.startswith('layer'):
                parts = key.split('.')
                if (len(parts) != 3 or parts[0] not in ('layer1', 'layer2', 'layer3', 'layer4')
                        or not parts[1].isdigit()):
                    raise ValueError(f'Invalid layer override: {key!r}; expected layerN.index.conv')
                stage, index, conv = parts
                self.block_overrides.setdefault(stage + '_' + index, {})[conv] = value

        self.conv1 = TTConv2d(
            3, 64, device, image_size=image_size,
            kernel_size=7, stride=2, padding=3, relu=True,
            batch_size=batch_size, cnn_defaults=cnn_defaults,
            overrides=self.layer_overrides.get('conv1'),
            act_block_h_override=128, deallocate_activation=True,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            bn_eps=self.bn_eps, initialize=initialize,
        )

        self.stem_size = (image_size + 1) // 2
        self.pool_size = (self.stem_size + 1) // 2
        self.stage2_size = (self.pool_size + 1) // 2
        self.stage3_size = (self.stage2_size + 1) // 2
        self.stage4_size = (self.stage3_size + 1) // 2
        pool_defaults = pool_defaults or {}
        if set(pool_defaults) - {'max', 'global_avg'}:
            raise ValueError('pool_defaults keys must be max or global_avg')
        self.maxpool_options = pool_options(
            'max', self.stem_size, 64, batch_size,
            pool_defaults.get('max'), self.layer_overrides.get('pool1'),
        )
        self.avgpool_options = pool_options(
            'global_avg', self.stage4_size, out_features, batch_size,
            pool_defaults.get('global_avg'), self.layer_overrides.get('avgpool'),
        )
        self.pool_dtype = self.avgpool_options['dtype'] or ttnn.bfloat16

    def named_convs(self):

        yield 'conv1', self.conv1
        for attribute, block in vars(self).items():
            if isinstance(block, (TTBasicBlock, TTBottleneck)):
                prefix = attribute.replace('_', '.')
                for conv_name in ('conv1', 'conv2', 'conv3', 'downsample'):
                    conv = getattr(block, conv_name, None)
                    if isinstance(conv, TTConv2d):
                        yield prefix + '.' + conv_name, conv

    def reset_parameters(self):
        for _, conv in self.named_convs():
            conv.reset_parameters()

    def _validate_layer_overrides(self):
        allowed = {'pool1', 'avgpool'}
        allowed.update(name for name, _ in self.named_convs())
        unknown = self.layer_overrides.keys() - allowed
        if unknown:
            raise ValueError(f'Unknown layer overrides: {sorted(unknown)}')

    def pool1(self, x, batch_size):
        return ttnn.max_pool2d(
            input_tensor=x, batch_size=batch_size,
            input_h=self.stem_size, input_w=self.stem_size, channels=64,
            kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1],
            **self.maxpool_options,
        )

    def avgpool(self, x, batch_size):

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if self._pool_needs_cast:
            x = ttnn.typecast(x, dtype=self.pool_dtype)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch_size, self.stage4_size, self.stage4_size, self.out_features))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.global_avg_pool2d(x, **self.avgpool_options)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (1, 1, batch_size, self.out_features))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


class TTNNResNet18(_ResNetIO):
    def __init__(self, device=None, image_size=224, *, batch_size=None,
                 cnn_defaults=None, pool_defaults=None, layer_overrides=None,
                 initialize=True):
        super().__init__(device, image_size, 512,
                         cnn_defaults, pool_defaults, layer_overrides, batch_size,
                         initialize=initialize)
        settings = dict(cnn_defaults=cnn_defaults, initialize=initialize)


        self.layer1_0 = TTBasicBlock(
            64, 64, device, stride=1, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer1_0'), **settings,
        )
        self.layer1_1 = TTBasicBlock(
            64, 64, device, stride=1, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer1_1'), **settings,
        )


        self.layer2_0 = TTBasicBlock(
            64, 128, device, stride=2, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_0'), **settings,
        )
        self.layer2_1 = TTBasicBlock(
            128, 128, device, stride=1, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_1'), **settings,
        )


        self.layer3_0 = TTBasicBlock(
            128, 256, device, stride=2, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_0'), **settings,
        )
        self.layer3_1 = TTBasicBlock(
            256, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_1'), **settings,
        )


        self.layer4_0 = TTBasicBlock(
            256, 512, device, stride=2, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer4_0'), **settings,
        )
        self.layer4_1 = TTBasicBlock(
            512, 512, device, stride=1, image_size=self.stage4_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer4_1'), **settings,
        )

        self._pool_needs_cast = self.layer4_1.conv2.dtype != self.pool_dtype
        self._validate_layer_overrides()

    def __call__(self, images):
        batch_size = images.shape[0]
        x = self.conv1(images, batch_size)
        x = self.pool1(x, batch_size)

        x = self.layer1_0(x, batch_size)
        x = self.layer1_1(x, batch_size)
        x = self.layer2_0(x, batch_size)
        x = self.layer2_1(x, batch_size)
        x = self.layer3_0(x, batch_size)
        x = self.layer3_1(x, batch_size)
        x = self.layer4_0(x, batch_size)
        x = self.layer4_1(x, batch_size)

        return self.avgpool(x, batch_size)


class TTNNResNet50(_ResNetIO):
    def __init__(self, device=None, image_size=224, *, batch_size=None,
                 cnn_defaults=None, pool_defaults=None, layer_overrides=None,
                 initialize=True):
        super().__init__(device, image_size, 2048,
                         cnn_defaults, pool_defaults, layer_overrides, batch_size,
                         initialize=initialize)
        settings = dict(cnn_defaults=cnn_defaults, initialize=initialize)


        self.layer1_0 = TTBottleneck(
            64, 64, device, stride=1, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer1_0'), **settings,
        )
        self.layer1_1 = TTBottleneck(
            256, 64, device, stride=1, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer1_1'), **settings,
        )
        self.layer1_2 = TTBottleneck(
            256, 64, device, stride=1, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer1_2'), **settings,
        )


        self.layer2_0 = TTBottleneck(
            256, 128, device, stride=2, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_0'), **settings,
        )
        self.layer2_1 = TTBottleneck(
            512, 128, device, stride=1, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_1'), **settings,
        )
        self.layer2_2 = TTBottleneck(
            512, 128, device, stride=1, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_2'), **settings,
        )
        self.layer2_3 = TTBottleneck(
            512, 128, device, stride=1, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_3'), **settings,
        )


        self.layer3_0 = TTBottleneck(
            512, 256, device, stride=2, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_0'), **settings,
        )
        self.layer3_1 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_1'), **settings,
        )
        self.layer3_2 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_2'), **settings,
        )
        self.layer3_3 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_3'), **settings,
        )
        self.layer3_4 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_4'), **settings,
        )
        self.layer3_5 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_5'), **settings,
        )


        self.layer4_0 = TTBottleneck(
            1024, 512, device, stride=2, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer4_0'), **settings,
        )
        self.layer4_1 = TTBottleneck(
            2048, 512, device, stride=1, image_size=self.stage4_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer4_1'), **settings,
        )
        self.layer4_2 = TTBottleneck(
            2048, 512, device, stride=1, image_size=self.stage4_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer4_2'), **settings,
        )

        self._pool_needs_cast = self.layer4_2.conv3.dtype != self.pool_dtype
        self._validate_layer_overrides()

    def __call__(self, images):
        batch_size = images.shape[0]
        x = self.conv1(images, batch_size)
        x = self.pool1(x, batch_size)

        x = self.layer1_0(x, batch_size)
        x = self.layer1_1(x, batch_size)
        x = self.layer1_2(x, batch_size)
        x = self.layer2_0(x, batch_size)
        x = self.layer2_1(x, batch_size)
        x = self.layer2_2(x, batch_size)
        x = self.layer2_3(x, batch_size)
        x = self.layer3_0(x, batch_size)
        x = self.layer3_1(x, batch_size)
        x = self.layer3_2(x, batch_size)
        x = self.layer3_3(x, batch_size)
        x = self.layer3_4(x, batch_size)
        x = self.layer3_5(x, batch_size)
        x = self.layer4_0(x, batch_size)
        x = self.layer4_1(x, batch_size)
        x = self.layer4_2(x, batch_size)

        return self.avgpool(x, batch_size)


class TTNNResNet101(_ResNetIO):
    def __init__(self, device=None, image_size=224, *, batch_size=None,
                 cnn_defaults=None, pool_defaults=None, layer_overrides=None,
                 initialize=True):
        super().__init__(device, image_size, 2048,
                         cnn_defaults, pool_defaults, layer_overrides, batch_size,
                         initialize=initialize)
        settings = dict(cnn_defaults=cnn_defaults, initialize=initialize)


        self.layer1_0 = TTBottleneck(
            64, 64, device, stride=1, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer1_0'), **settings,
        )
        self.layer1_1 = TTBottleneck(
            256, 64, device, stride=1, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer1_1'), **settings,
        )
        self.layer1_2 = TTBottleneck(
            256, 64, device, stride=1, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer1_2'), **settings,
        )


        self.layer2_0 = TTBottleneck(
            256, 128, device, stride=2, image_size=self.pool_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_0'), **settings,
        )
        self.layer2_1 = TTBottleneck(
            512, 128, device, stride=1, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_1'), **settings,
        )
        self.layer2_2 = TTBottleneck(
            512, 128, device, stride=1, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_2'), **settings,
        )
        self.layer2_3 = TTBottleneck(
            512, 128, device, stride=1, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer2_3'), **settings,
        )


        self.layer3_0 = TTBottleneck(
            512, 256, device, stride=2, image_size=self.stage2_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_0'), **settings,
        )
        self.layer3_1 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_1'), **settings,
        )
        self.layer3_2 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_2'), **settings,
        )
        self.layer3_3 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_3'), **settings,
        )
        self.layer3_4 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_4'), **settings,
        )
        self.layer3_5 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_5'), **settings,
        )
        self.layer3_6 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_6'), **settings,
        )
        self.layer3_7 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_7'), **settings,
        )
        self.layer3_8 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_8'), **settings,
        )
        self.layer3_9 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_9'), **settings,
        )
        self.layer3_10 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_10'), **settings,
        )
        self.layer3_11 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_11'), **settings,
        )
        self.layer3_12 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_12'), **settings,
        )
        self.layer3_13 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_13'), **settings,
        )
        self.layer3_14 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_14'), **settings,
        )
        self.layer3_15 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_15'), **settings,
        )
        self.layer3_16 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_16'), **settings,
        )
        self.layer3_17 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_17'), **settings,
        )
        self.layer3_18 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_18'), **settings,
        )
        self.layer3_19 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_19'), **settings,
        )
        self.layer3_20 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_20'), **settings,
        )
        self.layer3_21 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_21'), **settings,
        )
        self.layer3_22 = TTBottleneck(
            1024, 256, device, stride=1, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer3_22'), **settings,
        )


        self.layer4_0 = TTBottleneck(
            1024, 512, device, stride=2, image_size=self.stage3_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer4_0'), **settings,
        )
        self.layer4_1 = TTBottleneck(
            2048, 512, device, stride=1, image_size=self.stage4_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer4_1'), **settings,
        )
        self.layer4_2 = TTBottleneck(
            2048, 512, device, stride=1, image_size=self.stage4_size,
            batch_size=batch_size, overrides=self.block_overrides.get('layer4_2'), **settings,
        )

        self._pool_needs_cast = self.layer4_2.conv3.dtype != self.pool_dtype
        self._validate_layer_overrides()

    def __call__(self, images):
        batch_size = images.shape[0]
        x = self.conv1(images, batch_size)
        x = self.pool1(x, batch_size)

        x = self.layer1_0(x, batch_size)
        x = self.layer1_1(x, batch_size)
        x = self.layer1_2(x, batch_size)
        x = self.layer2_0(x, batch_size)
        x = self.layer2_1(x, batch_size)
        x = self.layer2_2(x, batch_size)
        x = self.layer2_3(x, batch_size)
        x = self.layer3_0(x, batch_size)
        x = self.layer3_1(x, batch_size)
        x = self.layer3_2(x, batch_size)
        x = self.layer3_3(x, batch_size)
        x = self.layer3_4(x, batch_size)
        x = self.layer3_5(x, batch_size)
        x = self.layer3_6(x, batch_size)
        x = self.layer3_7(x, batch_size)
        x = self.layer3_8(x, batch_size)
        x = self.layer3_9(x, batch_size)
        x = self.layer3_10(x, batch_size)
        x = self.layer3_11(x, batch_size)
        x = self.layer3_12(x, batch_size)
        x = self.layer3_13(x, batch_size)
        x = self.layer3_14(x, batch_size)
        x = self.layer3_15(x, batch_size)
        x = self.layer3_16(x, batch_size)
        x = self.layer3_17(x, batch_size)
        x = self.layer3_18(x, batch_size)
        x = self.layer3_19(x, batch_size)
        x = self.layer3_20(x, batch_size)
        x = self.layer3_21(x, batch_size)
        x = self.layer3_22(x, batch_size)
        x = self.layer4_0(x, batch_size)
        x = self.layer4_1(x, batch_size)
        x = self.layer4_2(x, batch_size)

        return self.avgpool(x, batch_size)
