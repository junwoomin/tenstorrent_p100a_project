from .ttnn_conv import TTNNStemConv
from .tt_layers import TTConv2d, TTMaxPool2d, TTGlobalAvgPool2d, TTFlatten, TTLinear, tt_add, tt_relu
from .tt_tensor import input_value, result_value


class TTBasicBlock:
    def __init__(self, in_channels, channels, device, *, stride=1, name,
                 cnn_defaults=None, layer_overrides=None):
        self.name = name
        options = layer_overrides or {}
        self.conv1 = TTConv2d(in_channels, channels, device, 3, stride, 1,
            batch_norm=True, relu=True, name=name+'.conv1',
            defaults=cnn_defaults, config=options.get(name+'.conv1'))
        self.conv2 = TTConv2d(channels, channels, device, 3, 1, 1,
            batch_norm=True, relu=False, name=name+'.conv2',
            defaults=cnn_defaults, config=options.get(name+'.conv2'))
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = TTConv2d(in_channels, channels, device, 1, stride, 0,
                batch_norm=True, relu=False, name=name+'.downsample',
                defaults=cnn_defaults, config=options.get(name+'.downsample'))

    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = tt_add(out, identity, name=self.name+'.add')
        return tt_relu(out, name=self.name+'.relu')


class TTBottleneck:
    def __init__(self, in_channels, channels, device, *, stride=1, name,
                 cnn_defaults=None, layer_overrides=None):
        self.name = name
        options = layer_overrides or {}
        self.conv1 = TTConv2d(in_channels, channels, device, 1, 1, 0,
            batch_norm=True, relu=True, name=name+'.conv1',
            defaults=cnn_defaults, config=options.get(name+'.conv1'))

        self.conv2 = TTConv2d(channels, channels, device, 3, stride, 1,
            batch_norm=True, relu=True, name=name+'.conv2',
            defaults=cnn_defaults, config=options.get(name+'.conv2'))
        self.conv3 = TTConv2d(channels, channels*4, device, 1, 1, 0,
            batch_norm=True, relu=False, name=name+'.conv3',
            defaults=cnn_defaults, config=options.get(name+'.conv3'))
        self.downsample = None
        if stride != 1 or in_channels != channels*4:
            self.downsample = TTConv2d(in_channels, channels*4, device, 1, stride, 0,
                batch_norm=True, relu=False, name=name+'.downsample',
                defaults=cnn_defaults, config=options.get(name+'.downsample'))

    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = tt_add(out, identity, name=self.name+'.add')
        return tt_relu(out, name=self.name+'.relu')


class _ResNetIO:

    def __init__(self, device, image_size, out_features, num_classes,
                 cnn_defaults, pool_defaults, layer_overrides, batch_size):
        if image_size < 32:
            raise ValueError('ResNet requires image_size >= 32')
        self.device, self.image_size, self.batch_size = device, image_size, batch_size
        self.out_features = out_features
        self.num_classes = num_classes
        options, pools = layer_overrides or {}, pool_defaults or {}
        if set(pools) - {'max', 'global_avg'}:
            raise ValueError('pool_defaults keys must be max or global_avg')
        self.conv1 = TTNNStemConv(device, defaults=cnn_defaults, config=options.get('conv1'))
        self.pool1 = TTMaxPool2d(3, 2, 1, name='pool1',
            defaults=pools.get('max'), config=options.get('pool1'))
        self.avgpool = TTGlobalAvgPool2d(name='avgpool',
            defaults=pools.get('global_avg'), config=options.get('avgpool'))
        self.flatten = TTFlatten()
        self.fc = TTLinear(out_features, num_classes, device, name='fc') if num_classes is not None else None

    def check_overrides(self, overrides):

        from .tt_tensor import trace_model
        graph = trace_model(self, (self.batch_size or 1, 3, self.image_size, self.image_size))
        allowed = {op.name for op in graph.ops if isinstance(op.owner, (TTConv2d, TTMaxPool2d, TTGlobalAvgPool2d))}
        unknown = set(overrides or {}) - allowed
        if unknown:
            raise ValueError(f'Unknown layer overrides: {sorted(unknown)}')

    def named_layers(self):
        from .tt_tensor import trace_model
        graph = trace_model(self, (self.batch_size or 1, 3, self.image_size, self.image_size))
        return ((op.name, op.owner) for op in graph.ops if op.owner is not None)


class TTNNResNet18(_ResNetIO):
    def __init__(self, device=None, image_size=224, num_classes=None, *, batch_size=None,
                 cnn_defaults=None, pool_defaults=None, layer_overrides=None):
        super().__init__(device, image_size, 512, num_classes,
                         cnn_defaults, pool_defaults, layer_overrides, batch_size)
        settings = dict(cnn_defaults=cnn_defaults, layer_overrides=layer_overrides)


        self.layer1_0 = TTBasicBlock(64, 64, device, stride=1, name="layer1.0", **settings)
        self.layer1_1 = TTBasicBlock(64, 64, device, stride=1, name="layer1.1", **settings)


        self.layer2_0 = TTBasicBlock(64, 128, device, stride=2, name="layer2.0", **settings)
        self.layer2_1 = TTBasicBlock(128, 128, device, stride=1, name="layer2.1", **settings)


        self.layer3_0 = TTBasicBlock(128, 256, device, stride=2, name="layer3.0", **settings)
        self.layer3_1 = TTBasicBlock(256, 256, device, stride=1, name="layer3.1", **settings)


        self.layer4_0 = TTBasicBlock(256, 512, device, stride=2, name="layer4.0", **settings)
        self.layer4_1 = TTBasicBlock(512, 512, device, stride=1, name="layer4.1", **settings)
        self.check_overrides(layer_overrides)

    def __call__(self, images):
        x = input_value(images, self.image_size)
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        if self.fc is not None:
            x = self.fc(x)
        return result_value(x)


class TTNNResNet50(_ResNetIO):
    def __init__(self, device=None, image_size=224, num_classes=None, *, batch_size=None,
                 cnn_defaults=None, pool_defaults=None, layer_overrides=None):
        super().__init__(device, image_size, 2048, num_classes,
                         cnn_defaults, pool_defaults, layer_overrides, batch_size)
        settings = dict(cnn_defaults=cnn_defaults, layer_overrides=layer_overrides)


        self.layer1_0 = TTBottleneck(64, 64, device, stride=1, name="layer1.0", **settings)
        self.layer1_1 = TTBottleneck(256, 64, device, stride=1, name="layer1.1", **settings)
        self.layer1_2 = TTBottleneck(256, 64, device, stride=1, name="layer1.2", **settings)


        self.layer2_0 = TTBottleneck(256, 128, device, stride=2, name="layer2.0", **settings)
        self.layer2_1 = TTBottleneck(512, 128, device, stride=1, name="layer2.1", **settings)
        self.layer2_2 = TTBottleneck(512, 128, device, stride=1, name="layer2.2", **settings)
        self.layer2_3 = TTBottleneck(512, 128, device, stride=1, name="layer2.3", **settings)


        self.layer3_0 = TTBottleneck(512, 256, device, stride=2, name="layer3.0", **settings)
        self.layer3_1 = TTBottleneck(1024, 256, device, stride=1, name="layer3.1", **settings)
        self.layer3_2 = TTBottleneck(1024, 256, device, stride=1, name="layer3.2", **settings)
        self.layer3_3 = TTBottleneck(1024, 256, device, stride=1, name="layer3.3", **settings)
        self.layer3_4 = TTBottleneck(1024, 256, device, stride=1, name="layer3.4", **settings)
        self.layer3_5 = TTBottleneck(1024, 256, device, stride=1, name="layer3.5", **settings)


        self.layer4_0 = TTBottleneck(1024, 512, device, stride=2, name="layer4.0", **settings)
        self.layer4_1 = TTBottleneck(2048, 512, device, stride=1, name="layer4.1", **settings)
        self.layer4_2 = TTBottleneck(2048, 512, device, stride=1, name="layer4.2", **settings)
        self.check_overrides(layer_overrides)

    def __call__(self, images):
        x = input_value(images, self.image_size)
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        if self.fc is not None:
            x = self.fc(x)
        return result_value(x)


class TTNNResNet101(_ResNetIO):
    def __init__(self, device=None, image_size=224, num_classes=None, *, batch_size=None,
                 cnn_defaults=None, pool_defaults=None, layer_overrides=None):
        super().__init__(device, image_size, 2048, num_classes,
                         cnn_defaults, pool_defaults, layer_overrides, batch_size)
        settings = dict(cnn_defaults=cnn_defaults, layer_overrides=layer_overrides)


        self.layer1_0 = TTBottleneck(64, 64, device, stride=1, name="layer1.0", **settings)
        self.layer1_1 = TTBottleneck(256, 64, device, stride=1, name="layer1.1", **settings)
        self.layer1_2 = TTBottleneck(256, 64, device, stride=1, name="layer1.2", **settings)


        self.layer2_0 = TTBottleneck(256, 128, device, stride=2, name="layer2.0", **settings)
        self.layer2_1 = TTBottleneck(512, 128, device, stride=1, name="layer2.1", **settings)
        self.layer2_2 = TTBottleneck(512, 128, device, stride=1, name="layer2.2", **settings)
        self.layer2_3 = TTBottleneck(512, 128, device, stride=1, name="layer2.3", **settings)


        self.layer3_0 = TTBottleneck(512, 256, device, stride=2, name="layer3.0", **settings)
        self.layer3_1 = TTBottleneck(1024, 256, device, stride=1, name="layer3.1", **settings)
        self.layer3_2 = TTBottleneck(1024, 256, device, stride=1, name="layer3.2", **settings)
        self.layer3_3 = TTBottleneck(1024, 256, device, stride=1, name="layer3.3", **settings)
        self.layer3_4 = TTBottleneck(1024, 256, device, stride=1, name="layer3.4", **settings)
        self.layer3_5 = TTBottleneck(1024, 256, device, stride=1, name="layer3.5", **settings)
        self.layer3_6 = TTBottleneck(1024, 256, device, stride=1, name="layer3.6", **settings)
        self.layer3_7 = TTBottleneck(1024, 256, device, stride=1, name="layer3.7", **settings)
        self.layer3_8 = TTBottleneck(1024, 256, device, stride=1, name="layer3.8", **settings)
        self.layer3_9 = TTBottleneck(1024, 256, device, stride=1, name="layer3.9", **settings)
        self.layer3_10 = TTBottleneck(1024, 256, device, stride=1, name="layer3.10", **settings)
        self.layer3_11 = TTBottleneck(1024, 256, device, stride=1, name="layer3.11", **settings)
        self.layer3_12 = TTBottleneck(1024, 256, device, stride=1, name="layer3.12", **settings)
        self.layer3_13 = TTBottleneck(1024, 256, device, stride=1, name="layer3.13", **settings)
        self.layer3_14 = TTBottleneck(1024, 256, device, stride=1, name="layer3.14", **settings)
        self.layer3_15 = TTBottleneck(1024, 256, device, stride=1, name="layer3.15", **settings)
        self.layer3_16 = TTBottleneck(1024, 256, device, stride=1, name="layer3.16", **settings)
        self.layer3_17 = TTBottleneck(1024, 256, device, stride=1, name="layer3.17", **settings)
        self.layer3_18 = TTBottleneck(1024, 256, device, stride=1, name="layer3.18", **settings)
        self.layer3_19 = TTBottleneck(1024, 256, device, stride=1, name="layer3.19", **settings)
        self.layer3_20 = TTBottleneck(1024, 256, device, stride=1, name="layer3.20", **settings)
        self.layer3_21 = TTBottleneck(1024, 256, device, stride=1, name="layer3.21", **settings)
        self.layer3_22 = TTBottleneck(1024, 256, device, stride=1, name="layer3.22", **settings)


        self.layer4_0 = TTBottleneck(1024, 512, device, stride=2, name="layer4.0", **settings)
        self.layer4_1 = TTBottleneck(2048, 512, device, stride=1, name="layer4.1", **settings)
        self.layer4_2 = TTBottleneck(2048, 512, device, stride=1, name="layer4.2", **settings)
        self.check_overrides(layer_overrides)

    def __call__(self, images):
        x = input_value(images, self.image_size)
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = self.layer3_6(x)
        x = self.layer3_7(x)
        x = self.layer3_8(x)
        x = self.layer3_9(x)
        x = self.layer3_10(x)
        x = self.layer3_11(x)
        x = self.layer3_12(x)
        x = self.layer3_13(x)
        x = self.layer3_14(x)
        x = self.layer3_15(x)
        x = self.layer3_16(x)
        x = self.layer3_17(x)
        x = self.layer3_18(x)
        x = self.layer3_19(x)
        x = self.layer3_20(x)
        x = self.layer3_21(x)
        x = self.layer3_22(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        if self.fc is not None:
            x = self.fc(x)
        return result_value(x)
