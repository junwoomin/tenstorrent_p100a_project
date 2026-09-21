from compat.graph import pair
from .tt_tensor import apply


class TTConv2d:
    def __init__(self, in_channels, out_channels, device=None, kernel_size=3,
                 stride=1, padding=1, *, dilation=1, groups=1, bias=False,
                 batch_norm=False, bn_eps=1e-5, relu=False, name='conv',
                 defaults=None, config=None, rules=None):
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError('Invalid Conv2d groups/channels')
        self.device, self.name = device, name
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = pair(kernel_size), pair(stride)
        self.padding, self.dilation = pair(padding), pair(dilation)
        self.groups, self.has_bias = groups, bool(bias)
        self.batch_norm, self.bn_eps, self.relu = batch_norm, bn_eps, relu
        self.defaults, self.config = dict(defaults or {}), dict(config or {})
        self.rules = rules
        self.weight = self.bias = self._backend = None

    @property
    def weight_shape(self):
        return (self.out_channels, self.in_channels // self.groups, *self.kernel_size)

    def spec(self):
        return dict(in_channels=self.in_channels, out_channels=self.out_channels,
                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups, bias=self.has_bias,
                    padding_mode='zeros', weight_shape=self.weight_shape,
                    batch_norm=(self.out_channels, self.bn_eps, True, True) if self.batch_norm else None,
                    activation='relu' if self.relu else None)

    def install_weights(self, weight, bias):
        self.weight, self.bias = weight, bias

        self._backend = None

    def __call__(self, x):
        def run(shape):
            import ttnn
            from layer_factory import MakeCNN, DEFAULT_CONV_ACTIVATION
            if self.weight is None:
                raise RuntimeError(f'{self.name}: load_torch_weights_into_ttnn must succeed first')
            if self.device is None:
                raise RuntimeError('Set a TTNN device before execution')
            if self._backend is None:
                activation = DEFAULT_CONV_ACTIVATION if self.relu else None

                for options in (self.defaults, self.config):
                    if 'activation' in options and options['activation'] != activation:
                        raise ValueError(f'{self.name}: activation config conflicts with model relu={self.relu}')
                self._backend = MakeCNN(
                    self.in_channels, self.out_channels, self.device, self.kernel_size,
                    self.stride, self.padding, dilation=self.dilation, groups=self.groups,
                    name=self.name, batch_size=x.shape[0], input_height=x.shape[2],
                    input_width=x.shape[3], defaults=self.defaults, overrides=self.config,
                    rules=self.rules, activation=activation,
                )
                self._backend.load_converted_weights(self.weight, self.bias)
            y, h, w = self._backend(x.data, x.shape[0], x.shape[2], x.shape[3])
            if (h, w) != shape[2:]:
                raise RuntimeError(f'{self.name}: runtime output shape mismatch')
            return y
        return apply(self.name, 'Conv2d', self.spec(), [x], run, self)

    def print_config(self):
        if self._backend is None:
            print(f'{self.name}: options resolve at the first actual input; config={self.config}')
        else:
            self._backend.print_config()


class TTMaxPool2d:
    def __init__(self, kernel_size=2, stride=None, padding=0, *, dilation=1,
                 ceil_mode=False, name='pool', defaults=None, config=None):
        self.name = name
        self.defaults, self.config = dict(defaults or {}), dict(config or {})
        self.attrs = dict(kernel_size=pair(kernel_size), stride=pair(kernel_size if stride is None else stride),
                          padding=pair(padding), dilation=pair(dilation), ceil_mode=ceil_mode,
                          return_indices=False)
        for options in (self.defaults, self.config):
            if 'ceil_mode' in options and options['ceil_mode'] != ceil_mode:
                raise ValueError(f'{name}: set structural ceil_mode on TTMaxPool2d directly')
        self._backend = None

    def __call__(self, x):
        def run(shape):
            from layer_factory import MakePool
            if self._backend is None:
                self._backend = MakePool(
                    'max', self.attrs['kernel_size'], self.attrs['stride'], self.attrs['padding'],
                    dilation=self.attrs['dilation'], channels=x.shape[1],
                    batch_size=x.shape[0], input_height=x.shape[2], input_width=x.shape[3],
                    name=self.name, defaults=self.defaults, overrides=self.config,
                    ceil_mode=self.attrs['ceil_mode'],
                )
            y, h, w = self._backend(x.data, x.shape[0], x.shape[2], x.shape[3])
            if (h, w) != shape[2:]:
                raise RuntimeError(f'{self.name}: runtime shape mismatch')
            return y
        return apply(self.name, 'MaxPool2d', self.attrs, [x], run, self)


class TTGlobalAvgPool2d:
    def __init__(self, name='avgpool', defaults=None, config=None):
        self.name, self.defaults, self.config = name, defaults, config
        self._backend = None

    def __call__(self, x):
        def run(shape):
            from layer_factory import MakePool
            if self._backend is None:
                self._backend = MakePool('global_avg', channels=x.shape[1],
                    batch_size=x.shape[0], input_height=x.shape[2], input_width=x.shape[3],
                    name=self.name, defaults=self.defaults, overrides=self.config)
            return self._backend(x.data, x.shape[0], x.shape[2], x.shape[3])[0]
        return apply(self.name, 'AdaptiveAvgPool2d', {'output_size': (1, 1)}, [x], run, self)


class TTFlatten:

    def __init__(self, name='flatten'):
        self.name = name

    def __call__(self, x):
        if len(x.shape) != 4 or x.shape[2:] != (1, 1):
            raise ValueError('TTFlatten currently supports global-pool output only')
        def run(shape):
            import ttnn
            y = ttnn.to_layout(x.data, ttnn.ROW_MAJOR_LAYOUT)
            y = ttnn.reshape(y, (1, 1, *shape))
            return ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        return apply(self.name, 'Flatten', {'start_dim': 1, 'end_dim': -1}, [x], run, self)


class TTLinear:
    def __init__(self, in_features, out_features, device=None, bias=True, name='fc'):
        self.in_features, self.out_features = in_features, out_features
        self.device, self.has_bias, self.name = device, bool(bias), name
        self.weight = self.bias = None

    def install_weights(self, weight, bias):
        self.weight, self.bias = weight, bias

    def __call__(self, x):
        attrs = dict(in_features=self.in_features, out_features=self.out_features,
                     bias=self.has_bias, weight_shape=(self.out_features, self.in_features))
        def run(shape):
            import ttnn
            from optimization_config import CLASSIFIER_DTYPE
            if self.weight is None:
                raise RuntimeError(f'{self.name}: weights not loaded')
            y = ttnn.to_memory_config(x.data, ttnn.DRAM_MEMORY_CONFIG)
            y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
            y = ttnn.typecast(y, CLASSIFIER_DTYPE) if y.dtype != CLASSIFIER_DTYPE else y
            return ttnn.linear(y, self.weight, bias=self.bias, transpose_b=False,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=CLASSIFIER_DTYPE)
        return apply(self.name, 'Linear', attrs, [x], run, self)


def tt_add(x, identity, name='add'):
    def run(shape):
        import ttnn
        a = ttnn.to_layout(ttnn.to_memory_config(x.data, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
        b = ttnn.to_layout(ttnn.to_memory_config(identity.data, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
        return ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return apply(name, 'Add', {}, [x, identity], run)


def tt_relu(x, name='relu'):
    def run(shape):
        import ttnn
        return ttnn.relu(x.data)
    return apply(name, 'ReLU', {}, [x], run)
