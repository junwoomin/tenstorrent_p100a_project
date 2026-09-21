from .tt_layers import TTConv2d
from .tt_tensor import apply


class TTNNStemConv(TTConv2d):
    def __init__(self, device, *, defaults=None, config=None):
        super().__init__(3, 64, device, 7, 2, 3, batch_norm=True,
                         relu=True, name='conv1', defaults=defaults, config=config)
        self.prepared_weights = {}
        self._configs = {}
        self.conv_config = None
        self.resolved_options = self.option_sources = None

    def install_weights(self, weight, bias):
        self.weight, self.bias = weight, bias
        self.prepared_weights.clear()
        self._configs.clear()
        self.conv_config = None
        self.resolved_options = self.option_sources = None

    def __call__(self, x):
        def run(shape):
            import ttnn
            from optimization_config import (
                ACTIVATION_DTYPE, ACTIVATION_LAYOUT, WEIGHT_DTYPE,
                CNN_DEFAULTS, DEFAULT_CONV_ACTIVATION, resolve_cnn_options,
            )

            if self.weight is None:
                raise RuntimeError(f'{self.name}: load_torch_weights_into_ttnn must succeed first')
            if self.device is None:
                raise RuntimeError('Set a TTNN device before execution')

            batch, _, height, width = x.shape
            key = (batch, height, width)
            if key not in self._configs:


                self.conv_config = ttnn.Conv2dConfig(
                    weights_dtype=WEIGHT_DTYPE,
                    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),

                    act_block_h_override=64,

                    enable_act_double_buffer=False,
                    enable_weights_double_buffer=False,
                    reshard_if_not_optimal=False,
                    deallocate_activation=True,

                    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    output_layout=ACTIVATION_LAYOUT,
                    config_tensors_in_dram=True,
                )

                stem_defaults = {
                    k: getattr(self.conv_config, k)
                    for k in CNN_DEFAULTS if k not in ('dtype', 'memory_config')
                }
                stem_defaults.update(dtype=ACTIVATION_DTYPE, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                stem_defaults.update(self.defaults)
                activation = DEFAULT_CONV_ACTIVATION if self.relu else None
                for options in (self.defaults, self.config):
                    if 'activation' in options and options['activation'] != activation:
                        raise ValueError(f'{self.name}: activation config conflicts with model relu={self.relu}')
                context = dict(name=self.name, batch_size=batch,
                               input_height=height, input_width=width,
                               in_channels=self.in_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation,
                               groups=self.groups, pool_type=None)
                resolved, sources = resolve_cnn_options(
                    context, defaults=stem_defaults, rules=self.rules,
                    overrides=self.config, activation=activation,
                )
                for name, value in resolved.items():
                    if name not in ('dtype', 'memory_config'):
                        setattr(self.conv_config, name, value)
                self._configs[key] = resolved, sources, self.conv_config
            resolved, sources, self.conv_config = self._configs[key]
            self.resolved_options, self.option_sources = dict(resolved), dict(sources)


            prepared = self.prepared_weights.get(key)
            result = ttnn.conv2d(
                input_tensor=x.data,
                weight_tensor=self.weight if prepared is None else prepared[0],
                bias_tensor=self.bias if prepared is None else prepared[1],
                device=self.device,
                in_channels=self.in_channels, out_channels=self.out_channels,
                batch_size=batch, input_height=height, input_width=width,
                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups,
                dtype=resolved['dtype'], memory_config=resolved['memory_config'],
                conv_config=self.conv_config, return_output_dim=True,
                return_weights_and_bias=prepared is None,
            )
            if prepared is None:
                y, output_dim, prepared = result
                self.prepared_weights[key] = prepared
            else:
                y, output_dim = result
            if tuple(output_dim) != shape[2:]:
                raise RuntimeError(f'{self.name}: runtime output shape mismatch')
            return y
        return apply(self.name, 'Conv2d', self.spec(), [x], run, self)

    def print_config(self):
        if self.resolved_options is None:
            print(f'{self.name}: options resolve at the first actual input; config={self.config}')
            return
        print(f'Layer: {self.name} (ttnn.conv2d)')
        for key, value in self.resolved_options.items():
            print(f'{key:28} = {value} [{self.option_sources[key]}]')
