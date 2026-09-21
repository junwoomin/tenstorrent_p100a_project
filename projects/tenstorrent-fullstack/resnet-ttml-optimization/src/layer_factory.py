from numbers import Integral
import ttnn
from optimization_config import (
    ACTIVATION_DTYPE, WEIGHT_DTYPE, CLASSIFIER_DTYPE, HOST_DTYPE,
    ACTIVATION_LAYOUT, DEFAULT_CONV_ACTIVATION, CNN_DEFAULTS, POOL_DEFAULTS,
    CNN_RULES, POOL_RULES, _normalize_options, resolve_cnn_options, resolve_pool_options,
)

def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _pair(value, name, minimum=1):
    if isinstance(value, Integral):
        value = (value, value)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an integer or a pair (height, width)")
    return tuple(_integer(v, name, minimum) for v in value)


def _output_dim(size, kernel, stride, padding, dilation, ceil_mode=False):
    numerator = size + 2 * padding - dilation * (kernel - 1) - 1
    output = (numerator + (stride - 1 if ceil_mode else 0)) // stride + 1
    if ceil_mode and (output - 1) * stride >= size + padding:
        output -= 1
    if output <= 0:
        raise ValueError("Kernel/padding produces a non-positive output size")
    return output


class _LayerInfo:


    @property
    def input_shape(self):
        return (self.batch_size, self.input_height, self.input_width, self.in_channels)

    @property
    def output_shape(self):
        return (self.batch_size, self.output_height, self.output_width, self.out_channels)

    @property
    def output_hw(self):
        return self.output_height, self.output_width

    def _set_shape(self, batch_size, height, width):


        self.batch_size = None if batch_size is None else _integer(batch_size, "batch_size")
        self.input_height = None if height is None else _integer(height, "input_height")
        self.input_width = None if width is None else _integer(width, "input_width")

    def _set_output(self, ceil_mode=False):
        dims = (self.input_height, self.input_width)
        self.output_height, self.output_width = (
            None if size is None else _output_dim(size, k, s, p, d, ceil_mode)
            for size, k, s, p, d in zip(dims, self.kernel_size, self.stride, self.padding, self.dilation)
        )

    def _runtime_shape(self, batch_size, height, width):

        values = tuple(given if given is not None else original
                       for given, original in zip((batch_size, height, width), self._declared_shape))
        if any(value is None for value in values):
            raise ValueError("Specify batch_size, input_height and input_width at construction or call time")
        return tuple(_integer(value, name) for value, name in
                     zip(values, ("batch_size", "input_height", "input_width")))

    def _context(self):


        return dict(name=self.name, batch_size=self.batch_size,
                    input_height=self.input_height, input_width=self.input_width,
                    in_channels=self.in_channels, out_channels=self.out_channels,
                    kernel_size=self.kernel_size, stride=self.stride,
                    padding=self.padding, dilation=self.dilation, groups=self.groups,
                    pool_type=getattr(self, "pool_type", None))

    def print_config(self):

        print(f"Layer: {self.name}")
        print(f"Input : {list(self.input_shape)}")
        print(f"Output: {list(self.output_shape)}")
        for key in ("kernel_size", "stride", "padding", "dilation", "groups"):
            print(f"{key} = {getattr(self, key)}")
        print("options (source):")
        for key, value in self.resolved_options.items():
            print(f"{key:28} = {value} [{self.option_sources[key]}]")


class MakeCNN(_LayerInfo):


    def __init__(self, in_channels, out_channels, device=None, kernel_size=3,
                 stride=1, padding=1, *, input_height=None, input_width=None,
                 batch_size=None, dilation=1, groups=1, name=None,
                 defaults=None, rules=None, overrides=None, weight=None, bias=None,
                 **options):
        self.name = name or "conv"
        self.device = device
        self.in_channels = _integer(in_channels, "in_channels")
        self.out_channels = _integer(out_channels, "out_channels")
        self.groups = _integer(groups, "groups")
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding", 0)
        self.dilation = _pair(dilation, "dilation")
        self._declared_shape = (batch_size, input_height, input_width)
        self._defaults = dict(defaults or {})
        self._overrides = _normalize_options(overrides)
        self._overrides.update(_normalize_options(options))

        self._rules = [dict(rule, options=dict(rule["options"]))
                       for rule in (CNN_RULES if rules is None else rules)]
        self._configs = {}
        self.prepared_weights = {}
        self._configure(*self._declared_shape)


        self.weight = self.bias = None
        if weight is not None or bias is not None:
            raise ValueError("Use compat.weight_mapper after compatibility PASS")

    def _configure(self, batch_size, height, width):
        self._set_shape(batch_size, height, width)
        self._set_output()
        key = (batch_size, height, width)
        if key not in self._configs:
            resolved, sources = resolve_cnn_options(
                self._context(), defaults=self._defaults, rules=self._rules,
                overrides=self._overrides,
            )

            config = ttnn.Conv2dConfig(**{k: v for k, v in resolved.items()
                                         if k not in ("dtype", "memory_config")})
            self._configs[key] = resolved, sources, config
        resolved, sources, self.conv_config = self._configs[key]
        self.resolved_options = dict(resolved)
        self.option_sources = dict(sources)
        self.common_args = dict(
            device=self.device, in_channels=self.in_channels, out_channels=self.out_channels,
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups, dtype=resolved["dtype"],
            conv_config=self.conv_config, memory_config=resolved["memory_config"],
            return_output_dim=True,
        )

    def load_converted_weights(self, weight, bias):

        self.weight, self.bias = weight, bias
        self.prepared_weights.clear()

    def __call__(self, x, batch_size=None, height=None, width=None):
        if self.weight is None:
            raise RuntimeError(f"{self.name}: weights not loaded; run the compatibility mapper first")
        legacy_result = any(v is not None for v in (batch_size, height, width))
        key = self._runtime_shape(batch_size, height, width)
        self._configure(*key)
        prepared = self.prepared_weights.get(key)
        result = ttnn.conv2d(
            input_tensor=x, weight_tensor=self.weight if prepared is None else prepared[0],
            bias_tensor=self.bias if prepared is None else prepared[1],
            batch_size=key[0], input_height=key[1], input_width=key[2],
            **self.common_args, return_weights_and_bias=prepared is None,
        )
        if prepared is None:
            x, output_dim, prepared = result
            self.prepared_weights[key] = prepared
        else:
            x, output_dim = result
        if tuple(output_dim) != self.output_hw:
            raise RuntimeError(f"{self.name}: predicted {self.output_hw}, TTNN returned {output_dim}")
        return (x, *self.output_hw) if legacy_result else x


class MakePool(_LayerInfo):


    def __init__(self, pool_type="max", kernel_size=None, stride=None, padding=0, *,
                 input_height, input_width, channels, batch_size=None, device=None,
                 dilation=1, name=None, defaults=None, rules=None, overrides=None, **options):
        if pool_type not in POOL_DEFAULTS:
            raise ValueError("pool_type must be 'max' or 'global_avg'")
        self.pool_type = pool_type
        self.name = name or pool_type + "_pool"
        self.device = device
        self.channels = _integer(channels, "channels")
        self.in_channels = self.out_channels = self.channels
        self.groups = 1
        self.padding = _pair(padding, "padding", 0)
        self.dilation = _pair(dilation, "dilation")
        if pool_type == "global_avg":
            if (kernel_size is not None or stride is not None or
                    self.padding != (0, 0) or self.dilation != (1, 1)):
                raise ValueError("global_avg determines its own kernel/stride and requires padding=0, dilation=1")
            self.kernel_size = (input_height, input_width)
            self.stride = (1, 1)
        else:
            self.kernel_size = _pair(2 if kernel_size is None else kernel_size, "kernel_size")
            self.stride = _pair(self.kernel_size if stride is None else stride, "stride")
        self._declared_shape = (batch_size, input_height, input_width)
        self._defaults = dict(defaults or {})
        self._overrides = _normalize_options(overrides)
        self._overrides.update(_normalize_options(options))
        self._rules = [dict(rule, options=dict(rule["options"]))
                       for rule in (POOL_RULES if rules is None else rules)]
        self._configs = {}
        self._configure(*self._declared_shape)

    def _configure(self, batch_size, height, width):
        self._set_shape(batch_size, height, width)
        if self.pool_type == "global_avg":
            self.kernel_size = (height, width)
        key = (batch_size, height, width)
        if key not in self._configs:
            self._configs[key] = resolve_pool_options(
                dict(self._context(), channels=self.channels), defaults=self._defaults,
                rules=self._rules, overrides=self._overrides,
            )
        resolved, sources = self._configs[key]
        self.resolved_options, self.option_sources = dict(resolved), dict(sources)
        if self.pool_type == "global_avg":
            self.output_height = self.output_width = 1
        else:
            self._set_output(ceil_mode=resolved["ceil_mode"])

    def __call__(self, x, batch_size=None, height=None, width=None):
        legacy_result = any(v is not None for v in (batch_size, height, width))
        self._configure(*self._runtime_shape(batch_size, height, width))
        if self.pool_type == "max":
            x = ttnn.max_pool2d(
                input_tensor=x, batch_size=self.batch_size,
                input_h=self.input_height, input_w=self.input_width, channels=self.channels,
                kernel_size=list(self.kernel_size), stride=list(self.stride),
                padding=list(self.padding), dilation=list(self.dilation), **self.resolved_options,
            )
        else:


            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            target_dtype = self.resolved_options["dtype"]
            if target_dtype is None:
                target_dtype = ttnn.bfloat16
            if x.dtype != target_dtype:
                x = ttnn.typecast(x, target_dtype)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, self.input_shape)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.global_avg_pool2d(x, **self.resolved_options)
        return (x, *self.output_hw) if legacy_result else x
