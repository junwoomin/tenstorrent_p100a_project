import ttnn

from optimization_config import (
    CNN_DEFAULTS, DEFAULT_CONV_ACTIVATION, resolve_cnn_options, resolve_pool_options,
)


def apply_conv_options(config, in_channels, out_channels, kernel, stride, padding,
                       *, image_size, batch_size, defaults=None, overrides=None):
    activation = DEFAULT_CONV_ACTIVATION if config.activation is not None else None
    for options in (defaults or {}, overrides or {}):
        if 'activation' in options and options['activation'] != activation:
            raise ValueError('activation conflicts with the ResNet block')
    baseline = {key: getattr(config, key) for key in CNN_DEFAULTS
                if key not in ('dtype', 'memory_config')}
    baseline.update(defaults or {})
    context = dict(
        batch_size=batch_size, input_height=image_size, input_width=image_size,
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=(kernel, kernel), stride=(stride, stride), padding=(padding, padding),
        dilation=(1, 1), groups=1, pool_type=None,
    )
    options, _ = resolve_cnn_options(
        context, defaults=baseline, overrides=overrides, activation=activation,
    )
    dtype = options.pop('dtype')
    memory_config = options.pop('memory_config')
    for key, value in options.items():
        setattr(config, key, value)
    return dtype, memory_config


def pool_options(kind, image_size, channels, batch_size, defaults=None, overrides=None):
    context = dict(
        name='pool1' if kind == 'max' else 'avgpool',
        batch_size=batch_size, input_height=image_size, input_width=image_size,
        in_channels=channels, out_channels=channels, channels=channels,
        kernel_size=(3, 3) if kind == 'max' else (image_size, image_size),
        stride=(2, 2) if kind == 'max' else (1, 1),
        padding=(1, 1) if kind == 'max' else (0, 0),
        dilation=(1, 1), groups=1, pool_type=kind,
    )
    if kind == 'max':
        for options in (defaults or {}, overrides or {}):
            if options.get('ceil_mode', False):
                raise ValueError('pool1: ceil_mode must remain False for this ResNet')
    resolved, _ = resolve_pool_options(context, defaults=defaults, overrides=overrides)
    return resolved
