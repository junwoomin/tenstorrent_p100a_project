import ttnn
from dtype_config import get_ttnn_dtype, get_ttnn_layout

ACTIVATION_DTYPE = get_ttnn_dtype(ttnn, "activation")
WEIGHT_DTYPE = get_ttnn_dtype(ttnn, "weight")
CLASSIFIER_DTYPE = get_ttnn_dtype(ttnn, "classifier")
HOST_DTYPE = get_ttnn_dtype(ttnn, "host")
ACTIVATION_LAYOUT = get_ttnn_layout(ttnn)
DEFAULT_CONV_ACTIVATION = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

CNN_DEFAULTS = {
    "weights_dtype": WEIGHT_DTYPE,
    "activation": DEFAULT_CONV_ACTIVATION,
    "act_block_h_override": 0,
    "enable_act_double_buffer": False,
    "enable_weights_double_buffer": False,
    "reshard_if_not_optimal": False,
    "deallocate_activation": False,
    "shard_layout": None,
    "output_layout": ACTIVATION_LAYOUT,
    "config_tensors_in_dram": True,
    "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    "dtype": ACTIVATION_DTYPE,
}

POOL_DEFAULTS = {
    "max": {
        "ceil_mode": False,
        "config_tensor_in_dram": True,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "applied_shard_scheme": None,
        "deallocate_input": False,
        "dtype": ACTIVATION_DTYPE,
        "output_layout": ACTIVATION_LAYOUT,
    },
    "global_avg": {
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "dtype": CLASSIFIER_DTYPE,
    },
}


def _vgg_conv(context, channels):
    return (
        (context["in_channels"], context["out_channels"]) in channels
        and context["kernel_size"] == (3, 3)
        and context["stride"] == (1, 1)
        and context["padding"] == (1, 1)
        and context["dilation"] == (1, 1)
        and context["groups"] == 1
    )


CNN_RULES = [
    {
        "name": "vgg11_conv1",
        "condition": lambda c: _vgg_conv(c, {(3, 64)}),
        "options": {
            "act_block_h_override": 64,
            "reshard_if_not_optimal": True,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        },
    },
    {
        "name": "vgg11_conv2_conv4",
        "condition": lambda c: _vgg_conv(c, {(64, 128), (256, 256)}),
        "options": {"shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED},
    },
]
POOL_RULES = []


def _normalize_options(options):
    result = dict(options or {})
    if "activation_dtype" in result:
        if "dtype" in result:
            raise ValueError("Use only one of dtype and activation_dtype")
        result["dtype"] = result.pop("activation_dtype")
    return result


def _resolve(context, baseline, rules, defaults, overrides, keywords):
    options = dict(baseline)
    sources = dict.fromkeys(options, "DEFAULT")

    def apply(values, source):
        values = _normalize_options(values)
        unknown = values.keys() - baseline.keys()
        if unknown:
            raise ValueError(f"Unsupported options: {', '.join(sorted(unknown))}")
        options.update(values)
        sources.update(dict.fromkeys(values, source))

    apply(defaults, "DEFAULT")
    explicit = _normalize_options(overrides)
    explicit.update(_normalize_options(keywords))
    context = dict(context, dtype=explicit.get("dtype", options["dtype"]))
    for rule in rules:
        if rule["condition"](context):
            apply(rule["options"], "AUTO")
    apply(explicit, "OVERRIDE")
    if options.get("dtype") == ttnn.bfloat8_b and options.get("output_layout") == ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("bfloat8_b requires output_layout=ttnn.TILE_LAYOUT")
    return options, sources


def resolve_cnn_options(context, *, defaults=None, rules=None, overrides=None, **options):
    return _resolve(context, CNN_DEFAULTS, CNN_RULES if rules is None else rules,
                    defaults, overrides, options)


def resolve_pool_options(context, *, defaults=None, rules=None, overrides=None, **options):
    return _resolve(context, POOL_DEFAULTS[context["pool_type"]],
                    POOL_RULES if rules is None else rules, defaults, overrides, options)
