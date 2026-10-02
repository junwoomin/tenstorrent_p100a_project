import torch


PRECISION = "bfloat16"


_PRECISION_PRESETS = {
    "bfloat16": {
        "torch": torch.bfloat16,
        "host": "bfloat16",
        "activation": "bfloat16",
        "weight": "bfloat16",
        "classifier": "bfloat16",
        "layout": "ROW_MAJOR_LAYOUT",
    },
    "bfloat8_b": {
        "torch": torch.bfloat16,
        "host": "bfloat16",
        "activation": "bfloat8_b",
        "weight": "bfloat8_b",
        "classifier": "bfloat16",
        "layout": "TILE_LAYOUT",
    },
    "float32": {
        "torch": torch.float32,
        "host": "float32",
        "activation": "float32",
        "weight": "float32",
        "classifier": "bfloat16",
        "layout": "ROW_MAJOR_LAYOUT",
    },
}

try:
    _SELECTED_PRESET = _PRECISION_PRESETS[PRECISION]
except KeyError as error:
    raise ValueError(f"지원하지 않는 PRECISION: {PRECISION}") from error

MODEL_TORCH_DTYPE = _SELECTED_PRESET["torch"]


def get_ttnn_dtype(ttnn_module, role):

    try:
        dtype_name = _SELECTED_PRESET[role]
    except KeyError as error:
        raise ValueError(f"지원하지 않는 dtype 역할: {role}") from error
    try:
        return getattr(ttnn_module, dtype_name)
    except AttributeError as error:
        raise ValueError(f"TTNN이 지원하지 않는 dtype: {dtype_name}") from error


def get_ttnn_layout(ttnn_module):

    return getattr(ttnn_module, _SELECTED_PRESET["layout"])
