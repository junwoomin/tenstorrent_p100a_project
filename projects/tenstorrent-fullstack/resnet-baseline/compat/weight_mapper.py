from dataclasses import dataclass
import torch
from dtype_config import MODEL_TORCH_DTYPE
from .model_compare import compare_models


@dataclass
class ConvertedWeight:
    name: str
    target: object
    weight: torch.Tensor
    bias: object
    kind: str


def require_shape(tensor, shape, name):

    if tensor is None or tuple(tensor.shape) != tuple(shape):
        got = None if tensor is None else tuple(tensor.shape)
        raise ValueError(f'{name}: expected shape {tuple(shape)}, got {got}')
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
        raise ValueError(f'{name}: weight must be floating-point and finite')


def convert_conv_weight(weight, expected_shape):
    require_shape(weight, expected_shape, 'Conv weight before conversion')

    converted = weight.detach().cpu().to(MODEL_TORCH_DTYPE).contiguous()
    require_shape(converted, expected_shape, 'Conv weight after conversion')
    return converted


def convert_linear_weight(weight, in_features, out_features):
    require_shape(weight, (out_features, in_features), 'Linear weight before conversion')


    converted = weight.detach().cpu().transpose(0,1).to(torch.bfloat16).contiguous()
    require_shape(converted, (in_features,out_features), 'Linear weight after conversion')
    return converted


def convert_bias(bias, out_features, *, linear=False):
    if bias is None:
        return None
    require_shape(bias, (out_features,), 'Bias before conversion')

    converted = bias.detach().cpu().to(torch.bfloat16 if linear else MODEL_TORCH_DTYPE)
    converted = converted.reshape(1,1,1,out_features).contiguous()
    require_shape(converted, (1,1,1,out_features), 'Bias after conversion')
    return converted


def _fuse_conv_bn(conv, bn):

    weight = conv.weight.detach().cpu().float()
    bias = torch.zeros(conv.out_channels) if conv.bias is None else conv.bias.detach().cpu().float()
    if bn is None:
        return weight, None if conv.bias is None else bias
    if bn.training or not bn.track_running_stats:
        raise ValueError('BN folding requires eval mode and running statistics')
    channels = conv.out_channels
    for name in ('running_mean','running_var'):
        require_shape(getattr(bn,name), (channels,), 'BN '+name)
    if bn.affine:
        require_shape(bn.weight, (channels,), 'BN weight')
        require_shape(bn.bias, (channels,), 'BN bias')
    if not torch.all(bn.running_var.detach().cpu().float() + bn.eps > 0).item():
        raise ValueError('BN running_var + eps must be positive')
    gamma = bn.weight.detach().cpu().float() if bn.affine else torch.ones(channels)
    beta = bn.bias.detach().cpu().float() if bn.affine else torch.zeros(channels)
    scale = gamma / torch.sqrt(bn.running_var.detach().cpu().float() + bn.eps)


    return weight * scale.reshape(-1,1,1,1), beta + (bias-bn.running_mean.detach().cpu().float())*scale


def prepare_torch_weights(torch_model, ttnn_model, *, input_shape=None, mapping=None):

    report = compare_models(torch_model, ttnn_model, input_shape, mapping)
    if not report.compatible:
        raise RuntimeError(str(report))
    pairs = [m for m in report.matches if m.torch.kind in ('Conv2d','Linear')]

    for match in pairs:
        source, target = match.torch, match.ttnn
        require_shape(source.owner.weight, target.attrs['weight_shape'], source.name+'.weight')
        expected_bias = source.attrs['out_channels'] if source.kind == 'Conv2d' else source.attrs['out_features']
        if source.attrs['bias']:
            require_shape(source.owner.bias, (expected_bias,), source.name+'.bias')
        elif source.owner.bias is not None:
            raise ValueError(f'{source.name}: unexpected bias')
        if source.bn is not None:
            for name in ('running_mean','running_var','weight','bias'):
                tensor = getattr(source.bn,name)
                if tensor is not None:
                    require_shape(tensor, (expected_bias,), source.name+'.BN.'+name)
    converted, seen = [], set()
    for match in pairs:
        source, target = match.torch, match.ttnn
        if id(target.owner) in seen:
            continue
        seen.add(id(target.owner))
        if source.kind == 'Conv2d':
            weight, bias = _fuse_conv_bn(source.owner, source.bn)
            weight = convert_conv_weight(weight, target.attrs['weight_shape'])
            bias = convert_bias(bias, target.attrs['out_channels'])
        else:
            weight = convert_linear_weight(source.owner.weight, target.attrs['in_features'], target.attrs['out_features'])
            bias = convert_bias(source.owner.bias, target.attrs['out_features'], linear=True)
        converted.append(ConvertedWeight(target.name, target.owner, weight, bias, target.kind))
    return report, converted


def load_torch_weights_into_ttnn(torch_model, ttnn_model, *, input_shape=None, mapping=None):

    report, converted = prepare_torch_weights(torch_model, ttnn_model, input_shape=input_shape, mapping=mapping)
    import ttnn
    from optimization_config import HOST_DTYPE, CLASSIFIER_DTYPE
    pending = []
    for item in converted:
        if item.kind == 'Conv2d':

            options = dict(dtype=HOST_DTYPE, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            if item.target.device is None:
                raise ValueError(f'{item.name}: Linear weight upload requires a device')
            options = dict(dtype=CLASSIFIER_DTYPE, layout=ttnn.TILE_LAYOUT,
                           device=item.target.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        weight = ttnn.from_torch(item.weight, **options)
        bias = None if item.bias is None else ttnn.from_torch(item.bias, **options)
        if tuple(weight.shape) != tuple(item.weight.shape):
            raise RuntimeError(f'{item.name}: TTNN logical weight shape changed unexpectedly')
        if bias is not None and tuple(bias.shape) != tuple(item.bias.shape):
            raise RuntimeError(f'{item.name}: TTNN logical bias shape changed unexpectedly')
        pending.append((item.target,weight,bias))

    for target, weight, bias in pending:
        target.install_weights(weight,bias)
    return report
