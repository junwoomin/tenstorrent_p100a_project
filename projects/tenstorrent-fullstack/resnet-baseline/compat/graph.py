from dataclasses import dataclass, field
from math import prod


@dataclass
class Op:
    key: str
    name: str
    kind: str
    attrs: dict
    inputs: tuple
    shape: tuple
    owner: object = None
    bn: object = None
    endpoint: str = ''


@dataclass
class Graph:
    ops: list = field(default_factory=list)
    output: str = ''

    def add(self, name, kind, attrs, inputs, shape, owner=None):
        key = f'n{len(self.ops)}'
        self.ops.append(Op(key, name, kind, dict(attrs), tuple(inputs), tuple(shape), owner))
        self.output = key
        return key

    @property
    def by_key(self):
        return {op.key: op for op in self.ops}


def pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def output_shape(kind, a, shapes):
    x = tuple(shapes[0])
    if kind in ('ReLU', 'Dropout', 'Identity'):
        return x
    if kind == 'Add':
        if x != tuple(shapes[1]):
            raise ValueError(f'Residual Add shape mismatch: {shapes}')
        return x
    if kind in ('Conv2d', 'MaxPool2d'):
        if len(x) != 4:
            raise ValueError(f'{kind} requires logical NCHW input, got {x}')
        if kind == 'Conv2d' and x[1] != a['in_channels']:
            raise ValueError(f"Conv2d in_channels mismatch: input {x[1]}, layer {a['in_channels']}")
        hw = []
        for size, k, s, p, d in zip(x[2:], a['kernel_size'], a['stride'], a['padding'], a['dilation']):
            ceil = a.get('ceil_mode', False)
            dim = (size + 2*p - d*(k-1) - 1 + (s-1 if ceil else 0)) // s + 1
            if ceil and (dim-1)*s >= size+p:
                dim -= 1
            if dim <= 0:
                raise ValueError(f'{kind}: non-positive output dimension')
            hw.append(dim)
        return (x[0], a['out_channels'] if kind == 'Conv2d' else x[1], *hw)
    if kind == 'AdaptiveAvgPool2d':
        return (x[0], x[1], *a['output_size'])
    if kind == 'Flatten':
        start, end = a['start_dim'], a['end_dim']
        start, end = start % len(x), end % len(x)
        if start > end:
            raise ValueError('Invalid Flatten dimensions')
        return (*x[:start], prod(x[start:end+1]), *x[end+1:])
    if kind == 'Linear':
        if x[-1] != a['in_features']:
            raise ValueError(f"Linear in_features mismatch: {x[-1]} != {a['in_features']}")
        return (*x[:-1], a['out_features'])
    raise ValueError(f'Unsupported operation: {kind}')
