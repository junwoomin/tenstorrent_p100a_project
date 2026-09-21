from contextvars import ContextVar
from dataclasses import dataclass
from compat.graph import Graph, output_shape


observer = ContextVar('ttnn_validation_observer', default=None)


@dataclass
class Value:
    data: object
    shape: tuple
    graph: object = None
    key: str = ''


def input_value(x, image_size):
    if isinstance(x, Value):
        if len(x.shape) != 4 or x.shape[1:] != (3, image_size, image_size):
            raise ValueError(f'Input shape mismatch: model expects N,3,{image_size},{image_size}, got {x.shape}')
        return x
    shape = tuple(x.shape)
    if len(shape) != 4 or shape[1:] != (image_size, image_size, 3):
        raise ValueError(f'Expected NHWC [N,{image_size},{image_size},3], got {shape}')
    return Value(x, (shape[0], 3, image_size, image_size))


def result_value(x):
    return x if x.graph is not None else x.data


def apply(name, kind, attrs, inputs, run, owner=None):
    shape = output_shape(kind, attrs, [x.shape for x in inputs])
    graph = inputs[0].graph
    if graph is not None:
        if any(x.graph is not graph for x in inputs):
            raise ValueError('Inputs belong to different graphs')
        key = graph.add(name, kind, attrs, [x.key for x in inputs], shape, owner)
        return Value(None, shape, graph, key)
    data = run(shape)
    hook = observer.get()
    if hook is not None:
        hook(name, kind, data, shape)
    return Value(data, shape)


def trace_model(model, input_shape):
    graph = Graph()
    key = graph.add('input', 'Input', {}, (), input_shape)
    x = model(Value(None, tuple(input_shape), graph, key))
    if not isinstance(x, Value) or x.graph is not graph:
        raise ValueError('TTNN model must return a single traced Value')
    graph.output = x.key
    return graph
