import copy
import operator
import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from .graph import Graph, pair


def analyze_torch(model, input_shape):
    if any(module.training for module in model.modules()):
        raise ValueError('Architecture comparison requires torch_model.eval() (BN/Dropout inference semantics)')
    gm = symbolic_trace(model)

    meta = copy.deepcopy(gm).to(device='meta')
    ShapeProp(meta).propagate(torch.empty(input_shape, device='meta'))
    for node, meta_node in zip(gm.graph.nodes, meta.graph.nodes):
        node.meta = meta_node.meta
    del meta
    graph, env = Graph(), {}
    by_key = {}

    def emit(node, kind, attrs, owner=None):
        args = list(node.all_input_nodes)
        if any(arg not in env for arg in args):
            raise ValueError(f'{node.name}: unsupported tensor dependency')
        if 'tensor_meta' not in node.meta:
            raise ValueError(f'{node.name}: only a single tensor output is supported')
        name = str(node.target) if node.op == 'call_module' else node.name
        key = graph.add(name, kind, attrs, [env[arg] for arg in args], node.meta['tensor_meta'].shape, owner)
        op = graph.ops[-1]
        op.endpoint = node.name
        env[node], by_key[key] = key, op
        return op

    def previous(node):
        if len(node.all_input_nodes) != 1:
            return None
        inp = node.all_input_nodes[0]
        op = by_key[env[inp]]

        if len(inp.users) != 1 or op.endpoint != inp.name:
            return None
        return op

    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            emit(node, 'Input', {})
        elif node.op == 'output':
            if not isinstance(node.args[0], torch.fx.Node):
                raise ValueError('Only one tensor output is supported')
            graph.output = env[node.args[0]]
        elif node.op == 'call_module':
            module = gm.get_submodule(node.target)
            if type(module) is nn.Conv2d:
                emit(node, 'Conv2d', dict(
                    in_channels=module.in_channels, out_channels=module.out_channels,
                    kernel_size=pair(module.kernel_size), stride=pair(module.stride),
                    padding=pair(module.padding), dilation=pair(module.dilation), groups=module.groups,
                    bias=module.bias is not None, padding_mode=module.padding_mode,
                    weight_shape=tuple(module.weight.shape), batch_norm=None, activation=None), module)
            elif type(module) is nn.BatchNorm2d:
                op = previous(node)
                if (op is None or op.kind != 'Conv2d' or op.attrs['batch_norm'] is not None
                        or op.attrs['activation'] is not None):
                    raise ValueError(f'{node.target}: BN must follow a single-consumer Conv for supported fusion')
                if not module.track_running_stats or module.running_mean is None or module.running_var is None:
                    raise ValueError(f'{node.target}: BN requires fixed running statistics')
                op.attrs['batch_norm'] = (module.num_features, module.eps, module.affine, module.track_running_stats)
                op.bn, op.endpoint, env[node] = module, node.name, op.key
                op.shape = tuple(node.meta['tensor_meta'].shape)
            elif type(module) is nn.ReLU:
                if module.inplace and len(node.all_input_nodes[0].users) > 1:
                    raise ValueError(f'{node.target}: inplace ReLU on a branched value is unsupported')
                op = previous(node)
                if op is not None and op.kind == 'Conv2d' and op.attrs['activation'] is None:
                    op.attrs['activation'] = 'relu'
                    op.endpoint, env[node] = node.name, op.key
                else:
                    emit(node, 'ReLU', {})
            elif type(module) is nn.MaxPool2d:
                emit(node, 'MaxPool2d', dict(kernel_size=pair(module.kernel_size),
                    stride=pair(module.kernel_size if module.stride is None else module.stride),
                    padding=pair(module.padding), dilation=pair(module.dilation),
                    ceil_mode=module.ceil_mode, return_indices=module.return_indices))
            elif type(module) is nn.AdaptiveAvgPool2d:
                emit(node, 'AdaptiveAvgPool2d', {'output_size': pair(module.output_size)})
            elif type(module) is nn.Linear:
                emit(node, 'Linear', dict(in_features=module.in_features, out_features=module.out_features,
                    bias=module.bias is not None, weight_shape=tuple(module.weight.shape)), module)
            elif type(module) is nn.Flatten:
                emit(node, 'Flatten', dict(start_dim=module.start_dim, end_dim=module.end_dim))
            elif type(module) in (nn.Identity, nn.Dropout):

                env[node] = env[node.all_input_nodes[0]]
            else:
                raise ValueError(f'{node.target}: unsupported module {type(module).__name__}')
        elif node.op in ('call_function', 'call_method'):
            target = node.target
            if target in (operator.add, torch.add, 'add', '__add__'):
                if len(node.all_input_nodes) != 2 or node.kwargs.get('alpha', 1) != 1 or 'out' in node.kwargs:
                    raise ValueError(f'{node.name}: only tensor + tensor with alpha=1 is supported')
                emit(node, 'Add', {})
            elif target in (torch.flatten, 'flatten'):
                emit(node, 'Flatten', dict(start_dim=node.args[1] if len(node.args)>1 else node.kwargs.get('start_dim',0),
                    end_dim=node.args[2] if len(node.args)>2 else node.kwargs.get('end_dim',-1)))
            elif target in (torch.relu, torch.nn.functional.relu, 'relu'):
                inplace = node.kwargs.get('inplace', node.args[1] if len(node.args)>1 else False)
                if inplace and len(node.all_input_nodes[0].users) > 1:
                    raise ValueError(f'{node.name}: inplace ReLU on a branched value is unsupported')
                op = previous(node)
                if op is not None and op.kind == 'Conv2d' and op.attrs['activation'] is None:
                    op.attrs['activation'] = 'relu'
                    op.endpoint, env[node] = node.name, op.key
                else:
                    emit(node, 'ReLU', {})
            else:
                raise ValueError(f'{node.name}: unsupported operation {target}; no silent omission')
        else:
            raise ValueError(f'{node.name}: unsupported FX node type {node.op}')
    if len([op for op in graph.ops if op.kind == 'Input']) != 1:
        raise ValueError('Only single-input models are supported')
    return graph, gm
