import torch
import torch.nn.functional as F


def evaluate(graph, converted, x):
    weights = {id(item.target):item for item in converted}
    values = {}
    for op in graph.ops:
        a = op.attrs
        inputs = [values[key] for key in op.inputs]
        if op.kind == 'Input':
            y = x
        elif op.kind == 'Conv2d':
            item = weights[id(op.owner)]
            y = F.conv2d(inputs[0],item.weight.float(), None if item.bias is None else item.bias.flatten().float(),
                         a['stride'],a['padding'],a['dilation'],a['groups'])
            if a['activation'] == 'relu':
                y = F.relu(y)
        elif op.kind == 'MaxPool2d':
            y = F.max_pool2d(inputs[0],a['kernel_size'],a['stride'],a['padding'],a['dilation'],a['ceil_mode'])
        elif op.kind == 'AdaptiveAvgPool2d':
            y = F.adaptive_avg_pool2d(inputs[0],a['output_size'])
        elif op.kind == 'Flatten':
            y = torch.flatten(inputs[0],a['start_dim'],a['end_dim'])
        elif op.kind == 'Linear':
            item = weights[id(op.owner)]
            y = inputs[0] @ item.weight.float()
            if item.bias is not None:
                y = y + item.bias.flatten().float()
        elif op.kind == 'Add':
            y = inputs[0] + inputs[1]
        elif op.kind == 'ReLU':
            y = F.relu(inputs[0])
        else:
            raise ValueError(op.kind)
        values[op.key] = y
    return values[graph.output]
