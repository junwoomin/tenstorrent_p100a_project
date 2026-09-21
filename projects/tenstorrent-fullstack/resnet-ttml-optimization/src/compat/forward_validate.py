from dataclasses import dataclass
import torch
from .model_compare import compare_models
from models.tt_tensor import observer


@dataclass
class ValidationRow:
    name: str
    torch_shape: tuple
    ttnn_shape: tuple
    max_abs_error: float = float('nan')
    mean_abs_error: float = float('nan')
    cosine_similarity: float = float('nan')
    numeric_match: bool = False


@dataclass
class ForwardReport:
    rows: list

    @property
    def shape_compatible(self):
        return all(row.torch_shape == row.ttnn_shape for row in self.rows)

    @property
    def numerical_compatible(self):
        return all(row.numeric_match for row in self.rows)

    def __str__(self):
        lines = ['[Forward Validation]', 'Logical shape = PyTorch NCHW / N,C; errors after layout normalization']
        for row in self.rows:
            status = 'MATCH' if row.torch_shape == row.ttnn_shape else 'FAIL'
            lines.append(f'{row.name:34} {str(row.torch_shape):22} {str(row.ttnn_shape):22} {status} '
                         f'max={row.max_abs_error:.6g} mean={row.mean_abs_error:.6g} '
                         f'cos={row.cosine_similarity:.6g} numeric={"PASS" if row.numeric_match else "FAIL"}')
        lines.append(f'Shape Compatibility: {"PASS" if self.shape_compatible else "FAIL"}')
        lines.append(f'Numerical Compatibility: {"PASS" if self.numerical_compatible else "FAIL"}')
        return '\n'.join(lines)


def to_logical_torch(tensor, shape):

    import ttnn
    host = ttnn.to_torch(ttnn.from_device(tensor)).float()
    raw = tuple(host.shape)
    if len(shape) == 4:
        n,c,h,w = shape
        if raw not in ((n,h,w,c),(1,1,n*h*w,c)):
            raise ValueError(f'TTNN logical activation shape mismatch: got {raw}, expected NHWC {(n,h,w,c)} or flattened NHW')
        return host.reshape(n,h,w,c).permute(0,3,1,2).contiguous()
    if len(shape) == 2:
        if raw not in (tuple(shape),(1,1,*shape)):
            raise ValueError(f'TTNN logical output shape mismatch: {raw} vs {shape}')
        return host.reshape(shape)
    raise ValueError(f'Unsupported validation rank: {shape}')


def _row(name, expected, actual, atol, rtol):
    row = ValidationRow(name,tuple(expected.shape),tuple(actual.shape))
    if row.torch_shape != row.ttnn_shape:
        return row
    a,b = expected.detach().cpu().float().flatten(), actual.detach().cpu().float().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return row
    difference = (a-b).abs()
    row.max_abs_error, row.mean_abs_error = difference.max().item(), difference.mean().item()
    norm = a.norm()*b.norm()
    row.cosine_similarity = ((a@b)/norm).item() if norm.item() else (1.0 if torch.equal(a,b) else 0.0)
    row.numeric_match = torch.allclose(a,b,atol=atol,rtol=rtol)
    return row


def validate_forward(torch_model, ttnn_model, dummy_input, *, atol=0.1, rtol=0.1, mapping=None):

    if dummy_input.device.type != 'cpu' or any(p.device.type != 'cpu' for p in torch_model.parameters()):
        raise ValueError('Forward validation expects the reference model and dummy input on CPU')
    report = compare_models(torch_model,ttnn_model,tuple(dummy_input.shape),mapping)
    if not report.compatible:
        raise RuntimeError(str(report))
    captured = {}
    endpoints = {m.torch.endpoint for m in report.matches}
    class Recorder(torch.fx.Interpreter):
        def run_node(self,node):
            value = super().run_node(node)
            if node.name in endpoints:

                captured[node.name] = value.detach().cpu().float().clone()
            return value
    with torch.no_grad():
        reference_output = Recorder(report.fx_model).run(dummy_input)
    import ttnn
    from optimization_config import ACTIVATION_DTYPE, ACTIVATION_LAYOUT
    device = ttnn_model.device
    device_input = ttnn.from_torch(dummy_input.permute(0,2,3,1).contiguous(),
        dtype=ACTIVATION_DTYPE, layout=ACTIVATION_LAYOUT, device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    matches = {match.ttnn.key:match for match in report.matches}
    rows = [_row('input',dummy_input,to_logical_torch(device_input,tuple(dummy_input.shape)),atol,rtol)]
    index = 1
    def capture(name, kind, data, shape):
        nonlocal index
        key = f'n{index}'
        index += 1
        match = matches[key]
        if match.ttnn.name != name or match.ttnn.kind != kind:
            raise RuntimeError('Runtime execution differs from the traced graph')
        ttnn.synchronize_device(device)
        actual = to_logical_torch(data,shape)
        rows.append(_row(name,captured[match.torch.endpoint],actual,atol,rtol))
    token = observer.set(capture)
    try:
        actual_output = ttnn_model(device_input)
        ttnn.synchronize_device(device)
    finally:
        observer.reset(token)
    if index != len(report.ttnn_graph.ops):
        raise RuntimeError('Runtime execution omitted traced operations')
    rows.append(_row('final output',reference_output,
        to_logical_torch(actual_output,report.ttnn_graph.by_key[report.ttnn_graph.output].shape),atol,rtol))
    return ForwardReport(rows)
