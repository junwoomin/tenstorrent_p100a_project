from dataclasses import dataclass, field
from .torch_graph import analyze_torch
from models.tt_tensor import trace_model


@dataclass
class Match:
    torch: object
    ttnn: object
    reasons: list = field(default_factory=list)


@dataclass
class CompatibilityReport:
    architecture_compatible: bool = False
    weight_compatible: bool = False
    matches: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    torch_graph: object = None
    ttnn_graph: object = None
    fx_model: object = None
    input_shape: tuple = ()

    @property
    def compatible(self):
        return self.architecture_compatible and self.weight_compatible

    @property
    def weight_mapping(self):
        return {m.torch.name: m.ttnn.name for m in self.matches
                if m.torch.kind in ('Conv2d', 'Linear') and m.ttnn.kind == m.torch.kind}

    def __str__(self):
        lines = ['[Model Compatibility Check]', f"{'Torch':49} {'TTNN':49} Status", '-'*110]
        def label(op):
            a = op.attrs
            detail = ''
            if op.kind == 'Conv2d':
                detail = f" {a['in_channels']}->{a['out_channels']}"
                detail += '+BN' if a['batch_norm'] is not None else ''
                detail += '+ReLU' if a['activation'] == 'relu' else ''
            elif op.kind == 'Linear':
                detail = f" {a['in_features']}->{a['out_features']}"
            return f'{op.name} {op.kind}{detail}'
        for match in self.matches:
            lines.append(f"{label(match.torch):49} {label(match.ttnn):49} {'FAIL' if match.reasons else 'MATCH'}")
            lines.extend('  Reason: '+reason for reason in match.reasons)
        lines.extend('Reason: '+error for error in self.errors)
        lines += [f"Architecture Compatibility: {'PASS' if self.architecture_compatible else 'FAIL'}",
                  f"Weight Compatibility: {'PASS' if self.weight_compatible else 'FAIL'}",
                  f"Compatibility: {'PASS' if self.compatible else 'FAIL'}"]
        return '\n'.join(lines)


def compare_models(torch_model, ttnn_model, input_shape=None, mapping=None):

    if input_shape is None:
        size = getattr(ttnn_model, 'image_size', 224)
        input_shape = (getattr(ttnn_model, 'batch_size', None) or 1, 3, size, size)
    report = CompatibilityReport(input_shape=tuple(input_shape))
    try:
        report.torch_graph, report.fx_model = analyze_torch(torch_model, input_shape)
        report.ttnn_graph = trace_model(ttnn_model, input_shape)
    except Exception as error:
        report.errors.append(f'Graph analysis failed: {type(error).__name__}: {error}')
        return report
    source, target = report.torch_graph, report.ttnn_graph
    sa, ta = source.by_key, target.by_key
    s_to_t, t_to_s = {}, {}
    stack = [(source.output, target.output)]
    weight_ok = True
    seen_source_owners, seen_target_owners = {}, {}
    while stack:
        sk, tk = stack.pop()
        if sk in s_to_t or tk in t_to_s:
            if s_to_t.get(sk) != tk or t_to_s.get(tk) != sk:
                report.errors.append(f'Branch/reuse topology mismatch at {sa[sk].name} -> {ta[tk].name}')
            continue
        s_to_t[sk], t_to_s[tk] = tk, sk
        s, t = sa[sk], ta[tk]
        match = Match(s,t)
        report.matches.append(match)
        if s.kind != t.kind:
            match.reasons.append(f'layer type mismatch: {s.kind} != {t.kind}')
        for key in sorted(s.attrs.keys() | t.attrs.keys()):
            if s.attrs.get(key) != t.attrs.get(key):
                match.reasons.append(f'{key} mismatch: Torch {s.attrs.get(key)} / TTNN {t.attrs.get(key)}')
        if s.shape != t.shape:
            match.reasons.append(f'output shape mismatch: {s.shape} != {t.shape}')
        if len(s.inputs) != len(t.inputs):
            match.reasons.append('input edge count mismatch')
        stack.extend(reversed(list(zip(s.inputs,t.inputs))))
        if s.kind in ('Conv2d','Linear') or t.kind in ('Conv2d','Linear'):
            if match.reasons:
                weight_ok = False
            if s.owner is not None and t.owner is not None:
                sid, tid = id(s.owner), id(t.owner)
                if (sid in seen_source_owners and seen_source_owners[sid] != tid or
                    tid in seen_target_owners and seen_target_owners[tid] != sid):
                    match.reasons.append('shared parameter ownership mismatch')
                    weight_ok = False
                seen_source_owners[sid], seen_target_owners[tid] = tid, sid

    if len(s_to_t) != len(source.ops) or len(t_to_s) != len(target.ops):
        report.errors.append('Unmatched or unused graph nodes; mapping is not a complete graph bijection')
    order = {op.key:i for i,op in enumerate(source.ops)}
    report.matches.sort(key=lambda m:order[m.torch.key])
    explicit = dict(mapping or {})
    if len(set(explicit.values())) != len(explicit):
        report.errors.append('Explicit mapping has duplicate TTNN destinations')
    actual = report.weight_mapping
    for name, target_name in explicit.items():
        if name not in actual:
            report.errors.append(f'Explicit mapping source is unknown or unpaired: {name}')
        elif actual[name] != target_name:
            report.errors.append(f'Explicit mapping conflicts with graph topology: {name} -> {target_name}; graph gives {actual[name]}')
    report.architecture_compatible = not report.errors and all(not m.reasons for m in report.matches)
    report.weight_compatible = weight_ok and not report.errors
    return report
