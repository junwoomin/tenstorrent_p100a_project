"""Installed inside ttnn; attach TTML autograd to the public TTNN name lazily.

Raw TTNN inputs keep the native forward behavior. TTML inputs track first-order
gradients. Importing TTNN does not import TTML (avoids circular imports).
"""
import importlib


def install(ttnn_module):
    raw_forward = getattr(ttnn_module, "custom_add_relu", None)
    if getattr(raw_forward, "__custom_add_relu_dispatch__", False):
        return True
    raw_backward = getattr(ttnn_module, "custom_add_relu_backward", None)
    if raw_forward is None or raw_backward is None:
        return False  # Keep import ttnn usable before the native op is built.
    function_class = None
    ttml_module = None

    def get_ttml():
        nonlocal function_class, ttml_module
        if ttml_module is not None:
            return ttml_module, function_class
        try:
            candidate = importlib.import_module("ttml")
        except ImportError as exc:
            raise ImportError(
                "TTML inputs require an installed TTML with autograd.Function. "
                "Two ttnn.Tensor inputs work without TTML."
            ) from exc
        autograd = getattr(candidate, "autograd", None)
        if autograd is None or not hasattr(autograd, "Function"):
            raise RuntimeError("This integration requires ttml.autograd.Function")

        class CustomAddReLU(autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                # Captured native callable, never re-enter the public dispatcher.
                output = raw_forward(a.get_value(), b.get_value())
                ctx.save_for_backward(output)
                ctx.needs_a = a.get_requires_grad()
                ctx.needs_b = b.get_requires_grad()
                return autograd.create_tensor(
                    output, requires_grad=ctx.needs_a or ctx.needs_b
                )

            @staticmethod
            def backward(ctx, grad_output):
                (saved_output,) = ctx.saved_tensors
                if grad_output.dtype != ttnn_module.bfloat16:
                    raise TypeError("custom_add_relu backward requires BF16 gradients")
                grad_output = ttnn_module.to_memory_config(
                    grad_output, ttnn_module.DRAM_MEMORY_CONFIG
                )
                if grad_output.layout != ttnn_module.TILE_LAYOUT:
                    grad_output = ttnn_module.to_layout(grad_output, ttnn_module.TILE_LAYOUT)
                grad_a, grad_b = raw_backward(grad_output, saved_output)
                return (
                    grad_a if ctx.needs_a else None,
                    grad_b if ctx.needs_b else None,
                )

        function_class = CustomAddReLU
        ttml_module = candidate
        return candidate, function_class

    def custom_add_relu(a, b):
        """Add+ReLU; TTML inputs enable autograd, raw TTNN inputs run forward.

        A raw input mixed with a TTML input is a constant (requires_grad=False).
        No broadcasting. Preserve the output until backward completes.
        """
        a_raw = isinstance(a, ttnn_module.Tensor)
        b_raw = isinstance(b, ttnn_module.Tensor)
        if a_raw and b_raw:
            return raw_forward(a, b)
        ttml, operation = get_ttml()
        autograd = ttml.autograd
        for name, tensor in (("a", a), ("b", b)):
            if not isinstance(tensor, (ttnn_module.Tensor, autograd.Tensor)):
                raise TypeError(f"{name} must be a ttnn.Tensor or ttml.autograd.Tensor")
        if a_raw:
            a = autograd.create_tensor(a, requires_grad=False)
        if b_raw:
            b = autograd.create_tensor(b, requires_grad=False)
        context = autograd.AutoContext.get_instance()
        enabled = context.get_gradient_mode() == autograd.GradMode.ENABLED
        if not enabled or not (a.get_requires_grad() or b.get_requires_grad()):
            output = raw_forward(a.get_value(), b.get_value())
            return autograd.create_tensor(output, requires_grad=False)
        return operation.apply(a, b)

    custom_add_relu.__module__ = "ttnn"
    custom_add_relu.__custom_add_relu_dispatch__ = True
    custom_add_relu.__wrapped__ = raw_forward
    ttnn_module.custom_add_relu = custom_add_relu
    return True
