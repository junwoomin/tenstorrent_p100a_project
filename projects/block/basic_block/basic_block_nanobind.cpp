



#include "basic_block_nanobind.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "basic_block.hpp"

namespace ttnn::operations::basic_block {
namespace nb = nanobind;



void bind_basic_block(nb::module_& mod) {
    ttnn::bind_function<"basic_block">(
        mod,
        "BF16 bottleneck (1x1 -> 3x3 -> 1x1), without BN. L1-only tiled inference; no intermediate "
        "DRAM. Packed NHWC "
        "input; returns (out, weights, biases).",
        &ttnn::basic_block,
        nb::arg("x"),
        nb::arg("weights"),
        nb::arg("biases"),
        nb::arg("in_channels"),
        nb::arg("channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size") = 3,
        nb::arg("stride") = 1,
        nb::arg("padding") = 1,
        nb::arg("max_cores") = 0,
        nb::arg("block_tiles") = 32,
        nb::arg("parameter_cache_tiles") = 128
    );
    ttnn::bind_function<"basic_block_backward">(
        mod,
        "Device backward with recomputed activations; returns (dx, dweights, dbiases). BF16 "
        "tensors, FP32 accumulation. Uses DRAM workspaces.",
        &ttnn::basic_block_backward,
        nb::arg("grad_out"),
        nb::arg("x"),
        nb::arg("weights"),
        nb::arg("biases"),
        nb::arg("in_channels"),
        nb::arg("channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size") = 3,
        nb::arg("stride") = 1,
        nb::arg("padding") = 1
    );
}
}
