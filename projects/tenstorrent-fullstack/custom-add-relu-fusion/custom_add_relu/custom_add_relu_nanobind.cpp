#include "custom_add_relu_nanobind.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "custom_add_relu.hpp"

namespace ttnn::operations::custom_add_relu {
namespace nb = nanobind;

void bind_custom_add_relu(nb::module_& mod) {
    ttnn::bind_function<"custom_add_relu">(
        mod,
        R"doc(Compute relu(a + b) with a custom fused device program.
Inputs must have identical rank-4 shapes, BFLOAT16 dtype, standard 32x32
TILE layout without padding, and interleaved DRAM memory.)doc",
        &ttnn::custom_add_relu,
        nb::arg("a"),
        nb::arg("b"));

    ttnn::bind_function<"custom_add_relu_backward">(
        mod,
        R"doc(Return [grad_a, grad_b] for the saved custom_add_relu output.
Each gradient is grad_out * (saved_output > 0), with derivative zero at zero.
Finite BF16 tensors only. The two outputs have independent buffers.
No broadcasting. This manual TTNN API does not register an autograd node.)doc",
        &ttnn::custom_add_relu_backward,
        nb::arg("grad_out"),
        nb::arg("saved_output"));
}
}  // namespace ttnn::operations::custom_add_relu
