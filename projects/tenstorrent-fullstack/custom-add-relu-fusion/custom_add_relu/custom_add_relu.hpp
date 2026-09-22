#pragma once

#include "device/device_operation.hpp"
#include "device/backward_device_operation.hpp"

namespace ttnn {
inline Tensor custom_add_relu(const Tensor& a, const Tensor& b) {
    return prim::custom_add_relu(a, b);
}

inline std::vector<Tensor> custom_add_relu_backward(
    const Tensor& grad_out, const Tensor& saved_output) {
    return prim::custom_add_relu_backward(grad_out, saved_output);
}
}  // namespace ttnn
