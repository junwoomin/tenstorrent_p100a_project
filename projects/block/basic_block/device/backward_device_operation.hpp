#pragma once






#include "device_operation.hpp"

namespace ttnn::prim {
using BasicBlockBackwardReturn = std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>;
BasicBlockBackwardReturn basic_block_backward(
    const Tensor& grad_out,
    const Tensor& x,
    const std::vector<Tensor>& weights,
    const std::vector<Tensor>& biases,
    uint32_t in_channels,
    uint32_t channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t padding
);
}
