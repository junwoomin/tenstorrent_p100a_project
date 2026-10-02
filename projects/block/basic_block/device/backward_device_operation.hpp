#pragma once
#include "device_operation.hpp"

namespace ttnn::prim {
// Explicit device backward: returns (dX, dW1/dW2/dW3[/dWd], db1/db2/db3[/dbd]).
// Original X/weights/biases must be unchanged since forward. Recomputes A1/A2;
// uses DRAM stage workspaces, BF16 tensors and FP32 tile accumulation. Bias
// gradients are channel-vector sums replicated across the 32-row bias tile.
using BasicBlockBackwardReturn = std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>;
BasicBlockBackwardReturn basic_block_backward(const Tensor &grad_out,
                                              const Tensor &x,
                                              const std::vector<Tensor> &weights,
                                              const std::vector<Tensor> &biases,
                                              uint32_t in_channels,
                                              uint32_t channels,
                                              uint32_t out_channels,
                                              uint32_t batch_size,
                                              uint32_t input_height,
                                              uint32_t input_width,
                                              uint32_t kernel_size,
                                              uint32_t stride,
                                              uint32_t padding);
} // namespace ttnn::prim
