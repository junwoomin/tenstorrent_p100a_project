#pragma once






#include "device/device_operation.hpp"
#include "device/backward_device_operation.hpp"

namespace ttnn {


inline prim::BasicBlockReturn basic_block(
    const Tensor& x,
    const std::vector<Tensor>& weights,
    const std::vector<Tensor>& biases,
    uint32_t in_channels,
    uint32_t channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t kernel_size = 3,
    uint32_t stride = 1,
    uint32_t padding = 1,
    uint32_t max_cores = 0,
    uint32_t block_tiles = 32,
    uint32_t parameter_cache_tiles = 128
) {
    return prim::basic_block(
        x,
        weights,
        biases,
        in_channels,
        channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        max_cores,
        block_tiles,
        parameter_cache_tiles
    );
}



inline prim::BasicBlockBackwardReturn basic_block_backward(
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
    uint32_t kernel_size = 3,
    uint32_t stride = 1,
    uint32_t padding = 1
) {
    return prim::basic_block_backward(
        grad_out,
        x,
        weights,
        biases,
        in_channels,
        channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding
    );
}
}
