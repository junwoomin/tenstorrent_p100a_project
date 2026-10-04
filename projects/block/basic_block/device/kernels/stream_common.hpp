#pragma once






#include <cstdint>
#include "block_geometry.hpp"

namespace bottleneck_stream {
using namespace bottleneck_geometry;
constexpr uint32_t tile_bytes = 2048;
constexpr uint32_t stride = get_named_compile_time_arg_val("stride");

constexpr Geometry geometry{
    get_named_compile_time_arg_val("batch_size"),
    get_named_compile_time_arg_val("input_height"),
    get_named_compile_time_arg_val("input_width"),
    stride
};
constexpr bool has_downsample = get_named_compile_time_arg_val("has_downsample") != 0;
constexpr uint32_t cin = (get_named_compile_time_arg_val("in_channels") + 31) / 32;
constexpr uint32_t hidden = (get_named_compile_time_arg_val("channels") + 31) / 32;
constexpr uint32_t cout = (get_named_compile_time_arg_val("out_channels") + 31) / 32;


constexpr uint32_t block = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t patch_input_tiles = get_named_compile_time_arg_val("max_input_tiles");
constexpr uint32_t processing = get_named_compile_time_arg_val("processing_tiles");
constexpr uint32_t operand_slot = get_named_compile_time_arg_val("operand_slot_tiles");
constexpr uint32_t weight_chunk = get_named_compile_time_arg_val("weight_chunk");
constexpr uint32_t weight_slot = get_named_compile_time_arg_val("weight_slot_tiles");
constexpr uint32_t activation_a_tiles = patch_input_tiles * hidden;
constexpr uint32_t activation_b_tiles = processing * hidden;
constexpr uint32_t output_slot = processing * cout;



constexpr uint32_t cb_operand = 0, cb_output = 16, cb_activation_a = 20, cb_activation_b = 21,
                   cb_parameters = 23, cb_zero = 24;
static_assert(processing == 1 || processing == 2, "At most four FP32 DST tiles including residual");
static_assert(weight_chunk > 0 && weight_slot == weight_chunk + 1, "Invalid parameter FIFO slot");
static_assert(
    operand_slot >= processing * cin && operand_slot >= processing * 9 * hidden,
    "Operand FIFO must accommodate every phase"
);



constexpr uint32_t row_offset(uint32_t row, uint32_t half) {
    return ((row / 16) * 2 + half) * 512 + (row % 16) * 32;
}
}
