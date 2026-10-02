#pragma once
#include <cstdint>
#include "block_geometry.hpp"

namespace bottleneck_sharded {
using namespace bottleneck_geometry;
constexpr uint32_t tile_bytes = 2048;
constexpr uint32_t stride = get_named_compile_time_arg_val("stride");
constexpr Geometry geometry{get_named_compile_time_arg_val("batch_size"),
    get_named_compile_time_arg_val("input_height"), get_named_compile_time_arg_val("input_width"), stride};
constexpr bool has_downsample = get_named_compile_time_arg_val("has_downsample") != 0;
constexpr uint32_t cin = (get_named_compile_time_arg_val("in_channels") + 31) / 32;
constexpr uint32_t hidden = (get_named_compile_time_arg_val("channels") + 31) / 32;
constexpr uint32_t cout = (get_named_compile_time_arg_val("out_channels") + 31) / 32;
constexpr uint32_t n_shards = get_named_compile_time_arg_val("n_shards");
constexpr uint32_t local_hidden = (hidden + n_shards - 1) / n_shards;
constexpr uint32_t local_cout = (cout + n_shards - 1) / n_shards;
constexpr uint32_t block = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t patch_input_tiles = get_named_compile_time_arg_val("max_input_tiles");
constexpr uint32_t processing = get_named_compile_time_arg_val("processing_tiles");
constexpr uint32_t operand_slot = get_named_compile_time_arg_val("operand_slot_tiles");
constexpr uint32_t activation_a_tiles = patch_input_tiles * local_hidden;
constexpr uint32_t activation_b_tiles = processing * local_hidden;
constexpr uint32_t full_b_tiles = processing * hidden;
constexpr uint32_t output_slot = processing * local_cout;
constexpr uint32_t w0 = 0, b0 = cin * local_hidden, w1 = b0 + local_hidden,
    b1 = w1 + 9 * hidden * local_hidden, w2 = b1 + local_hidden,
    b2 = w2 + hidden * local_cout, wd = b2 + local_cout,
    bd = wd + cin * local_cout;
constexpr uint32_t parameter_tiles = has_downsample ? bd + local_cout : wd;
constexpr uint32_t cb_operand = 0, cb_output = 16, cb_activation_a = 20,
    cb_activation_b = 21, cb_full_b = 22, cb_parameters = 23, cb_zero = 24;
// Reader: X,W1,b1,W2,b2,W3,b3,Wd,bd,M start/count,N lane,H begin/count,O begin/count,
// then n_shards pairs of physical worker x,y coordinates (same M group).
constexpr uint32_t reader_peer_args = 16;
// Compute: M start,count,H begin,count,O begin,count.
// Writer: Y,M start,count,O begin,count.
static_assert(processing == 1 || processing == 2, "FP32 DST capacity exceeded");
static_assert(operand_slot >= processing * cin && operand_slot >= processing * 9 * hidden,
    "Operand slot too small");
constexpr uint32_t row_offset(uint32_t row, uint32_t half) {
    return ((row / 16) * 2 + half) * 512 + (row % 16) * 32;
}
// Inverse of balanced core_work; all shards are nonempty (planner invariant).
inline uint32_t channel_owner(uint32_t channel, uint32_t total) {
    return channel_shard_owner(channel, total, n_shards);
}
} // namespace bottleneck_sharded
