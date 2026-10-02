#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include "block_geometry.hpp"

// Host-only arithmetic with no SDK dependencies; shared with CPU tests.
// A pop_front releases queue ownership, NOT its program-lifetime allocation.
// Physical peak therefore sums every distinct fixed backing allocation. Reuse
// is real here: all phase operands use CB0 and all weights/biases use CB23.
namespace bottleneck_geometry {
constexpr uint64_t bf16_tile_bytes = 2048;

struct StreamPlan {
    bool enabled = false;
    uint32_t cores = 0, block = 0, ring = 0;
    uint32_t processing_tiles = 2, cb_depth = 2, weight_depth = 2, out_depth = 2;
    uint32_t weight_chunk = 1, operand_slot_tiles = 0, weight_slot_tiles = 0;
    uint64_t operand_bytes = 0, activation_a_bytes = 0, activation_b_bytes = 0;
    uint64_t weight_bytes = 0, output_bytes = 0, zero_bytes = bf16_tile_bytes;
    uint64_t dram_scratch_bytes = 0;
    uint64_t required_bytes = 0, available_bytes = 0;
    uint64_t phase_conv1_bytes = 0, phase_conv2_bytes = 0;
    uint64_t phase_conv3_bytes = 0, phase_add_bytes = 0;
    uint64_t conv1_weight_bytes = 0, conv2_weight_bytes = 0;
    uint64_t conv3_weight_bytes = 0, shortcut_weight_bytes = 0;
    uint64_t initial_single_bytes = 0, initial_double_bytes = 0;
    uint64_t selected_single_bytes = 0, selected_double_bytes = 0;
    uint32_t overlap_mask = 0;
    // Ordered adaptation flags: block, CB depth, ping-pong, weight chunk, M.
    uint32_t reductions = 0;
};

inline uint32_t maximum_patch_tiles(const Geometry &g, uint32_t cores, uint32_t block) {
    uint32_t ring = 0;
    const uint32_t total = ceil32(g.output_rows());
    for (uint32_t core = 0; core < cores; ++core) {
        uint32_t start, count;
        core_work(total, cores, core, start, count);
        for (uint32_t done = 0; done < count; done += block) {
            ring = std::max(ring, patch(g, start + done, std::min(block, count - done)).input_mt_count);
        }
    }
    return ring;
}

inline void account_stream_plan(StreamPlan &p, uint32_t cin, uint32_t hidden,
                                uint32_t cout, bool downsample) {
    const uint64_t m = p.processing_tiles, bytes = bf16_tile_bytes;
    const uint64_t operand_slot = m * std::max<uint64_t>(cin, uint64_t(9) * hidden);
    p.operand_slot_tiles = operand_slot > std::numeric_limits<uint32_t>::max()
                               ? 0 : uint32_t(operand_slot);
    p.weight_slot_tiles = p.weight_chunk + 1; // Chunk and bias in the same queue slot.
    p.operand_bytes = operand_slot * p.cb_depth * bytes;
    p.activation_a_bytes = uint64_t(p.ring) * hidden * bytes;
    p.activation_b_bytes = m * hidden * bytes;
    p.weight_bytes = uint64_t(p.weight_slot_tiles) * p.weight_depth * bytes;
    p.output_bytes = m * cout * p.out_depth * bytes;
    p.dram_scratch_bytes = downsample ? bf16_tile_bytes : 0;
    p.required_bytes = p.operand_bytes + p.activation_a_bytes + p.activation_b_bytes +
                       p.weight_bytes + p.output_bytes + p.zero_bytes + p.dram_scratch_bytes;
    const uint64_t fixed = p.activation_a_bytes + p.activation_b_bytes + p.zero_bytes + p.dram_scratch_bytes;
    const uint64_t one_queue = operand_slot * bytes + uint64_t(p.weight_slot_tiles) * bytes + m * cout * bytes;
    p.selected_single_bytes = fixed + one_queue;
    p.selected_double_bytes = fixed + 2 * one_queue;
    p.overlap_mask = (p.cb_depth == 2 ? 1u : 0u) | (p.weight_depth == 2 ? 2u : 0u) |
                     (p.out_depth == 2 ? 4u : 0u);
    const auto stage_weight = [&](uint32_t k) {
        return uint64_t(std::min(k, p.weight_chunk) + 1) * p.weight_depth * bytes;
    };
    p.conv1_weight_bytes = stage_weight(cin);
    p.conv2_weight_bytes = stage_weight(9 * hidden);
    p.conv3_weight_bytes = stage_weight(hidden);
    p.shortcut_weight_bytes = downsample ? stage_weight(cin) : 0;
    // All physical CB allocations, including optional DRAM scratch, coexist in every phase. Queue prefetch
    // may also hold operands/weights from the next phase. Report the physical
    // reservation honestly instead of subtracting empty queues or assuming
    // that logical payload lifetime frees their backing allocation.
    p.phase_conv1_bytes = p.phase_conv2_bytes = p.phase_conv3_bytes = p.phase_add_bytes = p.required_bytes;
    // CB descriptors use uint32_t sizes. Failing here is preferable to truncation.
    const uint64_t max_cb = std::max({p.operand_bytes, p.activation_a_bytes, p.activation_b_bytes,
                                      p.weight_bytes, p.output_bytes, p.zero_bytes, p.dram_scratch_bytes});
    p.enabled = p.operand_slot_tiles && max_cb <= std::numeric_limits<uint32_t>::max() &&
                p.required_bytes <= p.available_bytes;
}

inline StreamPlan streaming_plan(const Geometry &g, uint32_t cin, uint32_t hidden,
                                 uint32_t cout, bool downsample, uint32_t cores,
                                 uint32_t requested_block, uint32_t parameter_limit,
                                 uint64_t available_bytes) {
    StreamPlan p;
    p.available_bytes = available_bytes;
    if (!cores || !requested_block || !parameter_limit || !cin || !hidden || !cout ||
        !g.batch || !g.height || !g.width || (g.stride != 1 && g.stride != 2)) {
        return p;
    }
    p.cores = std::min(cores, ceil32(g.output_rows()));
    p.block = std::min(requested_block, (ceil32(g.output_rows()) + p.cores - 1) / p.cores);
    p.weight_chunk = std::min(parameter_limit, std::max(cin, 9 * hidden));
    const auto update = [&] {
        p.ring = maximum_patch_tiles(g, p.cores, p.block);
        account_stream_plan(p, cin, hidden, cout, downsample);
    };
    update();
    p.initial_double_bytes = p.required_bytes;
    auto single = p;
    single.cb_depth = single.weight_depth = single.out_depth = 1;
    account_stream_plan(single, cin, hidden, cout, downsample);
    p.initial_single_bytes = single.required_bytes;
    // Doubling a large halo queue is often counterproductive. Use a bounded
    // overlap heuristic, not "double everything that fits". These percentages
    // are policy limits, not claimed performance measurements. NoC/compute
    // overlap must still be measured for the caller's geometry on its device.
    const uint32_t max_core_work = (ceil32(g.output_rows()) + p.cores - 1) / p.cores;
    p.cb_depth = p.operand_bytes <= available_bytes / 4 &&
                         (p.ring > p.processing_tiles || p.block > p.processing_tiles) ? 2 : 1;
    p.weight_depth = p.weight_bytes <= available_bytes / 8 &&
                             (std::max(cin, 9 * hidden) > p.weight_chunk || hidden > 1 || cout > 1) ? 2 : 1;
    p.out_depth = p.output_bytes <= available_bytes / 8 && max_core_work > p.processing_tiles ? 2 : 1;
    update();
    if (p.enabled) return p;
    // This ordering is deliberate. Never substitute a DRAM activation tensor.
    while (p.block > 1) {
        p.block = std::max(1u, p.block / 2);
        p.reductions |= 1;
        update();
        if (p.enabled) return p;
    }
    if (p.cb_depth > 1) p.reductions |= 2;
    p.cb_depth = 1;
    update();
    if (p.enabled) return p;
    if (p.weight_depth > 1) p.reductions |= 4;
    p.weight_depth = 1;
    update();
    if (p.enabled) return p;
    if (p.out_depth > 1) p.reductions |= 4;
    p.out_depth = 1;
    update();
    if (p.enabled) return p;
    while (p.weight_chunk > 1) {
        p.weight_chunk = std::max(1u, p.weight_chunk / 2);
        p.reductions |= 8;
        update();
        if (p.enabled) return p;
    }
    p.processing_tiles = 1;
    p.reductions |= 16;
    update();
    return p;
}
} // namespace bottleneck_geometry
