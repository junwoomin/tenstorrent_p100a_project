#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include "l1_plan.hpp"

// Pure host arithmetic, also compiled by the CPU planner tests. Tile counts
// are BF16 32x32 pages. All allocations, including optional DRAM scratch, coexist.
namespace bottleneck_geometry {
struct ShardedPlan {
    bool enabled = false;
    uint32_t m_shards = 0, n_shards = 0, cores = 0;
    uint32_t block = 0, ring = 0, processing_tiles = 2;
    uint32_t cb_depth = 1, out_depth = 1;
    uint32_t local_hidden = 0, local_cout = 0;
    uint32_t parameter_tiles = 0, operand_slot_tiles = 0;
    uint64_t operand_bytes = 0, activation_a_bytes = 0, activation_b_bytes = 0;
    uint64_t full_b_bytes = 0, weight_bytes = 0, output_bytes = 0;
    uint64_t zero_bytes = bf16_tile_bytes, safety_bytes = 4096;
    uint64_t dram_scratch_bytes = 0;
    uint64_t allocated_bytes = 0, required_bytes = 0, available_bytes = 0;
    uint64_t conv1_weight_bytes = 0, conv2_weight_bytes = 0;
    uint64_t conv3_weight_bytes = 0, shortcut_weight_bytes = 0;
};

inline void account_sharded_plan(ShardedPlan& p, uint32_t cin, uint32_t hidden,
                                 uint32_t cout, bool downsample) {
    const uint64_t h = p.local_hidden, o = p.local_cout, m = p.processing_tiles;
    const uint64_t params = uint64_t(cin) * h + h + 9ull * hidden * h + h +
                            uint64_t(hidden) * o + o + (downsample ? uint64_t(cin) * o + o : 0);
    const uint64_t operand = m * std::max<uint64_t>(cin, 9ull * hidden);
    const uint64_t limit = std::numeric_limits<uint32_t>::max();
    p.parameter_tiles = params <= limit ? uint32_t(params) : 0;
    p.operand_slot_tiles = operand <= limit ? uint32_t(operand) : 0;
    p.weight_bytes = params * bf16_tile_bytes;
    p.operand_bytes = operand * p.cb_depth * bf16_tile_bytes;
    p.activation_a_bytes = uint64_t(p.ring) * h * bf16_tile_bytes;
    p.activation_b_bytes = m * h * bf16_tile_bytes;
    p.full_b_bytes = m * hidden * bf16_tile_bytes;
    p.output_bytes = m * o * p.out_depth * bf16_tile_bytes;
    p.dram_scratch_bytes = downsample ? bf16_tile_bytes : 0;
    p.conv1_weight_bytes = uint64_t(cin) * h * bf16_tile_bytes;
    p.conv2_weight_bytes = 9ull * hidden * h * bf16_tile_bytes;
    p.conv3_weight_bytes = uint64_t(hidden) * o * bf16_tile_bytes;
    p.shortcut_weight_bytes = downsample ? uint64_t(cin) * o * bf16_tile_bytes : 0;
    p.allocated_bytes = p.weight_bytes + p.operand_bytes + p.activation_a_bytes +
                        p.activation_b_bytes + p.full_b_bytes + p.output_bytes + p.zero_bytes + p.dram_scratch_bytes;
    p.required_bytes = p.allocated_bytes + p.safety_bytes;
    const uint64_t largest = std::max({p.weight_bytes, p.operand_bytes, p.activation_a_bytes,
        p.activation_b_bytes, p.full_b_bytes, p.output_bytes, p.zero_bytes, p.dram_scratch_bytes});
    // CB producer/consumer counters are 16-bit. Keep every queue capacity below
    // half the counter range as well as checking the byte descriptor range.
    // Real P100a L1 budgets are much tighter, but CPU callers can pass arbitrary
    // budgets and must not produce a descriptor with wrapped queue counts.
    const bool cb_counts_fit = largest / bf16_tile_bytes <= 32767;
    // Reader has 16 scalar/buffer arguments and two physical coordinates per
    // peer. The standard per-core TT kernel runtime-argument limit is 255 words.
    const bool args_fit = uint64_t(16) + 2ull * p.n_shards <= 255;
    p.enabled = p.n_shards && p.m_shards && p.parameter_tiles && p.operand_slot_tiles &&
                largest <= limit && cb_counts_fit && p.required_bytes <= p.available_bytes && args_fit;
    (void)cout;
}

inline ShardedPlan sharded_plan(const Geometry& g, uint32_t cin, uint32_t hidden,
                                uint32_t cout, bool downsample, uint32_t core_limit,
                                uint32_t requested_block, uint64_t available_bytes,
                                uint32_t requested_n_shards = 0) {
    ShardedPlan failed;
    failed.available_bytes = available_bytes;
    if (!cin || !hidden || !cout || !core_limit || !requested_block || !g.batch ||
        !g.height || !g.width || (g.stride != 1 && g.stride != 2)) return failed;
    // Disallow overflow before invoking Geometry's intentionally uint32 API.
    if (uint64_t(g.batch) * g.height * g.width > 0x7fffffffULL) return failed;
    const uint32_t total_m = ceil32(g.output_rows());
    const uint32_t max_n = std::min({hidden, cout, core_limit, 119u});
    if (requested_n_shards > max_n) return failed;

    // Prefer two output rows in FP32 DST; for each processing width use the
    // smallest feasible N fanout. This retains as many M groups as possible and
    // bounds duplicated input/activation traffic. N=1 is considered last.
    for (uint32_t processing : {2u, 1u}) {
        for (uint32_t index = 0; index < (requested_n_shards ? 1u : max_n); ++index) {
            const uint32_t n = requested_n_shards ? requested_n_shards :
                               (index + 2 <= max_n ? index + 2 : 1);
            ShardedPlan p;
            p.available_bytes = available_bytes;
            p.n_shards = n;
            p.m_shards = std::min(total_m, core_limit / n);
            p.cores = p.m_shards * n;
            p.processing_tiles = processing;
            p.local_hidden = hidden / n + (hidden % n != 0);
            p.local_cout = cout / n + (cout % n != 0);
            const uint32_t max_m_count = total_m / p.m_shards + (total_m % p.m_shards != 0);
            p.block = std::min(requested_block, max_m_count);
            while (true) {
                p.ring = maximum_patch_tiles(g, p.m_shards, p.block);
                account_sharded_plan(p, cin, hidden, cout, downsample);
                // Barrier epochs are monotonic, including arrival increments
                // on the leader. Conservatively bound both counters before
                // any descriptor is built. Four rounds per output tile exceeds
                // this protocol's actual 2 patch + 2 group rounds.
                const uint64_t max_epochs = 4ull * max_m_count;
                if (max_epochs * n > std::numeric_limits<uint32_t>::max()) p.enabled = false;
                if (p.enabled) {
                    // Each double queue is optional, and its real allocation is
                    // re-accounted. A small output queue can hide DRAM writes;
                    // a large full-K halo is deliberately kept single-buffered.
                    if (max_m_count > processing && p.output_bytes <= available_bytes / 8) {
                        auto doubled = p;
                        doubled.out_depth = 2;
                        account_sharded_plan(doubled, cin, hidden, cout, downsample);
                        if (doubled.enabled) p = doubled;
                    }
                    if (p.ring > processing && p.operand_bytes <= available_bytes / 4) {
                        auto doubled = p;
                        doubled.cb_depth = 2;
                        account_sharded_plan(doubled, cin, hidden, cout, downsample);
                        if (doubled.enabled) p = doubled;
                    }
                    return p;
                }
                failed = p;
                if (p.block == 1) break;
                p.block = std::max(1u, p.block / 2);
            }
        }
    }
    return failed;
}
} // namespace bottleneck_geometry
