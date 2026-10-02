#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "stream_common.hpp"
#include "stream_profile.hpp"
#include "timing.hpp"

using namespace ckernel;
using namespace bottleneck_stream;

namespace {
void relu_group(uint32_t n) {
    relu_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        relu_tile(i);
    }
}

// DST is acquired by the caller and remains FP32 across every K panel. Only
// the first panel initializes it with the bias. CB23 is consumed immediately
// after unpacking each panel; its backing store is shared by all four stages.
void reduce(bb_timing::Counters &timing, bb_timing::Metric matmul_metric,
            uint32_t source_cb, uint32_t k_tiles, uint32_t nm, uint32_t dst) {
    DataflowBuffer params(cb_parameters);
    for (uint32_t k0 = 0; k0 < k_tiles; k0 += weight_chunk) {
        BB_TIME(timing, WeightWait, params.wait_front(weight_slot));
        if (k0 == 0) {
            reconfig_data_format_srca(cb_parameters);
            copy_init(cb_parameters);
            for (uint32_t m = 0; m < nm; ++m) {
                copy_tile(cb_parameters, 0, dst + m);
            }
        }
        reconfig_data_format<SrcOrder::Reverse>(source_cb, cb_parameters);
        matmul_block_init(source_cb, cb_parameters, 0, 1, nm, k_tiles);
        {
            bb_timing::Scope issue_clock(timing, matmul_metric);
            for (uint32_t k = 0; k < min_u(weight_chunk, k_tiles - k0); ++k) {
                // kt_dim is the A row stride, NOT a hidden additional reduction.
                // N=1 allows arbitrary K panels without reformatting the weights.
                matmul_block(source_cb, cb_parameters, k0 + k, 1 + k, dst, 0, 1, nm, k_tiles);
            }
        }
        params.pop_front(weight_slot);
    }
}

void conv1(const Patch &p, bb_timing::Counters &timing) {
    DataflowBuffer operand(cb_operand);
    pack_reconfig_data_format(cb_activation_a);
    for (uint32_t mt = 0; mt < p.input_mt_count; mt += processing) {
        const uint32_t nm = min_u(processing, p.input_mt_count - mt);
        BB_TIME(timing, InputWait, operand.wait_front(operand_slot));
        for (uint32_t n = 0; n < hidden; ++n) {
            BB_TIME(timing, DstWait, tile_regs_acquire());
            reduce(timing, bb_timing::Metric::Conv1Matmul, cb_operand, cin, nm, 0);
            relu_group(nm);
            tile_regs_commit();
            BB_TIME(timing, DstWait, tile_regs_wait());
            BB_TIME(timing, Pack, for (uint32_t m = 0; m < nm; ++m) {
                pack_tile<true>(m, cb_activation_a, (mt + m) * hidden + n);
            });
            tile_regs_release();
        }
        operand.pop_front(operand_slot);
    }
}

void conv2(uint32_t nm, bool sample, bb_timing::Counters &timing) {
    DataflowBuffer operand(cb_operand), b(cb_activation_b);
    BB_TIME(timing, CbWait, b.reserve_back(activation_b_tiles));
    BB_STREAM_ZONE_IF(sample, "BB_CONV2_INPUT_PRELOAD", BB_TIME(timing, InputWait, operand.wait_front(operand_slot)));
    pack_reconfig_data_format(cb_activation_b);
    for (uint32_t n = 0; n < hidden; ++n) {
        BB_TIME(timing, DstWait, tile_regs_acquire());
        BB_STREAM_ZONE_IF(sample && n == 0, "BB_CONV2_COMPUTE",
            reduce(timing, bb_timing::Metric::Conv2Matmul, cb_operand, 9 * hidden, nm, 0));
        relu_group(nm);
        tile_regs_commit();
        BB_TIME(timing, DstWait, tile_regs_wait());
        BB_TIME(timing, Pack, for (uint32_t m = 0; m < nm; ++m) {
            pack_tile<true>(m, cb_activation_b, m * hidden + n);
        });
        tile_regs_release();
    }
    // The same window was consumed by every output channel before release.
    operand.pop_front(operand_slot);
    BB_STREAM_ZONE_IF(sample, "BB_CONV2_OUTPUT_HANDOFF", b.push_back(activation_b_tiles));
}

void conv3(uint32_t nm, bool sample, StageTotals &totals, bb_timing::Counters &timing) {
    DataflowBuffer operand(cb_operand), b(cb_activation_b), out(cb_output);
    BB_TIME(timing, CbWait, b.wait_front(activation_b_tiles));
    {
        CycleScope clock(totals.shortcut);
        BB_STREAM_ZONE_IF(sample, "BB_RESIDUAL_WAIT", BB_TIME(timing, InputWait, operand.wait_front(operand_slot)));
    }
    BB_TIME(timing, CbWait, out.reserve_back(output_slot));
    pack_reconfig_data_format(cb_output);
    for (uint32_t n = 0; n < cout; ++n) {
        BB_TIME(timing, DstWait, tile_regs_acquire());
        {
            CycleScope clock(totals.conv3);
            bb_timing::Scope stage_clock(timing, bb_timing::Metric::Conv3);
            BB_STREAM_ZONE_IF(sample && n == 0, "BB_CONV3_COMPUTE",
                reduce(timing, bb_timing::Metric::Conv3Matmul, cb_activation_b, hidden, nm, 0));
        }
        {
            CycleScope clock(totals.shortcut);
            bb_timing::Scope stage_clock(timing, bb_timing::Metric::Shortcut);
            if constexpr (has_downsample) {
                // Main and residual occupy disjoint DST tiles. There is one
                // compute engine/core; reader prefetch may overlap this work,
                // but two matmuls are not incorrectly called parallel threads.
                BB_STREAM_ZONE_IF(sample && n == 0, "BB_SHORTCUT_COMPUTE",
                    reduce(timing, bb_timing::Metric::ShortcutMatmul, cb_operand, cin, nm, processing));
            } else {
                reconfig_data_format_srca(cb_operand);
                copy_init(cb_operand);
                for (uint32_t m = 0; m < nm; ++m) {
                    copy_tile(cb_operand, m * cin + n, processing + m);
                }
            }
        }
        {
            CycleScope clock(totals.residual);
            bb_timing::Scope stage_clock(timing, bb_timing::Metric::Residual);
            BB_STREAM_ZONE_IF(sample && n == 0, "BB_RESIDUAL_ADD", {
                add_binary_tile_init();
                for (uint32_t m = 0; m < nm; ++m) {
                    add_binary_tile(m, processing + m, m);
                }
                relu_group(nm);
            });
            // Conv3 and projection are rounded only AFTER add + ReLU, exactly
            // as in the original direct fused path; no BF16 partial-sum spill.
            tile_regs_commit();
            BB_TIME(timing, DstWait, tile_regs_wait());
            BB_TIME(timing, Pack, for (uint32_t m = 0; m < nm; ++m) {
                pack_tile<true>(m, cb_output, m * cout + n);
            });
            tile_regs_release();
        }
    }
    out.push_back(output_slot);
    operand.pop_front(operand_slot);
    b.pop_front(activation_b_tiles);
}
} // namespace

void kernel_main() {
    bb_timing::Counters timing;
    const uint32_t start = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    DataflowBuffer a(cb_activation_a);
    // Current TT-Metal requires the matmul SrcOrder reversal at HW startup.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_operand, cb_parameters, cb_activation_a);
    StageTotals totals;
    {
        CycleScope all(totals.total);
        for (uint32_t done = 0; done < count; done += block) {
            const auto p = patch(geometry, start + done, min_u(block, count - done));
            {
                CycleScope clock(totals.conv1);
                bb_timing::Scope stage_clock(timing, bb_timing::Metric::Conv1);
                // Reserve the entire A region: the previous patch's reader
                // must explicitly release it before any random-offset pack.
                BB_TIME(timing, CbWait, a.reserve_back(activation_a_tiles));
                BB_STREAM_ZONE_IF(done == 0, "BB_STAGE1_CONV1", conv1(p, timing));
                a.push_back(activation_a_tiles);
            }
            for (uint32_t mt = 0; mt < p.output_mt_count; mt += processing) {
                const uint32_t nm = min_u(processing, p.output_mt_count - mt);
                const bool sample = done == 0 && mt == 0;
                {
                    CycleScope clock(totals.conv2);
                    bb_timing::Scope stage_clock(timing, bb_timing::Metric::Conv2);
                    BB_STREAM_ZONE_IF(sample, "BB_STAGE2_CONV2", conv2(nm, sample, timing));
                }
                BB_STREAM_ZONE_IF(sample, "BB_STAGE3_CONV3_RESIDUAL", conv3(nm, sample, totals, timing));
            }
        }
    }
    timing.print("stream", BB_TIMING_COMPUTE_ROLE, start, count);
    totals.print_compute();
}
