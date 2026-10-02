#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "sharded_common.hpp"
#include "stream_profile.hpp"
#include "timing.hpp"

using namespace ckernel;
using namespace bottleneck_sharded;
using bottleneck_stream::CycleScope;
using bottleneck_stream::StageTotals;

namespace {
void relu_group(uint32_t count) {
    relu_tile_init();
    for (uint32_t i = 0; i < count; ++i) {
        relu_tile(i);
    }
}

// All of this core's output-channel weights stay in CB23 for the invocation.
// Its leading dimension is the padded local channel count; K is never split
// across workers. Accumulators stay FP32 for the complete reduction.
void reduce(bb_timing::Counters &timing, bb_timing::Metric matmul_metric,
            uint32_t source_cb, uint32_t k_tiles, uint32_t weight_base,
            uint32_t bias_base, uint32_t weight_stride, uint32_t n,
            uint32_t nm, uint32_t dst) {
    reconfig_data_format_srca(cb_parameters);
    copy_init(cb_parameters);
    for (uint32_t m = 0; m < nm; ++m) {
        copy_tile(cb_parameters, bias_base + n, dst + m);
    }
    reconfig_data_format<SrcOrder::Reverse>(source_cb, cb_parameters);
    matmul_block_init(source_cb, cb_parameters, 0, 1, nm, k_tiles);
    bb_timing::Scope issue_clock(timing, matmul_metric);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        // N=1: explicit B indices select this worker's cached output channel.
        // kt_dim describes the full A row stride, not an implicit K loop.
        matmul_block(source_cb, cb_parameters, k, weight_base + k * weight_stride + n,
                     dst, 0, 1, nm, k_tiles);
    }
}

void conv1(const Patch &p, uint32_t hidden_count, bool sample, bb_timing::Counters &timing) {
    DataflowBuffer operand(cb_operand);
    pack_reconfig_data_format(cb_activation_a);
    for (uint32_t mt = 0; mt < p.input_mt_count; mt += processing) {
        const uint32_t nm = min_u(processing, p.input_mt_count - mt);
        BB_TIME(timing, InputWait, operand.wait_front(operand_slot));
        for (uint32_t n = 0; n < hidden_count; ++n) {
            BB_TIME(timing, DstWait, tile_regs_acquire());
            BB_STREAM_ZONE_IF(sample && mt == 0 && n == 0, "BB_CONV1_COMPUTE",
                reduce(timing, bb_timing::Metric::Conv1Matmul, cb_operand, cin, w0, b0, local_hidden, n, nm, 0));
            relu_group(nm);
            tile_regs_commit();
            BB_TIME(timing, DstWait, tile_regs_wait());
            BB_TIME(timing, Pack, for (uint32_t m = 0; m < nm; ++m) {
                pack_tile<true>(m, cb_activation_a, (mt + m) * local_hidden + n);
            });
            tile_regs_release();
        }
        operand.pop_front(operand_slot);
    }
}

void conv2(uint32_t nm, uint32_t hidden_count, bool sample, bb_timing::Counters &timing) {
    DataflowBuffer operand(cb_operand), a2(cb_activation_b);
    BB_TIME(timing, CbWait, a2.reserve_back(activation_b_tiles));
    BB_STREAM_ZONE_IF(sample, "BB_CONV2_INPUT_PRELOAD", BB_TIME(timing, InputWait, operand.wait_front(operand_slot)));
    pack_reconfig_data_format(cb_activation_b);
    for (uint32_t n = 0; n < hidden_count; ++n) {
        BB_TIME(timing, DstWait, tile_regs_acquire());
        BB_STREAM_ZONE_IF(sample && n == 0, "BB_CONV2_COMPUTE",
            reduce(timing, bb_timing::Metric::Conv2Matmul, cb_operand, 9 * hidden, w1, b1, local_hidden, n, nm, 0));
        relu_group(nm);
        tile_regs_commit();
        BB_TIME(timing, DstWait, tile_regs_wait());
        BB_TIME(timing, Pack, for (uint32_t m = 0; m < nm; ++m) {
            pack_tile<true>(m, cb_activation_b, m * local_hidden + n);
        });
        tile_regs_release();
    }
    operand.pop_front(operand_slot);
    // The reader releases CB21 after every peer has fetched its channel slice.
    BB_STREAM_ZONE_IF(sample, "BB_CONV2_OUTPUT_HANDOFF", a2.push_back(activation_b_tiles));
}

void conv3(uint32_t nm, uint32_t output_begin, uint32_t output_count,
           bool sample, StageTotals &totals, bb_timing::Counters &timing) {
    DataflowBuffer operand(cb_operand), full_b(cb_full_b), out(cb_output);
    BB_TIME(timing, CbWait, full_b.wait_front(full_b_tiles));
    {
        CycleScope clock(totals.shortcut);
        BB_STREAM_ZONE_IF(sample, "BB_RESIDUAL_WAIT", BB_TIME(timing, InputWait, operand.wait_front(operand_slot)));
    }
    BB_TIME(timing, CbWait, out.reserve_back(output_slot));
    pack_reconfig_data_format(cb_output);
    for (uint32_t n = 0; n < output_count; ++n) {
        BB_TIME(timing, DstWait, tile_regs_acquire());
        {
            CycleScope clock(totals.conv3);
            bb_timing::Scope stage_clock(timing, bb_timing::Metric::Conv3);
            BB_STREAM_ZONE_IF(sample && n == 0, "BB_CONV3_COMPUTE",
                reduce(timing, bb_timing::Metric::Conv3Matmul, cb_full_b, hidden, w2, b2, local_cout, n, nm, 0));
        }
        {
            CycleScope clock(totals.shortcut);
            bb_timing::Scope stage_clock(timing, bb_timing::Metric::Shortcut);
            if constexpr (has_downsample) {
                BB_STREAM_ZONE_IF(sample && n == 0, "BB_SHORTCUT_COMPUTE",
                    reduce(timing, bb_timing::Metric::ShortcutMatmul, cb_operand, cin, wd, bd, local_cout, n, nm, processing));
            } else {
                reconfig_data_format_srca(cb_operand);
                copy_init(cb_operand);
                for (uint32_t m = 0; m < nm; ++m) {
                    copy_tile(cb_operand, m * cin + output_begin + n, processing + m);
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
            // Main and shortcut use disjoint FP32 DST tiles until their sum
            // and ReLU finish. Only the final output is packed to BF16.
            tile_regs_commit();
            BB_TIME(timing, DstWait, tile_regs_wait());
            BB_TIME(timing, Pack, for (uint32_t m = 0; m < nm; ++m) {
                pack_tile<true>(m, cb_output, m * local_cout + n);
            });
            tile_regs_release();
        }
    }
    out.push_back(output_slot);
    operand.pop_front(operand_slot);
    full_b.pop_front(full_b_tiles);
}
} // namespace

void kernel_main() {
    bb_timing::Counters timing;
    const uint32_t start = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1),
                   hidden_count = get_arg_val<uint32_t>(3),
                   output_begin = get_arg_val<uint32_t>(4), output_count = get_arg_val<uint32_t>(5);
    DataflowBuffer a1(cb_activation_a), params(cb_parameters);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_operand, cb_parameters, cb_activation_a);
    StageTotals totals;
    {
        CycleScope all(totals.total);
        BB_STREAM_ZONE_IF(true, "BB_PARAMETER_CACHE_WAIT", BB_TIME(timing, WeightWait, params.wait_front(parameter_tiles)));
        for (uint32_t done = 0; done < count; done += block) {
            const auto p = patch(geometry, start + done, min_u(block, count - done));
            {
                CycleScope clock(totals.conv1);
                bb_timing::Scope stage_clock(timing, bb_timing::Metric::Conv1);
                // Fixed capacity keeps each worker's CB20 base stable for
                // remote halo reads. The reader owns the matching pop.
                BB_TIME(timing, CbWait, a1.reserve_back(activation_a_tiles));
                BB_STREAM_ZONE_IF(done == 0, "BB_STAGE1_CONV1", conv1(p, hidden_count, done == 0, timing));
                a1.push_back(activation_a_tiles);
            }
            for (uint32_t mt = 0; mt < p.output_mt_count; mt += processing) {
                const uint32_t nm = min_u(processing, p.output_mt_count - mt);
                const bool sample = done == 0 && mt == 0;
                {
                    CycleScope clock(totals.conv2);
                    bb_timing::Scope stage_clock(timing, bb_timing::Metric::Conv2);
                    BB_STREAM_ZONE_IF(sample, "BB_STAGE2_CONV2", conv2(nm, hidden_count, sample, timing));
                }
                BB_STREAM_ZONE_IF(sample, "BB_STAGE3_CONV3_RESIDUAL",
                    conv3(nm, output_begin, output_count, sample, totals, timing));
            }
        }
        params.pop_front(parameter_tiles);
    }
    timing.print("sharded", BB_TIMING_COMPUTE_ROLE, start, count,
                 bottleneck_geometry::channel_shard_owner(get_arg_val<uint32_t>(2), hidden, n_shards));
    totals.print_compute();
}
