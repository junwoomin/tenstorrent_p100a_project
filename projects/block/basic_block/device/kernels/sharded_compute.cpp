




#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "sharded_common.hpp"

using namespace ckernel;
using namespace bottleneck_sharded;

namespace {

void relu_group(uint32_t count) {
    relu_tile_init();
    for (uint32_t i = 0; i < count; ++i) {
        relu_tile(i);
    }
}




void reduce(
    uint32_t source_cb,
    uint32_t k_tiles,
    uint32_t weight_base,
    uint32_t bias_base,
    uint32_t weight_stride,
    uint32_t n,
    uint32_t nm,
    uint32_t dst
) {
    reconfig_data_format_srca(cb_parameters);
    copy_init(cb_parameters);
    for (uint32_t m = 0; m < nm; ++m) {
        copy_tile(cb_parameters, bias_base + n, dst + m);
    }
    reconfig_data_format<SrcOrder::Reverse>(source_cb, cb_parameters);
    matmul_block_init(source_cb, cb_parameters, 0, 1, nm, k_tiles);

    for (uint32_t k = 0; k < k_tiles; ++k) {
        matmul_block(
            source_cb, cb_parameters, k, weight_base + k * weight_stride + n, dst, 0, 1, nm, k_tiles
        );
    }
}



void conv1(const Patch& p, uint32_t hidden_count) {
    DataflowBuffer operand(cb_operand);
    pack_reconfig_data_format(cb_activation_a);
    for (uint32_t mt = 0; mt < p.input_mt_count; mt += processing) {
        const uint32_t nm = min_u(processing, p.input_mt_count - mt);
        operand.wait_front(operand_slot);
        for (uint32_t n = 0; n < hidden_count; ++n) {
            tile_regs_acquire();
            reduce(cb_operand, cin, w0, b0, local_hidden, n, nm, 0);
            relu_group(nm);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t m = 0; m < nm; ++m) {
                pack_tile<true>(m, cb_activation_a, (mt + m) * local_hidden + n);
            }
            tile_regs_release();
        }
        operand.pop_front(operand_slot);
    }
}




void conv2(uint32_t nm, uint32_t hidden_count) {
    DataflowBuffer operand(cb_operand), a2(cb_activation_b);
    a2.reserve_back(activation_b_tiles);
    operand.wait_front(operand_slot);
    pack_reconfig_data_format(cb_activation_b);
    for (uint32_t n = 0; n < hidden_count; ++n) {
        tile_regs_acquire();
        reduce(cb_operand, 9 * hidden, w1, b1, local_hidden, n, nm, 0);
        relu_group(nm);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t m = 0; m < nm; ++m) {
            pack_tile<true>(m, cb_activation_b, m * local_hidden + n);
        }
        tile_regs_release();
    }
    operand.pop_front(operand_slot);

    a2.push_back(activation_b_tiles);
}




void conv3(uint32_t nm, uint32_t output_begin, uint32_t output_count) {
    DataflowBuffer operand(cb_operand), full_b(cb_full_b), out(cb_output);
    full_b.wait_front(full_b_tiles);

    operand.wait_front(operand_slot);

    out.reserve_back(output_slot);
    pack_reconfig_data_format(cb_output);
    for (uint32_t n = 0; n < output_count; ++n) {
        tile_regs_acquire();

        reduce(cb_full_b, hidden, w2, b2, local_cout, n, nm, 0);



        if constexpr (has_downsample) {
            reduce(cb_operand, cin, wd, bd, local_cout, n, nm, processing);
        } else {
            reconfig_data_format_srca(cb_operand);
            copy_init(cb_operand);
            for (uint32_t m = 0; m < nm; ++m) {
                copy_tile(cb_operand, m * cin + output_begin + n, processing + m);
            }
        }

        add_binary_tile_init();
        for (uint32_t m = 0; m < nm; ++m) {
            add_binary_tile(m, processing + m, m);
        }
        relu_group(nm);

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t m = 0; m < nm; ++m) {
            pack_tile<true>(m, cb_output, m * local_cout + n);
        }
        tile_regs_release();
    }


    out.push_back(output_slot);
    operand.pop_front(operand_slot);
    full_b.pop_front(full_b_tiles);
}
}




void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1),
                   hidden_count = get_arg_val<uint32_t>(3), output_begin = get_arg_val<uint32_t>(4),
                   output_count = get_arg_val<uint32_t>(5);
    DataflowBuffer a1(cb_activation_a), params(cb_parameters);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_operand, cb_parameters, cb_activation_a);

    params.wait_front(parameter_tiles);
    for (uint32_t done = 0; done < count; done += block) {
        const auto p = patch(geometry, start + done, min_u(block, count - done));



        a1.reserve_back(activation_a_tiles);
        conv1(p, hidden_count);
        a1.push_back(activation_a_tiles);

        for (uint32_t mt = 0; mt < p.output_mt_count; mt += processing) {
            const uint32_t nm = min_u(processing, p.output_mt_count - mt);

            conv2(nm, hidden_count);

            conv3(nm, output_begin, output_count);
        }
    }
    params.pop_front(parameter_tiles);
}
