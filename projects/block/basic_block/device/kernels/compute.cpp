




#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "stream_common.hpp"

using namespace ckernel;
using namespace bottleneck_stream;

namespace {

void relu_group(uint32_t n) {
    relu_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        relu_tile(i);
    }
}




void reduce(uint32_t source_cb, uint32_t k_tiles, uint32_t nm, uint32_t dst) {
    DataflowBuffer params(cb_parameters);
    for (uint32_t k0 = 0; k0 < k_tiles; k0 += weight_chunk) {
        params.wait_front(weight_slot);
        if (k0 == 0) {
            reconfig_data_format_srca(cb_parameters);
            copy_init(cb_parameters);
            for (uint32_t m = 0; m < nm; ++m) {
                copy_tile(cb_parameters, 0, dst + m);
            }
        }
        reconfig_data_format<SrcOrder::Reverse>(source_cb, cb_parameters);
        matmul_block_init(source_cb, cb_parameters, 0, 1, nm, k_tiles);

        for (uint32_t k = 0; k < min_u(weight_chunk, k_tiles - k0); ++k) {
            matmul_block(source_cb, cb_parameters, k0 + k, 1 + k, dst, 0, 1, nm, k_tiles);
        }

        params.pop_front(weight_slot);
    }
}




void conv1(const Patch& p) {
    DataflowBuffer operand(cb_operand);
    pack_reconfig_data_format(cb_activation_a);
    for (uint32_t mt = 0; mt < p.input_mt_count; mt += processing) {
        const uint32_t nm = min_u(processing, p.input_mt_count - mt);
        operand.wait_front(operand_slot);
        for (uint32_t n = 0; n < hidden; ++n) {
            tile_regs_acquire();
            reduce(cb_operand, cin, nm, 0);
            relu_group(nm);


            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t m = 0; m < nm; ++m) {
                pack_tile<true>(m, cb_activation_a, (mt + m) * hidden + n);
            }
            tile_regs_release();
        }
        operand.pop_front(operand_slot);
    }
}




void conv2(uint32_t nm) {
    DataflowBuffer operand(cb_operand), b(cb_activation_b);
    b.reserve_back(activation_b_tiles);
    operand.wait_front(operand_slot);
    pack_reconfig_data_format(cb_activation_b);
    for (uint32_t n = 0; n < hidden; ++n) {
        tile_regs_acquire();
        reduce(cb_operand, 9 * hidden, nm, 0);
        relu_group(nm);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t m = 0; m < nm; ++m) {
            pack_tile<true>(m, cb_activation_b, m * hidden + n);
        }
        tile_regs_release();
    }

    operand.pop_front(operand_slot);
    b.push_back(activation_b_tiles);
}





void conv3(uint32_t nm) {
    DataflowBuffer operand(cb_operand), b(cb_activation_b), out(cb_output);
    b.wait_front(activation_b_tiles);

    operand.wait_front(operand_slot);

    out.reserve_back(output_slot);
    pack_reconfig_data_format(cb_output);
    for (uint32_t n = 0; n < cout; ++n) {
        tile_regs_acquire();

        reduce(cb_activation_b, hidden, nm, 0);



        if constexpr (has_downsample) {
            reduce(cb_operand, cin, nm, processing);
        } else {
            reconfig_data_format_srca(cb_operand);
            copy_init(cb_operand);
            for (uint32_t m = 0; m < nm; ++m) {
                copy_tile(cb_operand, m * cin + n, processing + m);
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
            pack_tile<true>(m, cb_output, m * cout + n);
        }
        tile_regs_release();
    }
    out.push_back(output_slot);
    operand.pop_front(operand_slot);
    b.pop_front(activation_b_tiles);
}
}



void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    DataflowBuffer a(cb_activation_a);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_operand, cb_parameters, cb_activation_a);



    for (uint32_t done = 0; done < count; done += block) {
        const auto p = patch(geometry, start + done, min_u(block, count - done));

        a.reserve_back(activation_a_tiles);
        conv1(p);
        a.push_back(activation_a_tiles);

        for (uint32_t mt = 0; mt < p.output_mt_count; mt += processing) {
            const uint32_t nm = min_u(processing, p.output_mt_count - mt);

            conv2(nm);

            conv3(nm);
        }
    }
}
