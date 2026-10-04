




#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/dataflow/dataflow_buffer.h"
#define BB_BACKWARD_KERNEL
#include "backward_common.hpp"
using namespace ckernel;
using namespace bottleneck_backward;




void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t left = mode == BIAS_GRAD ? cb_ones : cb_a;
    DataflowBuffer a(left), b(cb_b), out(cb_out);
    compute_kernel_hw_startup<SrcOrder::Reverse>(left, cb_b, cb_out);
    if constexpr (mode != ADD) {
        DataflowBuffer zero(cb_zero);
        zero.wait_front(1);
    }
    if constexpr (mode == BIAS_GRAD) {
        a.wait_front(1);
    }
    for (uint32_t job = 0; job < count; ++job) {
        out.reserve_back(1);
        tile_regs_acquire();

        if constexpr (mode == ADD) {
            a.wait_front(1);
            b.wait_front(1);
            reconfig_data_format_srca(cb_a);
            copy_init(cb_a);
            copy_tile(cb_a, 0, 0);
            reconfig_data_format_srca(cb_b);
            copy_init(cb_b);
            copy_tile(cb_b, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            a.pop_front(1);
            b.pop_front(1);
        } else {
            constexpr uint32_t initial = mode == FORWARD ? cb_bias : cb_zero;
            DataflowBuffer seed(initial);
            if constexpr (mode == FORWARD) {
                seed.wait_front(1);
            }
            reconfig_data_format_srca(initial);
            copy_init(initial);
            copy_tile(initial, 0, 0);
            if constexpr (mode == FORWARD) {
                seed.pop_front(1);
            }
            reconfig_data_format<SrcOrder::Reverse>(left, cb_b);


            matmul_init(left, cb_b, mode == INPUT_GRAD ? 1 : 0);
            const uint32_t reduction =
                mode == FORWARD
                    ? taps * cin_tiles
                    : (mode == INPUT_GRAD ? taps * cout_tiles : tiles(geometry.output_rows()));



            for (uint32_t k = 0; k < reduction; ++k) {
                if constexpr (mode != BIAS_GRAD) {
                    a.wait_front(1);
                }
                b.wait_front(1);
                matmul_tiles(left, cb_b, 0, 0, 0);
                if constexpr (mode != BIAS_GRAD) {
                    a.pop_front(1);
                }
                b.pop_front(1);
            }
            if constexpr (mode == FORWARD && apply_relu) {
                relu_tile_init();
                relu_tile(0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
    }
    if constexpr (mode != ADD) {
        DataflowBuffer zero(cb_zero);
        zero.pop_front(1);
    }
    if constexpr (mode == BIAS_GRAD) {
        a.pop_front(1);
    }
}
