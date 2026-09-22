#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/dataflow_buffer.h"

// Finite BF16 inputs only. dA = dB = grad_out * (saved_output > 0).
// At saved_output == 0, the derivative is zero.
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr auto cb_grad = tt::CBIndex::c_0;
    constexpr auto cb_saved = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    DataflowBuffer grad_cb(cb_grad);
    DataflowBuffer saved_cb(cb_saved);
    DataflowBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_saved, cb_out);
    for (uint32_t i = 0; i < count; ++i) {
        grad_cb.wait_front(1);
        saved_cb.wait_front(1);
        out_cb.reserve_back(1);
        tile_regs_acquire();

        // Both CBs use the same tile shape and BF16 data format.
        copy_init(cb_saved);
        copy_tile(cb_saved, 0, 0);
        unary_gt_tile_init();
        unary_gt_tile(0, 0u);  // DST[0] = saved_output > float(0)

        copy_init(cb_grad);
        copy_tile(cb_grad, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // DST[0] = mask * grad_out

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out_cb.push_back(1);
        grad_cb.pop_front(1);
        saved_cb.pop_front(1);
    }
}
