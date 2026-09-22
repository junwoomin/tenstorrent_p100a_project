#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    DataflowBuffer a_cb(cb_a);
    DataflowBuffer b_cb(cb_b);
    DataflowBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    add_init(cb_a, cb_b);
    relu_tile_init();
    for (uint32_t i = 0; i < count; ++i) {
        a_cb.wait_front(1);
        b_cb.wait_front(1);
        out_cb.reserve_back(1);
        tile_regs_acquire();
        add_tiles(cb_a, cb_b, 0, 0, 0);
        relu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out_cb.push_back(1);
        a_cb.pop_front(1);
        b_cb.pop_front(1);
    }
}
