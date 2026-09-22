#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

// One computed gradient tile is written to TWO independently allocated buffers.
void kernel_main() {
    const uint32_t grad_a_addr = get_arg_val<uint32_t>(0);
    const uint32_t grad_b_addr = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto grad_a_args = TensorAccessorArgs<0>();
    constexpr auto grad_b_args =
        TensorAccessorArgs<grad_a_args.next_compile_time_args_offset()>();
    const auto grad_a = TensorAccessor(grad_a_args, grad_a_addr);
    const auto grad_b = TensorAccessor(grad_b_args, grad_b_addr);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    DataflowBuffer out_cb(cb_out);
    Noc noc;
    for (uint32_t tile = start; tile < start + count; ++tile) {
        out_cb.wait_front(1);
        noc.async_write(out_cb, grad_a, tile_bytes, {}, {.page_id = tile});
        noc.async_write(out_cb, grad_b, tile_bytes, {}, {.page_id = tile});
        noc.async_write_barrier();
        out_cb.pop_front(1);
    }
}
