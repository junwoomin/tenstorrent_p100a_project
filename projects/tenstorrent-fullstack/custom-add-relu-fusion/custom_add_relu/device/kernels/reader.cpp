#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a_accessor = TensorAccessor(a_args, a_addr);
    const auto b_accessor = TensorAccessor(b_args, b_addr);
    const uint32_t a_tile_bytes = get_tile_size(cb_a);
    const uint32_t b_tile_bytes = get_tile_size(cb_b);
    DataflowBuffer a_cb(cb_a);
    DataflowBuffer b_cb(cb_b);
    Noc noc;
    for (uint32_t tile = start; tile < start + count; ++tile) {
        a_cb.reserve_back(1);
        b_cb.reserve_back(1);
        noc.async_read(a_accessor, a_cb, a_tile_bytes,
                       {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read(b_accessor, b_cb, b_tile_bytes,
                       {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        a_cb.push_back(1);
        b_cb.push_back(1);
    }
}
