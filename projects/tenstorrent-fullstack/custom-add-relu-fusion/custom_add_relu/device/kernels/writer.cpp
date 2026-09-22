#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_accessor = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    DataflowBuffer out_cb(cb_out);
    Noc noc;
    for (uint32_t tile = start; tile < start + count; ++tile) {
        out_cb.wait_front(1);
        noc.async_write(out_cb, out_accessor, tile_bytes, {}, {.page_id = tile});
        noc.async_write_barrier();
        out_cb.pop_front(1);
    }
}
