#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#define BB_BACKWARD_KERNEL
#include "backward_common.hpp"

void kernel_main() {
    using namespace bottleneck_backward;
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0), tile_bytes);
    const uint32_t start = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    DataflowBuffer out(cb_out);
    for (uint32_t page = start; page < start + count; ++page) {
        out.wait_front(1);
        const uint32_t address = out.get_read_ptr(), mt = page / result_nt, nt = page % result_nt;
        const bool channel_tail = (nt + 1) * 32 > result_cols;
        const bool row_tail = mode == WEIGHT_GRAD ? (mt % cin_tiles + 1) * 32 > geometry.cin :
            (mt + 1) * 32 > result_rows;
        if (channel_tail || row_tail) {
            auto ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(address);
            for (uint32_t r = 0; r < 32; ++r) {
                const bool valid_row = mode == WEIGHT_GRAD ? (mt % cin_tiles) * 32 + r < geometry.cin :
                    mt * 32 + r < result_rows;
                for (uint32_t c = 0; c < 32; ++c)
                    if (!valid_row || nt * 32 + c >= result_cols) ptr[tile_offset(r, c)] = 0;
            }
        }
        noc_async_write(address, output.get_noc_addr(page), tile_bytes);
        noc_async_write_barrier();
        out.pop_front(1);
    }
}
