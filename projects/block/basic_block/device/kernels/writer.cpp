



#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "stream_common.hpp"



void kernel_main() {
    using namespace bottleneck_stream;
    const uint32_t addr = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1),
                   start = get_arg_val<uint32_t>(2);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, addr, tile_bytes);
    DataflowBuffer out(cb_output);

    const uint32_t spatial_count = count / cout;



    for (uint32_t done = 0; done < spatial_count; done += block) {
        const uint32_t rows = min_u(block, spatial_count - done);
        for (uint32_t mt = 0; mt < rows; mt += processing) {
            out.wait_front(output_slot);
            const uint32_t src = out.get_read_ptr();

            for (uint32_t page = 0; page < min_u(processing, rows - mt) * cout; ++page) {
                noc_async_write(
                    src + page * tile_bytes,
                    output.get_noc_addr(start + (done + mt) * cout + page),
                    tile_bytes
                );
            }
            noc_async_write_barrier();

            out.pop_front(output_slot);
        }
    }
}
