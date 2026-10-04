




#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "sharded_common.hpp"



void kernel_main() {
    using namespace bottleneck_sharded;
    const uint32_t addr = get_arg_val<uint32_t>(0), start = get_arg_val<uint32_t>(1),
                   count = get_arg_val<uint32_t>(2), output_begin = get_arg_val<uint32_t>(3),
                   output_count = get_arg_val<uint32_t>(4);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, addr, tile_bytes);
    DataflowBuffer out(cb_output);

    for (uint32_t done = 0; done < count; done += block) {
        const uint32_t rows = min_u(block, count - done);
        for (uint32_t mt = 0; mt < rows; mt += processing) {
            const uint32_t nm = min_u(processing, rows - mt);
            out.wait_front(output_slot);
            const uint32_t src = out.get_read_ptr();

            for (uint32_t m = 0; m < nm; ++m) {
                for (uint32_t n = 0; n < output_count; ++n) {


                    const uint32_t page = (start + done + mt + m) * cout + output_begin + n;
                    noc_async_write(
                        src + (m * local_cout + n) * tile_bytes,
                        output.get_noc_addr(page),
                        tile_bytes
                    );
                }
            }
            noc_async_write_barrier();



            out.pop_front(output_slot);
        }
    }
}
