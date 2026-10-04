



#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "block_geometry.hpp"



void kernel_main() {
    using namespace bottleneck_geometry;
    constexpr uint32_t output_channel_tiles =
        (get_named_compile_time_arg_val("out_channels") + 31) / 32;
    constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
    const uint32_t addr = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1),
                   start = get_arg_val<uint32_t>(2);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, addr, 2048);
    DataflowBuffer cb(16);
    Noc noc;



    for (uint32_t done = 0; done < count / output_channel_tiles; done += block_tiles) {
        const uint32_t rows = min_u(block_tiles, count / output_channel_tiles - done);
        for (uint32_t m = 0; m < rows; m += 2) {
            cb.wait_front(2 * output_channel_tiles);

            for (uint32_t page = 0; page < min_u(2, rows - m) * output_channel_tiles; ++page) {
                noc.async_write(
                    cb,
                    output,
                    2048,
                    {.offset_bytes = page * 2048},
                    {.page_id = start + (done + m) * output_channel_tiles + page}
                );
            }
            noc.async_write_barrier();

            cb.pop_front(2 * output_channel_tiles);
        }
    }
}
