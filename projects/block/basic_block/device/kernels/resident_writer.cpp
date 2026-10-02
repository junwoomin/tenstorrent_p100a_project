#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "block_geometry.hpp"
#include "stream_profile.hpp"
#include "timing.hpp"

void kernel_main() {
    bb_timing::Counters timing;
    using namespace bottleneck_geometry;
    constexpr uint32_t output_channel_tiles = (get_named_compile_time_arg_val("out_channels") + 31) / 32;
    constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
    const uint32_t addr = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1),
                   start = get_arg_val<uint32_t>(2);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, addr, 2048);
    DataflowBuffer cb(16);
    Noc noc;
    bottleneck_stream::Traffic traffic;
    // Respect patch boundaries: an odd-sized patch publishes a padded 2M group.
    for (uint32_t done = 0; done < count / output_channel_tiles; done += block_tiles) {
        const uint32_t rows = min_u(block_tiles, count / output_channel_tiles - done);
        for (uint32_t m = 0; m < rows; m += 2) {
            BB_TIME(timing, OutputWait, cb.wait_front(2 * output_channel_tiles));
            {
            bb_timing::Scope write_scope(timing, bb_timing::Metric::OutputWrite);
            for (uint32_t page = 0; page < min_u(2, rows - m) * output_channel_tiles; ++page) {
                traffic.output_write(2048);
                noc.async_write(cb, output, 2048, {.offset_bytes = page * 2048},
                                {.page_id = start + (done + m) * output_channel_tiles + page});
            }
            BB_TIME(timing, OutputBarrier, noc.async_write_barrier());
            }
            cb.pop_front(2 * output_channel_tiles);
        }
    }
    timing.print("resident", "writer", start / output_channel_tiles, count / output_channel_tiles);
    traffic.print_writer();
}
