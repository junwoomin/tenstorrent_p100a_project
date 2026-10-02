#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "stream_common.hpp"
#include "stream_profile.hpp"
#include "timing.hpp"

void kernel_main() {
    bb_timing::Counters timing;
    using namespace bottleneck_stream;
    const uint32_t addr = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1),
                   start = get_arg_val<uint32_t>(2);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, addr, tile_bytes);
    DataflowBuffer out(cb_output);
    Traffic traffic;
    const uint32_t spatial_count = count / cout;
    // Respect patch boundaries: each odd final group has a fixed padded slot,
    // but only actual tiles belonging to this core are written to DRAM.
    for (uint32_t done = 0; done < spatial_count; done += block) {
        const uint32_t rows = min_u(block, spatial_count - done);
        for (uint32_t mt = 0; mt < rows; mt += processing) {
            BB_TIME(timing, OutputWait, out.wait_front(output_slot));
            const uint32_t src = out.get_read_ptr();
            {
            bb_timing::Scope write_scope(timing, bb_timing::Metric::OutputWrite);
            BB_STREAM_ZONE_IF(done == 0 && mt == 0, "BB_FINAL_WRITE", {
                for (uint32_t page = 0; page < min_u(processing, rows - mt) * cout; ++page) {
                    noc_async_write(src + page * tile_bytes,
                                    output.get_noc_addr(start + (done + mt) * cout + page), tile_bytes);
                    traffic.output_write(tile_bytes);
                }
                BB_TIME(timing, OutputBarrier, noc_async_write_barrier());
            });
            }
            out.pop_front(output_slot);
        }
    }
    timing.print("stream", "writer", start / cout, spatial_count);
    traffic.print_writer();
}
