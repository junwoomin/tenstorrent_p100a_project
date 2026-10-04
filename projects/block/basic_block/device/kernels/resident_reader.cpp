




#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "block_geometry.hpp"

namespace {
using namespace bottleneck_geometry;
constexpr uint32_t bytes = 2048;
constexpr uint32_t stride = get_named_compile_time_arg_val("stride");
constexpr Geometry geometry{
    get_named_compile_time_arg_val("batch_size"),
    get_named_compile_time_arg_val("input_height"),
    get_named_compile_time_arg_val("input_width"),
    stride
};
constexpr uint32_t input_channel_tiles = (get_named_compile_time_arg_val("in_channels") + 31) / 32;
constexpr uint32_t hidden_channel_tiles = (get_named_compile_time_arg_val("channels") + 31) / 32;
constexpr uint32_t output_channel_tiles =
    (get_named_compile_time_arg_val("out_channels") + 31) / 32;
constexpr uint32_t input_ring_tiles = get_named_compile_time_arg_val("max_input_tiles");
constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t parameter_tiles = get_named_compile_time_arg_val("parameter_tiles");
constexpr bool has_downsample = get_named_compile_time_arg_val("has_downsample") != 0;



constexpr uint32_t row_offset(uint32_t row, uint32_t half) {
    return ((row / 16) * 2 + half) * 512 + (row % 16) * 32;
}


struct Run {
    uint32_t dest, source, length;
    bool valid;
};




uint32_t make_runs(uint32_t output_mt, uint32_t ky, uint32_t kx, uint32_t padding, Run* runs) {
    const uint32_t oh = geometry.out_height(), ow = geometry.out_width(), area = oh * ow;
    uint32_t count = 0;
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t position = output_mt * 32 + r;
        const uint32_t image = position / area, local = position % area;
        const int32_t y = int32_t((local / ow) * stride + ky) - int32_t(padding);
        const int32_t x = int32_t((local % ow) * stride + kx) - int32_t(padding);
        const bool valid = position < geometry.output_rows() && y >= 0 && x >= 0 &&
                           y < int32_t(geometry.height) && x < int32_t(geometry.width);
        const uint32_t source =
            valid ? (image * geometry.height + uint32_t(y)) * geometry.width + uint32_t(x) : 0;
        if (count) {
            auto& last = runs[count - 1];
            const bool same_dest_face = last.dest / 16 == r / 16;
            const bool adjacent = valid && last.valid && source == last.source + last.length &&
                                  source / 16 == last.source / 16;
            if (same_dest_face && (adjacent || (!valid && !last.valid))) {
                ++last.length;
                continue;
            }
        }
        runs[count++] = {r, source, 1, valid};
    }
    return count;
}





void gather(
    DataflowBuffer& destination,
    DataflowBuffer& source,
    DataflowBuffer& zero,
    const Patch& p,
    uint32_t local_m,
    uint32_t channels,
    bool shortcut
) {
    const uint32_t taps = shortcut ? 1 : 9, group = 2 * taps * channels;
    destination.reserve_back(group);

    const auto dst = destination.get_write_ptr(), src = source.get_read_ptr(),
               z = zero.get_read_ptr();
    const uint32_t valid_m = min_u(2, p.output_mt_count - local_m);
    for (uint32_t m = 0; m < valid_m; ++m) {
        for (uint32_t tap = 0; tap < taps; ++tap) {
            Run runs[32];
            uint32_t nr = 0;
            if (shortcut) {
                nr = make_runs(p.output_mt_begin + local_m + m, 0, 0, 0, runs);
            } else {
                nr = make_runs(p.output_mt_begin + local_m + m, tap / 3, tap % 3, 1, runs);
            }
            for (uint32_t ct = 0; ct < channels; ++ct) {
                for (uint32_t j = 0; j < nr; ++j) {
                    const auto& run = runs[j];
                    if (run.valid) {
                        ASSERT(run.source / 32 >= p.input_mt_begin);
                        ASSERT(run.source / 32 < p.input_mt_begin + p.input_mt_count);
                    }
                    const uint32_t page = ((run.source / 32) % input_ring_tiles) * channels + ct;
                    for (uint32_t half = 0; half < 2; ++half) {
                        const auto from =
                            run.valid ? src + page * bytes + row_offset(run.source % 32, half) : z;
                        const auto to = dst + ((m * taps + tap) * channels + ct) * bytes +
                                        row_offset(run.dest, half);

                        noc_async_read(get_noc_addr(from), to, run.length * 32);
                    }
                }
            }
        }
    }

    noc_async_read_barrier();

    destination.push_back(group);
}
}



void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(9), count = get_arg_val<uint32_t>(10);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(args, get_arg_val<uint32_t>(0), bytes);


    const uint32_t lengths[8] = {
        input_channel_tiles * hidden_channel_tiles,
        hidden_channel_tiles,
        9 * hidden_channel_tiles * hidden_channel_tiles,
        hidden_channel_tiles,
        hidden_channel_tiles * output_channel_tiles,
        output_channel_tiles,
        has_downsample ? input_channel_tiles * output_channel_tiles : 0,
        has_downsample ? output_channel_tiles : 0
    };
    DataflowBuffer xc(22), pc(23), c1(20), im2col(11), shortcut(9), zero(24);
    Noc noc;

    zero.reserve_back(1);
    auto z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
    for (uint32_t i = 0; i < 512; ++i) {
        z[i] = 0;
    }
    zero.push_back(1);


    pc.reserve_back(parameter_tiles);
    {
        uint32_t base = 0;
        for (uint32_t tensor = 0; tensor < 8; ++tensor) {
            if (lengths[tensor]) {
                const auto accessor =
                    TensorAccessor(args, get_arg_val<uint32_t>(tensor + 1), bytes);
                for (uint32_t page = 0; page < lengths[tensor]; ++page) {
                    noc.async_read(
                        accessor,
                        pc,
                        bytes,
                        {.page_id = page},
                        {.offset_bytes = (base + page) * bytes}
                    );
                }
            }
            base += lengths[tensor];
        }
        ASSERT(base <= parameter_tiles);
        noc.async_read_barrier();
        pc.push_back(parameter_tiles);
    }


    uint32_t previous_end = 0;
    for (uint32_t done = 0; done < count; done += block_tiles) {
        const auto p = patch(geometry, start + done, min_u(block_tiles, count - done));
        const uint32_t end = p.input_mt_begin + p.input_mt_count;
        const uint32_t first_new =
            p.input_mt_begin > previous_end ? p.input_mt_begin : previous_end;
        xc.reserve_back(input_ring_tiles * input_channel_tiles);

        for (uint32_t mt = first_new; mt < end; ++mt) {
            for (uint32_t ct = 0; ct < input_channel_tiles; ++ct) {
                noc.async_read(
                    x,
                    xc,
                    bytes,
                    {.page_id = mt * input_channel_tiles + ct},
                    {.offset_bytes = ((mt % input_ring_tiles) * input_channel_tiles + ct) * bytes}
                );
            }
        }
        noc.async_read_barrier();
        xc.push_back(input_ring_tiles * input_channel_tiles);


        c1.wait_front(input_ring_tiles * hidden_channel_tiles);
        for (uint32_t m = 0; m < p.output_mt_count; m += 2) {
            gather(im2col, c1, zero, p, m, hidden_channel_tiles, false);
        }
        c1.pop_front(input_ring_tiles * hidden_channel_tiles);


        if constexpr (stride == 2) {
            for (uint32_t m = 0; m < p.output_mt_count; m += 2) {
                gather(shortcut, xc, zero, p, m, input_channel_tiles, true);
            }
        }
        previous_end = end;
    }
    zero.pop_front(1);
}
