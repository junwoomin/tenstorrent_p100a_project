#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "stream_common.hpp"
#include "stream_profile.hpp"
#include "timing.hpp"
#include "dram_gather.hpp"

namespace {
using namespace bottleneck_stream;

struct Run {
    uint32_t dest, source, length;
    bool valid;
};

// Coordinate construction occurs once per tap / spatial tile, outside the
// channel loop. Merge contiguous rows within a face into a single NoC request.
uint32_t make_runs(uint32_t output_mt, uint32_t tap, bool shortcut, Run *runs) {
    const uint32_t oh = geometry.out_height(), ow = geometry.out_width(), area = oh * ow;
    uint32_t count = 0;
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t position = output_mt * 32 + r;
        const uint32_t image = position / area, local = position % area;
        const int32_t y = int32_t((local / ow) * stride + (shortcut ? 0 : tap / 3)) - (shortcut ? 0 : 1);
        const int32_t x = int32_t((local % ow) * stride + (shortcut ? 0 : tap % 3)) - (shortcut ? 0 : 1);
        const bool valid = position < geometry.output_rows() && y >= 0 && x >= 0 &&
                           y < int32_t(geometry.height) && x < int32_t(geometry.width);
        const uint32_t source = valid ? (image * geometry.height + uint32_t(y)) * geometry.width + uint32_t(x) : 0;
        if (count) {
            auto &last = runs[count - 1];
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

void zero_tile(uint32_t from, uint32_t to, Traffic &traffic) {
    noc_async_read(get_noc_addr(from), to, tile_bytes);
    traffic.local_read(tile_bytes);
}

// Load scopes exclude FIFO reservation waits and include their existing NoC
// completion barrier. Barrier/plan counters are nested diagnostic subsets.
template <typename Accessor>
void read_input(const Accessor &x, DataflowBuffer &operand, DataflowBuffer &zero,
                uint32_t input_mt, uint32_t valid_m, Traffic &traffic, bb_timing::Counters &timing) {
    BB_TIME(timing, CbWait, operand.reserve_back(operand_slot));
    bb_timing::Scope load_scope(timing, bb_timing::Metric::InputLoad);
    const uint32_t dst = operand.get_write_ptr(), z = zero.get_read_ptr();
    for (uint32_t m = 0; m < processing; ++m) {
        for (uint32_t c = 0; c < cin; ++c) {
            const uint32_t to = dst + (m * cin + c) * tile_bytes;
            if (m < valid_m) {
                noc_async_read(x.get_noc_addr((input_mt + m) * cin + c), to, tile_bytes);
                traffic.input_read(tile_bytes);
            } else {
                zero_tile(z, to, traffic);
            }
        }
    }
    BB_TIME(timing, InputBarrier, noc_async_read_barrier());
    operand.push_back(operand_slot);
}

template <typename Accessor>
void gather_window(const Accessor &x, DataflowBuffer &operand, DataflowBuffer &a,
                   DataflowBuffer &zero, const Patch &p, uint32_t mt, bool shortcut,
                   Traffic &traffic, bb_timing::Counters &timing, uint32_t landing) {
    BB_TIME(timing, CbWait, operand.reserve_back(operand_slot));
    bb_timing::Scope load_scope(timing, shortcut ? bb_timing::Metric::ShortcutLoad : bb_timing::Metric::Routing);
    const uint32_t dst = operand.get_write_ptr(), src = a.get_read_ptr(), z = zero.get_read_ptr();
    const uint32_t channels = shortcut ? cin : hidden, taps = shortcut ? 1 : 9;
    const uint32_t valid_m = min_u(processing, p.output_mt_count - mt);
    for (uint32_t m = 0; m < processing; ++m) {
        for (uint32_t tap = 0; tap < taps; ++tap) {
            Run runs[32];
            uint32_t nr = 0;
            if (shortcut) {
                nr = m < valid_m ? make_runs(p.output_mt_begin + mt + m, tap, true, runs) : 0;
            } else {
                BB_TIME(timing, RoutingPlan,
                        nr = m < valid_m ? make_runs(p.output_mt_begin + mt + m, tap, false, runs) : 0);
            }
            for (uint32_t c = 0; c < channels; ++c) {
                const uint32_t tile = dst + ((m * taps + tap) * channels + c) * tile_bytes;
                if (m >= valid_m) {
                    zero_tile(z, tile, traffic);
                    continue;
                }
                for (uint32_t j = 0; j < nr; ++j) {
                    const auto &run = runs[j];
                    if (run.valid && !shortcut) {
                        ASSERT(run.source / 32 >= p.input_mt_begin);
                        ASSERT(run.source / 32 < p.input_mt_begin + p.input_mt_count);
                    }
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t offset = row_offset(run.source % 32, half);
                        uint64_t from = get_noc_addr(z);
                        if (run.valid) {
                            if (shortcut) {
                                from = x.get_noc_addr((run.source / 32) * cin + c, offset);
                            } else {
                                const uint32_t page = (run.source / 32 - p.input_mt_begin) * hidden + c;
                                from = get_noc_addr(src + page * tile_bytes + offset);
                            }
                        }
                        const uint32_t length = run.length * 32;
                        const uint32_t to = tile + row_offset(run.dest, half);
                        if (run.valid && shortcut) {
                            const auto bytes = bb_dram_gather::read_face_rows(from, to, length, landing);
                            traffic.input_read(bytes.dram);
                            traffic.local_read(bytes.local);
                        } else {
                            noc_async_read(from, to, length);
                            traffic.local_read(length);
                        }
                    }
                }
            }
        }
    }
    {
        bb_timing::Scope barrier_scope(timing, shortcut ? bb_timing::Metric::ShortcutBarrier : bb_timing::Metric::RoutingBarrier);
        noc_async_read_barrier();
    }
    operand.push_back(operand_slot);
}

// For one output-channel tile, stream K in panels. The first tile is the bias;
// weights remain in K order after it. K-tail tiles are never read by compute;
// leave them untouched instead of spending NoC bandwidth initializing padding.
// FIFO prefetch depth overlaps reader work with compute without a stage cache.
template <typename Accessor>
void parameters(const Accessor &weight, const Accessor &bias, DataflowBuffer &params,
                uint32_t k_tiles, uint32_t n_tiles, uint32_t n, Traffic &traffic, bb_timing::Counters &timing) {
    for (uint32_t k0 = 0; k0 < k_tiles; k0 += weight_chunk) {
        BB_TIME(timing, CbWait, params.reserve_back(weight_slot));
        bb_timing::Scope load_scope(timing, bb_timing::Metric::WeightLoad);
        const uint32_t dst = params.get_write_ptr();
        if (k0 == 0) {
            noc_async_read(bias.get_noc_addr(n), dst, tile_bytes);
            traffic.weight_read(tile_bytes);
        }
        for (uint32_t k = 0; k < min_u(weight_chunk, k_tiles - k0); ++k) {
            const uint32_t to = dst + (k + 1) * tile_bytes;
            noc_async_read(weight.get_noc_addr((k0 + k) * n_tiles + n), to, tile_bytes);
            traffic.weight_read(tile_bytes);
        }
        BB_TIME(timing, WeightBarrier, noc_async_read_barrier());
        params.push_back(weight_slot);
    }
}
} // namespace

void kernel_main() {
    bb_timing::Counters timing;
    using namespace bottleneck_stream;
    // X, W1, b1, W2, b2, W3, b3, Wd, bd, spatial_start, spatial_count.
    const uint32_t start = get_arg_val<uint32_t>(9), count = get_arg_val<uint32_t>(10);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto w0 = TensorAccessor(args, get_arg_val<uint32_t>(1), tile_bytes);
    const auto b0 = TensorAccessor(args, get_arg_val<uint32_t>(2), tile_bytes);
    const auto w1 = TensorAccessor(args, get_arg_val<uint32_t>(3), tile_bytes);
    const auto b1 = TensorAccessor(args, get_arg_val<uint32_t>(4), tile_bytes);
    const auto w2 = TensorAccessor(args, get_arg_val<uint32_t>(5), tile_bytes);
    const auto b2 = TensorAccessor(args, get_arg_val<uint32_t>(6), tile_bytes);
    const auto wd = TensorAccessor(args, get_arg_val<uint32_t>(7), tile_bytes);
    const auto bd = TensorAccessor(args, get_arg_val<uint32_t>(8), tile_bytes);
    DataflowBuffer operand(cb_operand), params(cb_parameters), a(cb_activation_a), zero(cb_zero);
    Traffic traffic;
    BB_TIME(timing, CbWait, zero.reserve_back(1));
    auto z = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(zero.get_write_ptr());
    for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
        z[i] = 0;
    }
    zero.push_back(1);
    uint32_t landing = 0;
    if constexpr (has_downsample) {
        DataflowBuffer scratch(bb_dram_gather::scratch_cb);
        scratch.reserve_back(1);
        landing = (scratch.get_write_ptr() + 63) & ~uint32_t(63);
    }


    for (uint32_t done = 0; done < count; done += block) {
        const auto p = patch(geometry, start + done, min_u(block, count - done));
        ASSERT(p.input_mt_count <= patch_input_tiles);
        for (uint32_t mt = 0; mt < p.input_mt_count; mt += processing) {
            const bool sample = done == 0 && mt == 0;
            BB_STREAM_ZONE_IF(sample, "BB_CONV1_INPUT_READ", {
                read_input(x, operand, zero, p.input_mt_begin + mt, min_u(processing, p.input_mt_count - mt), traffic, timing);
            });
            for (uint32_t n = 0; n < hidden; ++n) {
                parameters(w0, b0, params, cin, hidden, n, traffic, timing);
            }
        }
        BB_STREAM_ZONE_IF(done == 0, "BB_CONV2_ACTIVATION_WAIT", BB_TIME(timing, ActivationWait, a.wait_front(activation_a_tiles)));
        for (uint32_t mt = 0; mt < p.output_mt_count; mt += processing) {
            const bool sample = done == 0 && mt == 0;
            // Exactly ONE gathered window per M group, reused for every n.
            BB_STREAM_ZONE_IF(sample, "BB_CONV2_HALO_WINDOW_PREPARE", {
                gather_window(x, operand, a, zero, p, mt, false, traffic, timing, landing);
            });
            for (uint32_t n = 0; n < hidden; ++n) {
                parameters(w1, b1, params, 9 * hidden, hidden, n, traffic, timing);
            }
            // This reuses the operand FIFO after Conv2 consumes its window.
            // The shortcut is read directly from the original tiled input.
            BB_STREAM_ZONE_IF(sample, "BB_SHORTCUT_INPUT_READ", {
                gather_window(x, operand, a, zero, p, mt, true, traffic, timing, landing);
            });
            for (uint32_t n = 0; n < cout; ++n) {
                BB_STREAM_ZONE_IF(sample && n == 0, "BB_CONV3_WEIGHT_PRELOAD", {
                    parameters(w2, b2, params, hidden, cout, n, traffic, timing);
                });
                if constexpr (has_downsample) {
                    parameters(wd, bd, params, cin, cout, n, traffic, timing);
                }
            }
        }
        // Full-depth publication gives compute an explicit reuse handshake.
        // No future patch may overwrite A until all window reads completed.
        a.pop_front(activation_a_tiles);
    }
    if constexpr (has_downsample) {
        // Every staged read/copy completed inside read_face_rows; final gather
        // barrier also completes the direct fast path before reservation release.
        DataflowBuffer scratch(bb_dram_gather::scratch_cb);
        scratch.push_back(1);
        scratch.pop_front(1);
    }
    zero.pop_front(1);
    timing.print("stream", "reader", start, count);
    traffic.print_reader();
}
