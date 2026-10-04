




#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "sharded_common.hpp"

namespace {
using namespace bottleneck_sharded;




struct Peers {
    uint32_t lane;
    uint32_t x[n_shards], y[n_shards];
    uint32_t epoch = 0;
    uint32_t arrival_addr = get_semaphore(get_named_compile_time_arg_val("arrival_sem"));
    uint32_t release_addr = get_semaphore(get_named_compile_time_arg_val("release_sem"));


    explicit Peers(uint32_t id) : lane(id) {
        for (uint32_t n = 0; n < n_shards; ++n) {
            x[n] = get_arg_val<uint32_t>(reader_peer_args + 2 * n);
            y[n] = get_arg_val<uint32_t>(reader_peer_args + 2 * n + 1);
        }
    }

    uint64_t address(uint32_t owner, uint32_t offset) const {
        return get_noc_addr(x[owner], y[owner], offset);
    }





    void barrier() {
        ++epoch;
        if constexpr (n_shards > 1) {
            if (lane == 0) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr),
                    epoch * (n_shards - 1)
                );
                for (uint32_t n = 1; n < n_shards; ++n) {
                    noc_semaphore_inc(address(n, release_addr), 1);
                }
                noc_async_atomic_barrier();
            } else {
                noc_semaphore_inc(address(0, arrival_addr), 1);
                noc_async_atomic_barrier();
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release_addr), epoch
                );
            }
        }
    }
};



struct Run {
    uint32_t dest, source, length;
    bool valid;
};


uint32_t make_runs(uint32_t output_mt, uint32_t tap, bool shortcut, Run* runs) {
    const uint32_t ow = geometry.out_width(), area = geometry.out_height() * ow;
    uint32_t count = 0;
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t position = output_mt * 32 + r;
        const uint32_t image = position / area, local = position % area;
        const int32_t y =
            int32_t((local / ow) * stride + (shortcut ? 0 : tap / 3)) - (shortcut ? 0 : 1);
        const int32_t x =
            int32_t((local % ow) * stride + (shortcut ? 0 : tap % 3)) - (shortcut ? 0 : 1);
        const bool valid = position < geometry.output_rows() && y >= 0 && x >= 0 &&
                           y < int32_t(geometry.height) && x < int32_t(geometry.width);
        const uint32_t source =
            valid ? (image * geometry.height + uint32_t(y)) * geometry.width + uint32_t(x) : 0;
        if (count) {
            auto& last = runs[count - 1];
            const bool adjacent = valid && last.valid && source == last.source + last.length &&
                                  source / 16 == last.source / 16;
            if (last.dest / 16 == r / 16 && (adjacent || (!valid && !last.valid))) {
                ++last.length;
                continue;
            }
        }
        runs[count++] = {r, source, 1, valid};
    }
    return count;
}


void copy_zero(uint32_t source, uint32_t dest) {
    noc_async_read(get_noc_addr(source), dest, tile_bytes);
}




template <typename Accessor>
void preload(
    const Accessor& weight,
    const Accessor& bias,
    uint32_t base,
    uint32_t weight_offset,
    uint32_t bias_offset,
    uint32_t k_tiles,
    uint32_t global_n,
    uint32_t n_begin,
    uint32_t n_count,
    uint32_t local_stride
) {
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_count; ++n) {
            noc_async_read(
                weight.get_noc_addr(k * global_n + n_begin + n),
                base + (weight_offset + k * local_stride + n) * tile_bytes,
                tile_bytes
            );
        }
    }
    for (uint32_t n = 0; n < n_count; ++n) {
        noc_async_read(
            bias.get_noc_addr(n_begin + n), base + (bias_offset + n) * tile_bytes, tile_bytes
        );
    }
}



template <typename Accessor>
void read_input(
    const Accessor& x,
    DataflowBuffer& operand,
    DataflowBuffer& zero,
    uint32_t mt_begin,
    uint32_t valid_m
) {
    operand.reserve_back(operand_slot);

    const uint32_t dest = operand.get_write_ptr();
    for (uint32_t m = 0; m < processing; ++m) {
        for (uint32_t c = 0; c < cin; ++c) {
            const uint32_t to = dest + (m * cin + c) * tile_bytes;
            if (m < valid_m) {
                noc_async_read(x.get_noc_addr((mt_begin + m) * cin + c), to, tile_bytes);
            } else {
                copy_zero(zero.get_read_ptr(), to);
            }
        }
    }
    noc_async_read_barrier();
    operand.push_back(operand_slot);
}





template <typename Accessor>
void gather_window(
    const Accessor& x,
    DataflowBuffer& operand,
    DataflowBuffer& a,
    DataflowBuffer& zero,
    const Patch& p,
    uint32_t mt,
    bool shortcut,
    const Peers& peers
) {
    operand.reserve_back(operand_slot);

    const uint32_t dest = operand.get_write_ptr(), src = a.get_read_ptr(), z = zero.get_read_ptr();
    const uint32_t channels = shortcut ? cin : hidden, taps = shortcut ? 1 : 9;
    const uint32_t valid_m = min_u(processing, p.output_mt_count - mt);
    for (uint32_t m = 0; m < processing; ++m) {
        for (uint32_t tap = 0; tap < taps; ++tap) {
            Run runs[32];
            uint32_t nr = 0;
            if (shortcut) {
                nr = m < valid_m ? make_runs(p.output_mt_begin + mt + m, tap, true, runs) : 0;
            } else {
                nr = m < valid_m ? make_runs(p.output_mt_begin + mt + m, tap, false, runs) : 0;
            }
            for (uint32_t c = 0; c < channels; ++c) {
                const uint32_t tile = dest + ((m * taps + tap) * channels + c) * tile_bytes;
                if (m >= valid_m) {
                    copy_zero(z, tile);
                    continue;
                }


                const uint32_t owner = shortcut ? 0 : channel_owner(c, hidden);
                uint32_t owner_begin = 0, owner_count = 0;
                if (!shortcut) {
                    core_work(hidden, n_shards, owner, owner_begin, owner_count);
                }
                for (uint32_t j = 0; j < nr; ++j) {
                    const auto& run = runs[j];
                    if (run.valid && !shortcut) {
                        ASSERT(run.source / 32 >= p.input_mt_begin);
                        ASSERT(run.source / 32 < p.input_mt_begin + p.input_mt_count);
                        ASSERT(c - owner_begin < owner_count);
                    }
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t offset = row_offset(run.source % 32, half),
                                       length = run.length * 32;
                        uint64_t from = get_noc_addr(z);
                        if (run.valid) {
                            if (shortcut) {
                                from = x.get_noc_addr((run.source / 32) * cin + c, offset);
                            } else {
                                const uint32_t page =
                                    (run.source / 32 - p.input_mt_begin) * local_hidden + c -
                                    owner_begin;
                                from = peers.address(owner, src + page * tile_bytes + offset);
                            }
                        }
                        noc_async_read(from, tile + row_offset(run.dest, half), length);
                    }
                }
            }
        }
    }

    noc_async_read_barrier();

    operand.push_back(operand_slot);
}




void receive_a2(
    DataflowBuffer& b,
    DataflowBuffer& full_b,
    DataflowBuffer& zero,
    uint32_t valid_m,
    const Peers& peers
) {
    full_b.reserve_back(full_b_tiles);

    const uint32_t src = b.get_read_ptr(), dest = full_b.get_write_ptr();
    for (uint32_t m = 0; m < processing; ++m) {
        for (uint32_t c = 0; c < hidden; ++c) {
            const uint32_t to = dest + (m * hidden + c) * tile_bytes;
            if (m >= valid_m) {
                copy_zero(zero.get_read_ptr(), to);
                continue;
            }
            const uint32_t owner = channel_owner(c, hidden);
            uint32_t begin, count;
            core_work(hidden, n_shards, owner, begin, count);
            ASSERT(c - begin < count);
            noc_async_read(
                peers.address(owner, src + (m * local_hidden + c - begin) * tile_bytes),
                to,
                tile_bytes
            );
        }
    }
    noc_async_read_barrier();
    full_b.push_back(full_b_tiles);
}
}




void kernel_main() {
    using namespace bottleneck_sharded;

    const uint32_t start = get_arg_val<uint32_t>(9), count = get_arg_val<uint32_t>(10);
    const uint32_t lane = get_arg_val<uint32_t>(11), hbegin = get_arg_val<uint32_t>(12),
                   hcount = get_arg_val<uint32_t>(13);
    const uint32_t obegin = get_arg_val<uint32_t>(14), ocount = get_arg_val<uint32_t>(15);
    Peers peers(lane);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto weight0 = TensorAccessor(args, get_arg_val<uint32_t>(1), tile_bytes);
    const auto bias0 = TensorAccessor(args, get_arg_val<uint32_t>(2), tile_bytes);
    const auto weight1 = TensorAccessor(args, get_arg_val<uint32_t>(3), tile_bytes);
    const auto bias1 = TensorAccessor(args, get_arg_val<uint32_t>(4), tile_bytes);
    const auto weight2 = TensorAccessor(args, get_arg_val<uint32_t>(5), tile_bytes);
    const auto bias2 = TensorAccessor(args, get_arg_val<uint32_t>(6), tile_bytes);
    const auto weightd = TensorAccessor(args, get_arg_val<uint32_t>(7), tile_bytes);
    const auto biasd = TensorAccessor(args, get_arg_val<uint32_t>(8), tile_bytes);
    DataflowBuffer operand(cb_operand), params(cb_parameters), a(cb_activation_a),
        b(cb_activation_b), full_b(cb_full_b), zero(cb_zero);

    zero.reserve_back(1);
    auto z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
    for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
        z[i] = 0;
    }
    zero.push_back(1);



    params.reserve_back(parameter_tiles);
    const uint32_t base = params.get_write_ptr();

    preload(weight0, bias0, base, w0, b0, cin, hidden, hbegin, hcount, local_hidden);
    preload(weight1, bias1, base, w1, b1, 9 * hidden, hidden, hbegin, hcount, local_hidden);
    preload(weight2, bias2, base, w2, b2, hidden, cout, obegin, ocount, local_cout);
    if constexpr (has_downsample) {
        preload(weightd, biasd, base, wd, bd, cin, cout, obegin, ocount, local_cout);
    }

    noc_async_read_barrier();

    params.push_back(parameter_tiles);
    for (uint32_t done = 0; done < count; done += block) {
        const auto p = patch(geometry, start + done, min_u(block, count - done));
        ASSERT(p.input_mt_count <= patch_input_tiles);
        for (uint32_t mt = 0; mt < p.input_mt_count; mt += processing) {
            read_input(
                x, operand, zero, p.input_mt_begin + mt, min_u(processing, p.input_mt_count - mt)
            );
        }




        a.wait_front(activation_a_tiles);
        peers.barrier();

        for (uint32_t mt = 0; mt < p.output_mt_count; mt += processing) {
            const uint32_t valid_m = min_u(processing, p.output_mt_count - mt);

            gather_window(x, operand, a, zero, p, mt, false, peers);




            b.wait_front(activation_b_tiles);

            peers.barrier();

            receive_a2(b, full_b, zero, valid_m, peers);

            peers.barrier();

            b.pop_front(activation_b_tiles);
            gather_window(x, operand, a, zero, p, mt, true, peers);
        }

        peers.barrier();



        a.pop_front(activation_a_tiles);
    }
    noc_async_atomic_barrier();
    zero.pop_front(1);
}
