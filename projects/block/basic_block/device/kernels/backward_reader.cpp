




#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#define BB_BACKWARD_KERNEL
#include "backward_common.hpp"

namespace {
using namespace bottleneck_backward;



void fill(DataflowBuffer& cb, uint32_t bits) {
    cb.reserve_back(1);
    auto ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
        ptr[i] = bits;
    }
    cb.push_back(1);
}



void clear_padding(uint32_t address, uint32_t mt, uint32_t nt, uint32_t rows, uint32_t channels) {
    if ((mt + 1) * 32 <= rows && (nt + 1) * 32 <= channels) {
        return;
    }
    auto ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(address);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            if (mt * 32 + r >= rows || nt * 32 + c >= channels) {
                ptr[tile_offset(r, c)] = 0;
            }
        }
    }
}


template <typename Accessor>
void read_matrix(
    const Accessor& src,
    DataflowBuffer& dest,
    uint32_t mt,
    uint32_t nt,
    uint32_t rows,
    uint32_t channels
) {
    dest.reserve_back(1);
    noc_async_read(src.get_noc_addr(mt * tiles(channels) + nt), dest.get_write_ptr(), tile_bytes);
    noc_async_read_barrier();
    clear_padding(dest.get_write_ptr(), mt, nt, rows, channels);
    dest.push_back(1);
}




template <typename Accessor>
void gather_to(
    const Accessor& src,
    uint32_t dest,
    uint32_t zero,
    uint32_t mt,
    uint32_t tap,
    uint32_t nt,
    bool inverse
) {
    const uint32_t source_nt = inverse ? cout_tiles : cin_tiles;
    const uint32_t channels = inverse ? geometry.cout : geometry.cin;
    for (uint32_t row = 0; row < 32;) {
        const int32_t from =
            inverse ? geometry.inverse(mt * 32 + row, tap) : geometry.source(mt * 32 + row, tap);
        uint32_t length = 1;
        while (row + length < 32 && (row + length) / 16 == row / 16) {
            const int32_t next = inverse ? geometry.inverse(mt * 32 + row + length, tap)
                                         : geometry.source(mt * 32 + row + length, tap);
            if (from < 0 ? next >= 0
                         : (next != from + int32_t(length) ||
                            uint32_t(next) / 16 != uint32_t(from) / 16)) {
                break;
            }
            ++length;
        }
        for (uint32_t half = 0; half < 2; ++half) {
            const uint64_t address = from < 0 ? get_noc_addr(zero)
                                              : src.get_noc_addr(
                                                    (uint32_t(from) / 32) * source_nt + nt,
                                                    row_offset(uint32_t(from) % 32, half)
                                                );
            noc_async_read(address, dest + row_offset(row, half), length * 32);
        }
        row += length;
    }
    noc_async_read_barrier();

    clear_padding(dest, 0, nt, 32, channels);
}


template <typename Accessor>
void read_weight(
    const Accessor& src, DataflowBuffer& dest, uint32_t tap, uint32_t ci, uint32_t co
) {
    dest.reserve_back(1);
    noc_async_read(
        src.get_noc_addr((tap * cin_tiles + ci) * cout_tiles + co), dest.get_write_ptr(), tile_bytes
    );
    noc_async_read_barrier();
    clear_padding(dest.get_write_ptr(), ci, co, geometry.cin, geometry.cout);
    dest.push_back(1);
}
}



void kernel_main() {
    using namespace bottleneck_backward;
    constexpr auto accessor_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(accessor_args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto other = TensorAccessor(accessor_args, get_arg_val<uint32_t>(1), tile_bytes);
    const auto bias = TensorAccessor(accessor_args, get_arg_val<uint32_t>(2), tile_bytes);
    const uint32_t start = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);
    uint32_t zero_address = 0, scratch_address = 0;
    if constexpr (mode != RELU_GRAD && mode != ADD) {
        DataflowBuffer zero(cb_zero);
        fill(zero, 0);
        zero_address = zero.get_read_ptr();
    }

    if constexpr (mode == BIAS_GRAD) {
        DataflowBuffer ones(cb_ones);
        fill(ones, 0x3f803f80u);
    }
    if constexpr (mode == RELU_GRAD || mode == WEIGHT_GRAD) {
        DataflowBuffer scratch(cb_scratch);
        scratch.reserve_back(1);
        scratch_address = scratch.get_write_ptr();
    }

    for (uint32_t job = start; job < start + count; ++job) {
        const uint32_t mt = job / result_nt, nt = job % result_nt;


        if constexpr (mode == RELU_GRAD) {
            DataflowBuffer out(cb_out);
            out.reserve_back(1);
            noc_async_read(input.get_noc_addr(job), out.get_write_ptr(), tile_bytes);
            noc_async_read(other.get_noc_addr(job), scratch_address, tile_bytes);
            noc_async_read_barrier();
            invalidate_l1_cache();
            auto grad = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out.get_write_ptr());
            const auto mask = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_address);
            for (uint32_t i = 0; i < 1024; ++i) {
                const uint16_t v = mask[i];
                if ((v & 0x8000u) || !(v & 0x7fffu)) {
                    grad[i] = 0;
                }
            }
            clear_padding(out.get_write_ptr(), mt, nt, result_rows, result_cols);
            out.push_back(1);
        } else if constexpr (mode == ADD) {

            DataflowBuffer a(cb_a), b(cb_b);
            read_matrix(input, a, mt, nt, result_rows, result_cols);
            read_matrix(other, b, mt, nt, result_rows, result_cols);
        } else if constexpr (mode == BIAS_GRAD) {
            DataflowBuffer b(cb_b);
            for (uint32_t m = 0; m < tiles(geometry.output_rows()); ++m) {
                read_matrix(input, b, m, nt, geometry.output_rows(), geometry.cout);
            }
        } else if constexpr (mode == FORWARD) {


            DataflowBuffer a(cb_a), b(cb_b), initial(cb_bias);
            read_matrix(bias, initial, 0, nt, 32, geometry.cout);
            for (uint32_t tap = 0; tap < taps; ++tap) {
                for (uint32_t ci = 0; ci < cin_tiles; ++ci) {
                    a.reserve_back(1);
                    gather_to(input, a.get_write_ptr(), zero_address, mt, tap, ci, false);
                    a.push_back(1);
                    read_weight(other, b, tap, ci, nt);
                }
            }
        } else if constexpr (mode == INPUT_GRAD) {


            DataflowBuffer a(cb_a), b(cb_b);
            for (uint32_t tap = 0; tap < taps; ++tap) {
                for (uint32_t co = 0; co < cout_tiles; ++co) {
                    a.reserve_back(1);
                    gather_to(input, a.get_write_ptr(), zero_address, mt, tap, co, true);
                    a.push_back(1);
                    read_weight(other, b, tap, nt, co);
                }
            }
        } else if constexpr (mode == WEIGHT_GRAD) {



            DataflowBuffer a(cb_a), b(cb_b);
            const uint32_t tap = mt / cin_tiles, ci = mt % cin_tiles;
            for (uint32_t m = 0; m < tiles(geometry.output_rows()); ++m) {
                a.reserve_back(1);
                gather_to(input, scratch_address, zero_address, m, tap, ci, false);
                invalidate_l1_cache();
                const auto src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_address);
                auto dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a.get_write_ptr());

                for (uint32_t r = 0; r < 32; ++r) {
                    for (uint32_t c = 0; c < 32; ++c) {
                        dst[tile_offset(c, r)] = src[tile_offset(r, c)];
                    }
                }
                a.push_back(1);
                read_matrix(other, b, m, nt, geometry.output_rows(), geometry.cout);
            }
        }
    }
}
