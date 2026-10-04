




#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "block_geometry.hpp"
using namespace ckernel;
using namespace bottleneck_geometry;

namespace {
constexpr bool has_downsample = get_named_compile_time_arg_val("has_downsample") != 0;
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


constexpr uint32_t w0 = 0, b0 = input_channel_tiles * hidden_channel_tiles,
                   w1 = b0 + hidden_channel_tiles,
                   b1 = w1 + 9 * hidden_channel_tiles * hidden_channel_tiles,
                   w2 = b1 + hidden_channel_tiles,
                   b2 = w2 + hidden_channel_tiles * output_channel_tiles,
                   wd = b2 + output_channel_tiles,
                   bd = wd + input_channel_tiles * output_channel_tiles;



void biases(uint32_t base, uint32_t n, uint32_t nm, uint32_t nn) {
    reconfig_data_format_srca(23);
    copy_init(23);
    for (uint32_t m = 0; m < nm; ++m) {
        for (uint32_t j = 0; j < nn; ++j) {
            copy_tile(23, base + n + j, m * nn + j);
        }
    }
}


void relu_group(uint32_t n) {
    relu_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        relu_tile(i);
    }
}




void conv1(uint32_t begin, uint32_t end) {
    pack_reconfig_data_format(20);
    for (uint32_t mt = begin; mt < end;) {
        const uint32_t nm = min_u(min_u(2, end - mt), input_ring_tiles - mt % input_ring_tiles);
        for (uint32_t n = 0; n < hidden_channel_tiles; n += 2) {
            const uint32_t nn = min_u(2, hidden_channel_tiles - n);
            tile_regs_acquire();
            biases(b0, n, nm, nn);
            reconfig_data_format<SrcOrder::Reverse>(22, 23);
            matmul_block_init(22, 23, 0, nn, nm, input_channel_tiles);

            for (uint32_t k = 0; k < input_channel_tiles; ++k) {
                matmul_block(
                    22,
                    23,
                    (mt % input_ring_tiles) * input_channel_tiles + k,
                    w0 + k * hidden_channel_tiles + n,
                    0,
                    0,
                    nn,
                    nm,
                    input_channel_tiles
                );
            }
            relu_group(nm * nn);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t m = 0; m < nm; ++m) {
                for (uint32_t j = 0; j < nn; ++j) {
                    pack_tile<true>(
                        m * nn + j, 20, ((mt + m) % input_ring_tiles) * hidden_channel_tiles + n + j
                    );
                }
            }
            tile_regs_release();
        }
        mt += nm;
    }
}



void conv2(uint32_t mt_count) {
    DataflowBuffer input(11);
    pack_reconfig_data_format(21);
    for (uint32_t mt = 0; mt < mt_count; mt += 2) {
        const uint32_t nm = min_u(2, mt_count - mt);
        input.wait_front(18 * hidden_channel_tiles);
        for (uint32_t n = 0; n < hidden_channel_tiles; n += 2) {
            const uint32_t nn = min_u(2, hidden_channel_tiles - n);
            tile_regs_acquire();
            biases(b1, n, nm, nn);
            reconfig_data_format<SrcOrder::Reverse>(11, 23);
            matmul_block_init(11, 23, 0, nn, nm, 9 * hidden_channel_tiles);
            for (uint32_t k = 0; k < 9 * hidden_channel_tiles; ++k) {
                matmul_block(
                    11,
                    23,
                    k,
                    w1 + k * hidden_channel_tiles + n,
                    0,
                    0,
                    nn,
                    nm,
                    9 * hidden_channel_tiles
                );
            }
            relu_group(nm * nn);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t m = 0; m < nm; ++m) {
                for (uint32_t j = 0; j < nn; ++j) {
                    pack_tile<true>(m * nn + j, 21, (mt + m) * hidden_channel_tiles + n + j);
                }
            }
            tile_regs_release();
        }
        input.pop_front(18 * hidden_channel_tiles);
    }
}




void conv3(const Patch& p) {
    DataflowBuffer out(16), shortcut(9);
    pack_reconfig_data_format(16);
    for (uint32_t mt = 0; mt < p.output_mt_count; mt += 2) {
        const uint32_t nm = min_u(2, p.output_mt_count - mt);
        if constexpr (stride == 2) {
            shortcut.wait_front(2 * input_channel_tiles);
        }
        out.reserve_back(2 * output_channel_tiles);
        for (uint32_t n = 0; n < output_channel_tiles; ++n) {
            tile_regs_acquire();
            reconfig_data_format_srca(23);
            copy_init(23);
            for (uint32_t m = 0; m < nm; ++m) {
                copy_tile(23, b2 + n, m);
                if constexpr (has_downsample) {
                    copy_tile(23, bd + n, 2 + m);
                }
            }

            reconfig_data_format<SrcOrder::Reverse>(21, 23);
            matmul_block_init(21, 23, 0, 1, nm, hidden_channel_tiles);
            for (uint32_t k = 0; k < hidden_channel_tiles; ++k) {
                matmul_block(
                    21,
                    23,
                    mt * hidden_channel_tiles + k,
                    w2 + k * output_channel_tiles + n,
                    0,
                    0,
                    1,
                    nm,
                    hidden_channel_tiles
                );
            }

            if constexpr (has_downsample) {
                constexpr uint32_t x = stride == 2 ? 9 : 22;
                const uint32_t first = (p.output_mt_begin + mt) % input_ring_tiles;
                reconfig_data_format<SrcOrder::Reverse>(x, 23);


                if (stride == 2 || first + nm <= input_ring_tiles) {
                    matmul_block_init(x, 23, 0, 1, nm, input_channel_tiles);
                    for (uint32_t k = 0; k < input_channel_tiles; ++k) {
                        matmul_block(
                            x,
                            23,
                            (stride == 2 ? 0 : first * input_channel_tiles) + k,
                            wd + k * output_channel_tiles + n,
                            2,
                            0,
                            1,
                            nm,
                            input_channel_tiles
                        );
                    }
                } else {
                    matmul_init(x, 23);
                    for (uint32_t k = 0; k < input_channel_tiles; ++k) {
                        for (uint32_t m = 0; m < nm; ++m) {
                            matmul_tiles(
                                x,
                                23,
                                ((first + m) % input_ring_tiles) * input_channel_tiles + k,
                                wd + k * output_channel_tiles + n,
                                2 + m
                            );
                        }
                    }
                }
            } else {
                reconfig_data_format_srca(22);
                copy_init(22);
                for (uint32_t m = 0; m < nm; ++m) {
                    copy_tile(
                        22,
                        ((p.output_mt_begin + mt + m) % input_ring_tiles) * input_channel_tiles + n,
                        2 + m
                    );
                }
            }

            add_binary_tile_init();
            for (uint32_t m = 0; m < nm; ++m) {
                add_binary_tile(m, 2 + m, m);
            }
            relu_group(nm);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t m = 0; m < nm; ++m) {
                pack_tile<true>(m, 16, m * output_channel_tiles + n);
            }
            tile_regs_release();
        }
        out.push_back(2 * output_channel_tiles);
        if constexpr (stride == 2) {
            shortcut.pop_front(2 * input_channel_tiles);
        }
    }
}
}




void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    DataflowBuffer x(22), params(23), a1(20), a2(21);

    compute_kernel_hw_startup<SrcOrder::Reverse>(22, 23, 20);
    params.wait_front(parameter_tiles);
    uint32_t previous_end = 0;
    for (uint32_t done = 0; done < count; done += block_tiles) {
        const auto p = patch(geometry, start + done, min_u(block_tiles, count - done));
        const uint32_t end = p.input_mt_begin + p.input_mt_count;
        const uint32_t begin = p.input_mt_begin > previous_end ? p.input_mt_begin : previous_end;
        x.wait_front(input_ring_tiles * input_channel_tiles);

        a1.reserve_back(input_ring_tiles * hidden_channel_tiles);

        conv1(begin, end);

        a1.push_back(input_ring_tiles * hidden_channel_tiles);
        a2.reserve_back(block_tiles * hidden_channel_tiles);

        conv2(p.output_mt_count);

        a2.push_back(block_tiles * hidden_channel_tiles);
        a2.wait_front(block_tiles * hidden_channel_tiles);
        conv3(p);


        a2.pop_front(block_tiles * hidden_channel_tiles);
        x.pop_front(input_ring_tiles * input_channel_tiles);
        previous_end = end;
    }
    params.pop_front(parameter_tiles);
}
