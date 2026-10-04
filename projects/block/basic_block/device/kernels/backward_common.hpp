#pragma once






#include <cstdint>

namespace bottleneck_backward {


enum Mode : uint32_t {
    FORWARD = 0,
    INPUT_GRAD = 1,
    WEIGHT_GRAD = 2,
    BIAS_GRAD = 3,
    RELU_GRAD = 4,
    ADD = 5
};
constexpr uint32_t tile_bytes = 2048;
constexpr uint32_t cb_a = 0, cb_b = 1, cb_bias = 2, cb_out = 16, cb_zero = 24, cb_ones = 25,
                   cb_scratch = 26;

constexpr uint32_t tiles(uint32_t n) {
    return n / 32 + (n % 32 != 0);
}


constexpr uint32_t tile_offset(uint32_t r, uint32_t c) {
    return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + c % 16;
}
constexpr uint32_t row_offset(uint32_t r, uint32_t half) {
    return ((r / 16) * 2 + half) * 512 + (r % 16) * 32;
}



struct Geometry {
    uint32_t batch, height, width, cin, cout, kernel, stride, padding;
    uint32_t oh() const {
        return (height + 2 * padding - kernel) / stride + 1;
    }
    uint32_t ow() const {
        return (width + 2 * padding - kernel) / stride + 1;
    }
    uint32_t input_rows() const {
        return batch * height * width;
    }
    uint32_t output_rows() const {
        return batch * oh() * ow();
    }



    int32_t source(uint32_t output_position, uint32_t tap) const {
        if (output_position >= output_rows()) {
            return -1;
        }
        const uint32_t image = output_position / (oh() * ow());
        const uint32_t local = output_position % (oh() * ow());
        const int32_t y = int32_t(local / ow() * stride + tap / kernel) - int32_t(padding);
        const int32_t x = int32_t(local % ow() * stride + tap % kernel) - int32_t(padding);
        if (y < 0 || x < 0 || y >= int32_t(height) || x >= int32_t(width)) {
            return -1;
        }
        return int32_t((image * height + uint32_t(y)) * width + uint32_t(x));
    }




    int32_t inverse(uint32_t input_position, uint32_t tap) const {
        if (input_position >= input_rows()) {
            return -1;
        }
        const uint32_t image = input_position / (height * width),
                       local = input_position % (height * width);
        const int32_t y = int32_t(local / width) + int32_t(padding) - int32_t(tap / kernel);
        const int32_t x = int32_t(local % width) + int32_t(padding) - int32_t(tap % kernel);
        if (y < 0 || x < 0 || y % int32_t(stride) || x % int32_t(stride)) {
            return -1;
        }
        const uint32_t oy = uint32_t(y) / stride, ox = uint32_t(x) / stride;
        if (oy >= oh() || ox >= ow()) {
            return -1;
        }
        return int32_t((image * oh() + oy) * ow() + ox);
    }
};



inline uint32_t buffer_tiles(uint32_t mode) {
    if (mode == RELU_GRAD) {
        return 3;
    }
    uint32_t count = 2 + 2;
    if (mode != BIAS_GRAD) {
        count += 2;
    }
    if (mode != ADD) {
        count += 1;
    }
    if (mode == FORWARD || mode == BIAS_GRAD || mode == WEIGHT_GRAD) {
        count += 1;
    }
    return count;
}



#ifdef BB_BACKWARD_KERNEL
constexpr uint32_t mode = get_named_compile_time_arg_val("bwd_mode");
constexpr Geometry geometry{
    get_named_compile_time_arg_val("bwd_batch"),
    get_named_compile_time_arg_val("bwd_height"),
    get_named_compile_time_arg_val("bwd_width"),
    get_named_compile_time_arg_val("bwd_cin"),
    get_named_compile_time_arg_val("bwd_cout"),
    get_named_compile_time_arg_val("bwd_kernel"),
    get_named_compile_time_arg_val("bwd_stride"),
    get_named_compile_time_arg_val("bwd_padding")
};
constexpr uint32_t cin_tiles = tiles(geometry.cin), cout_tiles = tiles(geometry.cout);
constexpr uint32_t taps = geometry.kernel * geometry.kernel;
constexpr bool apply_relu = get_named_compile_time_arg_val("bwd_relu") != 0;
constexpr uint32_t result_rows = get_named_compile_time_arg_val("bwd_result_rows");
constexpr uint32_t result_cols = get_named_compile_time_arg_val("bwd_result_cols");
constexpr uint32_t result_nt = tiles(result_cols);
#endif
}
