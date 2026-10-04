




#include "backward_device_operation.hpp"
#include "host_common.hpp"
#include "l1_memory.hpp"
#include "kernels/backward_common.hpp"
#include "ttnn/device_operation.hpp"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace ttnn::operations::basic_block::backward_detail {
using namespace tt::tt_metal;
using bottleneck_backward::Geometry;
using bottleneck_backward::Mode;




struct BackwardStage {
    struct operation_attributes_t {
        uint32_t mode, batch, height, width, cin, cout, kernel, stride, padding;
        uint32_t rows, cols, cores;
        uint64_t l1_available, l1_reserved;
        bool relu;
    };
    struct tensor_args_t {
        const Tensor& a;
        const Tensor& b;
        const Tensor& c;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t
    compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static Tensor create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ProgramDescriptor
    create_descriptor(const operation_attributes_t&, const tensor_args_t&, Tensor&);
};



void check_matrix(const Tensor& tensor, const Tensor& device_source, uint32_t rows, uint32_t cols) {
    detail::validate_matrix(tensor, device_source, rows, cols);
    const auto& shape = tensor.padded_shape();
    TT_FATAL(shape.size() >= 2, "Backward expects a packed matrix tensor");
    TT_FATAL(
        shape[shape.size() - 2] == detail::padded(rows) &&
            shape[shape.size() - 1] == detail::padded(cols),
        "Backward packed shape mismatch: expected padded rows={} cols={}",
        detail::padded(rows),
        detail::padded(cols)
    );
    for (uint32_t i = 0; i + 2 < shape.size(); ++i) {
        TT_FATAL(
            shape[i] == 1, "Backward requires singleton leading dimensions in packed matrices"
        );
    }
}


Geometry geometry(const BackwardStage::operation_attributes_t& a) {
    return {a.batch, a.height, a.width, a.cin, a.cout, a.kernel, a.stride, a.padding};
}




void BackwardStage::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t
) {
    using namespace bottleneck_backward;
    const auto g = geometry(a);
    TT_FATAL(a.mode <= ADD && a.cores, "Invalid backward stage plan");
    const uint32_t wrows = a.kernel * a.kernel * detail::padded(a.cin);
    if (a.mode == FORWARD) {
        check_matrix(t.a, t.a, g.input_rows(), a.cin);
        check_matrix(t.b, t.a, wrows, a.cout);
        check_matrix(t.c, t.a, 32, a.cout);
    } else if (a.mode == INPUT_GRAD) {
        check_matrix(t.a, t.a, g.output_rows(), a.cout);
        check_matrix(t.b, t.a, wrows, a.cout);
        check_matrix(t.c, t.a, g.input_rows(), a.cin);
    } else if (a.mode == WEIGHT_GRAD) {
        check_matrix(t.a, t.a, g.input_rows(), a.cin);
        check_matrix(t.b, t.a, g.output_rows(), a.cout);
        check_matrix(t.c, t.a, wrows, a.cout);
    } else if (a.mode == BIAS_GRAD) {
        check_matrix(t.a, t.a, g.output_rows(), a.cout);
        check_matrix(t.c, t.a, 32, a.cout);
    } else {
        check_matrix(t.a, t.a, a.rows, a.cols);
        check_matrix(t.b, t.a, a.rows, a.cols);
        check_matrix(t.c, t.a, a.rows, a.cols);
    }
}



BackwardStage::spec_return_value_t
BackwardStage::compute_output_specs(const operation_attributes_t& a, const tensor_args_t& t) {
    if (a.mode != bottleneck_backward::FORWARD) {
        return t.c.tensor_spec();
    }
    return detail::matrix_spec(a.rows, detail::padded(a.cols), ttnn::DRAM_MEMORY_CONFIG);
}



Tensor
BackwardStage::create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
    return ttnn::create_device_tensor(compute_output_specs(a, t), t.a.device());
}




ProgramDescriptor BackwardStage::create_descriptor(
    const operation_attributes_t& a, const tensor_args_t& t, Tensor& output
) {
    using namespace bottleneck_backward;
    const auto grid = t.a.device()->compute_with_storage_grid_size();
    const auto cores = detail::core_ranges(a.cores, grid.y);
    const auto memory = detail::query_l1_memory(t.a.device(), cores);
    const uint64_t required = uint64_t(buffer_tiles(a.mode)) * tile_bytes + 4096;
    TT_FATAL(
        memory.available_bytes >= required && memory.reserved_bytes == a.l1_reserved,
        "Backward L1 allocator changed or cannot fit stage: required={} available={}",
        required,
        memory.available_bytes
    );
    ProgramDescriptor p;


    detail::add_cb(p, cb_out, 2, cores);
    if (a.mode == RELU_GRAD) {
        detail::add_cb(p, cb_scratch, 1, cores);
    } else {
        detail::add_cb(p, cb_b, 2, cores);
        if (a.mode != BIAS_GRAD) {
            detail::add_cb(p, cb_a, 2, cores);
        }
        if (a.mode != ADD) {
            detail::add_cb(p, cb_zero, 1, cores);
        }
        if (a.mode == FORWARD) {
            detail::add_cb(p, cb_bias, 1, cores);
        }
        if (a.mode == BIAS_GRAD) {
            detail::add_cb(p, cb_ones, 1, cores);
        }
        if (a.mode == WEIGHT_GRAD) {
            detail::add_cb(p, cb_scratch, 1, cores);
        }
    }
    auto reader = detail::kernel("backward_reader.cpp", cores);
    reader.config = ReaderConfigDescriptor{};
    detail::accessor(reader, t.a);
    reader.named_compile_time_args = {
        {"bwd_mode", a.mode},
        {"bwd_batch", a.batch},
        {"bwd_height", a.height},
        {"bwd_width", a.width},
        {"bwd_cin", a.cin},
        {"bwd_cout", a.cout},
        {"bwd_kernel", a.kernel},
        {"bwd_stride", a.stride},
        {"bwd_padding", a.padding},
        {"bwd_result_rows", a.rows},
        {"bwd_result_cols", a.cols},
        {"bwd_relu", uint32_t(a.relu)},
    };
    auto compute = detail::kernel("backward_compute.cpp", cores);
    compute.config = detail::compute_config();
    compute.named_compile_time_args = reader.named_compile_time_args;
    auto writer = detail::kernel("backward_writer.cpp", cores);
    writer.config = WriterConfigDescriptor{};
    writer.named_compile_time_args = reader.named_compile_time_args;
    detail::accessor(writer, output);


    const uint32_t jobs = detail::tiles(a.rows) * detail::tiles(a.cols);
    for (uint32_t i = 0; i < a.cores; ++i) {
        const auto core = detail::core_at(i, grid.y);
        uint32_t start, count;
        bottleneck_geometry::core_work(jobs, a.cores, i, start, count);
        reader.emplace_runtime_args(
            core,
            {
                t.a.buffer(),
                t.b.buffer(),
                t.c.buffer(),
                start,
                count,
            }
        );
        compute.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                count,
            }
        );
        writer.emplace_runtime_args(
            core,
            {
                output.buffer(),
                start,
                count,
            }
        );
    }
    p.kernels.push_back(std::move(reader));
    if (a.mode != RELU_GRAD) {
        p.kernels.push_back(std::move(compute));
    }
    p.kernels.push_back(std::move(writer));
    return p;
}



Tensor stage(
    uint32_t mode,
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Geometry& g,
    bool relu = false
) {
    using namespace bottleneck_backward;
    TT_FATAL(
        g.batch && g.height && g.width && g.cin && g.cout && (g.kernel == 1 || g.kernel == 3) &&
            (g.stride == 1 || g.stride == 2) && g.padding == g.kernel / 2,
        "Invalid backward convolution geometry"
    );
    TT_FATAL(
        uint64_t(g.batch) * g.height * g.width <= 0x7fffffffULL, "Backward spatial size overflow"
    );


    const uint32_t rows =
        mode == WEIGHT_GRAD
            ? g.kernel * g.kernel * detail::padded(g.cin)
            : (mode == BIAS_GRAD ? 32 : (mode == FORWARD ? g.output_rows() : g.input_rows()));
    const uint32_t cols = mode == INPUT_GRAD ? g.cin : g.cout;
    const uint64_t bytes = uint64_t(detail::padded(rows)) * detail::padded(cols) * 2;
    TT_FATAL(bytes <= 0xffffffffULL, "Backward tensor exceeds supported address range");
    const auto grid = a.device()->compute_with_storage_grid_size();
    const uint32_t cores =
        detail::select_core_count(detail::tiles(rows) * detail::tiles(cols), grid.x * grid.y, 0);
    const auto memory = detail::query_l1_memory(a.device(), detail::core_ranges(cores, grid.y));
    TT_FATAL(
        memory.available_bytes >= uint64_t(buffer_tiles(mode)) * tile_bytes + 4096,
        "Insufficient L1 for backward tile workspace"
    );
    const BackwardStage::operation_attributes_t attrs{
        mode,
        g.batch,
        g.height,
        g.width,
        g.cin,
        g.cout,
        g.kernel,
        g.stride,
        g.padding,
        rows,
        cols,
        cores,
        memory.available_bytes,
        memory.reserved_bytes,
        relu
    };
    return device_operation::launch<BackwardStage>(attrs, BackwardStage::tensor_args_t{a, b, c});
}


struct Gradients {
    Tensor dx, dw, db;
};


Gradients convolution_backward(
    const Tensor& delta,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Geometry& g
) {
    using namespace bottleneck_backward;
    auto dw = stage(WEIGHT_GRAD, input, delta, weight, g);

    auto db = stage(BIAS_GRAD, delta, delta, bias, g);
    auto dx = stage(INPUT_GRAD, delta, weight, input, g);
    return {std::move(dx), std::move(dw), std::move(db)};
}



Tensor relu_gradient(
    const Tensor& gradient,
    const Tensor& activation,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    uint32_t channels
) {
    return stage(
        bottleneck_backward::RELU_GRAD,
        gradient,
        activation,
        gradient,
        {batch, height, width, channels, channels, 1, 1, 0}
    );
}
}

namespace ttnn::prim {



BasicBlockBackwardReturn basic_block_backward(
    const Tensor& grad_out,
    const Tensor& x,
    const std::vector<Tensor>& weights,
    const std::vector<Tensor>& biases,
    uint32_t in_channels,
    uint32_t channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t padding
) {
    namespace bb = operations::basic_block;
    namespace impl = bb::backward_detail;
    using bottleneck_backward::Geometry;
    using namespace bottleneck_backward;
    const auto attrs = bb::detail::attributes(
        in_channels,
        channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding
    );
    bb::BasicBlockDeviceOperation::validate_on_program_cache_miss(attrs, {x, weights, biases});
    const uint32_t oh = bb::detail::output_height(attrs), ow = bb::detail::output_width(attrs);
    impl::check_matrix(grad_out, x, batch_size * oh * ow, out_channels);
    const bool projection = bb::detail::has_downsample(attrs);
    const Geometry g1{batch_size, input_height, input_width, in_channels, channels, 1, 1, 0};
    const Geometry g2{batch_size, input_height, input_width, channels, channels, 3, stride, 1};
    const Geometry g3{batch_size, oh, ow, channels, out_channels, 1, 1, 0};
    const Geometry gd{
        batch_size, input_height, input_width, in_channels, out_channels, 1, stride, 0
    };


    auto a1 = impl::stage(FORWARD, x, weights[0], biases[0], g1, true);
    auto a2 = impl::stage(FORWARD, a1, weights[1], biases[1], g2, true);



    Tensor delta;
    {
        auto forward = basic_block(
            x,
            weights,
            biases,
            in_channels,
            channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            0,
            32,
            128
        );
        delta =
            impl::relu_gradient(grad_out, std::get<0>(forward), batch_size, oh, ow, out_channels);
    }
    std::vector<Tensor> dw(projection ? 4 : 3), db(projection ? 4 : 3);

    Tensor da2;
    {
        auto grad = impl::convolution_backward(delta, a2, weights[2], biases[2], g3);
        dw[2] = std::move(grad.dw);
        db[2] = std::move(grad.db);
        da2 = impl::relu_gradient(grad.dx, a2, batch_size, oh, ow, channels);
    }

    Tensor da1;
    {
        auto grad = impl::convolution_backward(da2, a1, weights[1], biases[1], g2);
        dw[1] = std::move(grad.dw);
        db[1] = std::move(grad.db);
        da1 = impl::relu_gradient(grad.dx, a1, batch_size, input_height, input_width, channels);
    }

    auto first = impl::convolution_backward(da1, x, weights[0], biases[0], g1);
    dw[0] = std::move(first.dw);
    db[0] = std::move(first.db);


    Tensor shortcut_dx = delta;
    if (projection) {
        auto shortcut = impl::convolution_backward(delta, x, weights[3], biases[3], gd);
        dw[3] = std::move(shortcut.dw);
        db[3] = std::move(shortcut.db);
        shortcut_dx = std::move(shortcut.dx);
    }

    auto dx = impl::stage(
        ADD,
        first.dx,
        shortcut_dx,
        x,
        {batch_size, input_height, input_width, in_channels, in_channels, 1, 1, 0}
    );
    return {std::move(dx), std::move(dw), std::move(db)};
}
}
