




#include "device_operation.hpp"
#include "host_common.hpp"
#include "l1_memory.hpp"
#include <utility>

namespace ttnn::operations::basic_block {
using namespace tt::tt_metal;



ProgramDescriptor BasicBlockDeviceOperation::create_stream_descriptor(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensors,
    tensor_return_value_t& outputs
) {


    const auto grid = tensors.x.device()->compute_with_storage_grid_size();
    const auto plan = detail::make_stream_plan(attrs, grid.x * grid.y);
    const auto all_cores = detail::core_ranges(plan.cores, grid.y);
    const uint32_t pool =
        detail::select_core_count(grid.x * grid.y, grid.x * grid.y, attrs.max_cores);
    const auto current =
        detail::query_l1_memory(tensors.x.device(), detail::core_ranges(pool, grid.y));
    TT_FATAL(
        current.available_bytes >= plan.required_bytes &&
            current.reserved_bytes == attrs.l1_reserved_bytes,
        "L1 allocator changed during basic_block program creation: current_available={} "
        "current_reserved={}",
        current.available_bytes,
        current.reserved_bytes
    );
    TT_FATAL(
        plan.enabled,
        "basic_block strict L1 plan failed: required_l1={} available_l1={}",
        plan.required_bytes,
        plan.available_bytes
    );

    const uint32_t hidden = detail::tiles(attrs.channels), cout = detail::tiles(attrs.out_channels);




    ProgramDescriptor program;

    detail::add_cb(program, 0, plan.operand_slot_tiles * plan.cb_depth, all_cores);
    detail::add_cb(program, 20, plan.ring * hidden, all_cores);
    detail::add_cb(program, 21, plan.processing_tiles * hidden, all_cores);
    detail::add_cb(program, 23, plan.weight_slot_tiles * plan.weight_depth, all_cores);
    detail::add_cb(program, 16, plan.processing_tiles * cout * plan.out_depth, all_cores);
    detail::add_cb(program, 24, 1, all_cores);



    auto reader = detail::kernel("reader.cpp", all_cores);
    reader.config = ReaderConfigDescriptor{};
    detail::accessor(reader, tensors.x);
    reader.named_compile_time_args = {
        {"has_downsample", uint32_t(detail::has_downsample(attrs))},
        {"stride", attrs.stride},
        {"block_tiles", plan.block},
        {"max_input_tiles", plan.ring},
        {"processing_tiles", plan.processing_tiles},
        {"operand_slot_tiles", plan.operand_slot_tiles},
        {"weight_chunk", plan.weight_chunk},
        {"weight_slot_tiles", plan.weight_slot_tiles},
        {"cb_depth", plan.cb_depth},
        {"weight_depth", plan.weight_depth},
        {"out_depth", plan.out_depth},
        {"batch_size", attrs.batch_size},
        {"input_height", attrs.input_height},
        {"input_width", attrs.input_width},
        {"in_channels", attrs.in_channels},
        {"channels", attrs.channels},
        {"out_channels", attrs.out_channels},
    };



    auto compute = detail::kernel("compute.cpp", all_cores);
    compute.config = detail::compute_config();
    compute.named_compile_time_args = reader.named_compile_time_args;



    auto writer = detail::kernel("writer.cpp", all_cores);
    writer.config = WriterConfigDescriptor{};
    writer.named_compile_time_args = reader.named_compile_time_args;

    detail::accessor(writer, outputs[0]);




    const uint32_t output_row_tiles = detail::tiles(detail::output_rows(attrs));
    for (uint32_t i = 0; i < plan.cores; ++i) {
        const auto core = detail::core_at(i, grid.y);
        uint32_t start, count;
        bottleneck_geometry::core_work(output_row_tiles, plan.cores, i, start, count);
        if (detail::has_downsample(attrs)) {
            reader.emplace_runtime_args(
                core,
                {
                    tensors.x.buffer(),
                    tensors.weights[0].buffer(),
                    tensors.biases[0].buffer(),
                    tensors.weights[1].buffer(),
                    tensors.biases[1].buffer(),
                    tensors.weights[2].buffer(),
                    tensors.biases[2].buffer(),
                    tensors.weights[3].buffer(),
                    tensors.biases[3].buffer(),
                    start,
                    count,
                }
            );
        } else {
            reader.emplace_runtime_args(
                core,
                {
                    tensors.x.buffer(),
                    tensors.weights[0].buffer(),
                    tensors.biases[0].buffer(),
                    tensors.weights[1].buffer(),
                    tensors.biases[1].buffer(),
                    tensors.weights[2].buffer(),
                    tensors.biases[2].buffer(),
                    0u,
                    0u,
                    start,
                    count,
                }
            );
        }
        compute.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                start,
                count,
            }
        );
        writer.emplace_runtime_args(
            core,
            {
                outputs[0].buffer(),
                count * cout,
                start * cout,
            }
        );
    }




    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(compute));
    program.kernels.push_back(std::move(writer));
    return program;
}
}
