#include "device_operation.hpp"
#include "host_common.hpp"
#include "timing_host.hpp"
#include "l1_memory.hpp"
#include "resident_memory_report.hpp"
#include <cstdio>
#include <utility>

namespace ttnn::operations::basic_block {
using namespace tt::tt_metal;

ProgramDescriptor BasicBlockDeviceOperation::create_resident_descriptor(const operation_attributes_t &attrs,
                                                               const tensor_args_t &tensors,
                                                               tensor_return_value_t &outputs) {
    const auto grid = tensors.x.device()->compute_with_storage_grid_size();
    const auto plan = detail::make_direct_plan(attrs, grid.x * grid.y);
    TT_FATAL(plan.enabled, "Resident plan exceeds checked L1 budget");
    const uint32_t input_channel_tiles = detail::tiles(attrs.in_channels),
                   hidden_channel_tiles = detail::tiles(attrs.channels),
                   output_channel_tiles = detail::tiles(attrs.out_channels);
    const auto all_cores = detail::core_ranges(plan.cores, grid.y);
    const uint32_t pool = detail::select_core_count(grid.x * grid.y, grid.x * grid.y, attrs.max_cores);
    const auto current = detail::query_l1_memory(tensors.x.device(), detail::core_ranges(pool, grid.y));
    TT_FATAL(current.available_bytes >= uint64_t(plan.l1_tiles) * 2048 &&
                 current.reserved_bytes == attrs.l1_reserved_bytes,
             "L1 allocator changed during resident program creation: required_l1={} available_l1={}\n{}",
             uint64_t(plan.l1_tiles) * 2048, current.available_bytes, detail::resident_l1_report(attrs, plan));
    if (attrs.l1_debug) {
        std::printf("%s", detail::resident_l1_report(attrs, plan).c_str());
        std::printf("Allocator: total=%uB firmware/config_base=%uB persistent_end=%uB L1_SMALL=%uB occupied_cap=%uB\n",
                    current.total_bytes, current.static_base_bytes, current.persistent_end_bytes,
                    current.l1_small_bytes, current.occupied_cap_bytes);
        std::fflush(stdout);
    }
    ProgramDescriptor program;
    detail::add_cb(program, 11, 18 * plan.queue_depth * hidden_channel_tiles,
                   all_cores); // adaptive number of 2M groups
    detail::add_cb(program, 16, 2 * plan.queue_depth * output_channel_tiles, all_cores);
    if (attrs.stride == 2) {
        detail::add_cb(program, 9, 2 * plan.queue_depth * input_channel_tiles, all_cores);
    }
    detail::add_cb(program, 24, 1, all_cores); // immutable zero source for invalid spatial runs
    // 작은 블록: 입력, 중간 결과, 파라미터를 각 코어의 L1에 보관한다.
    detail::add_cb(program, 20, plan.ring * hidden_channel_tiles, all_cores);
    detail::add_cb(program, 21, plan.block * hidden_channel_tiles, all_cores);
    detail::add_cb(program, 22, plan.ring * input_channel_tiles, all_cores);
    detail::add_cb(program, 23, plan.parameters, all_cores);
    auto reader = detail::kernel("resident_reader.cpp", all_cores);
    reader.config = ReaderConfigDescriptor{};
    // Every input is validated interleaved DRAM with a 2048-byte page.
    detail::accessor(reader, tensors.x);
    reader.named_compile_time_args = {{"has_downsample", uint32_t(detail::has_downsample(attrs))},
                                      {"stride", attrs.stride},
                                      {"block_tiles", plan.block},
                                      {"max_input_tiles", plan.ring},
                                      {"parameter_tiles", plan.parameters},
                                      {"batch_size", attrs.batch_size},
                                      {"input_height", attrs.input_height},
                                      {"input_width", attrs.input_width},
                                      {"in_channels", attrs.in_channels},
                                      {"channels", attrs.channels},
                                      {"out_channels", attrs.out_channels}};
    reader.defines = {{"BB_PROFILE", attrs.profile ? "1" : "0"}, {"BB_AUDIT", attrs.audit ? "1" : "0"}};
    detail::timing_defines(reader, attrs, 1, 0);
    auto compute = detail::kernel("resident_compute.cpp", all_cores);
    compute.config = detail::compute_config();
    compute.named_compile_time_args = reader.named_compile_time_args;
    compute.defines = reader.defines;
    detail::timing_defines(compute, attrs, 1, 1);
    auto writer = detail::kernel("resident_writer.cpp", all_cores);
    writer.config = WriterConfigDescriptor{};
    writer.named_compile_time_args = reader.named_compile_time_args;
    writer.defines = reader.defines;
    detail::timing_defines(writer, attrs, 1, 2);
    detail::accessor(writer, outputs[0]);
    const uint32_t output_row_tiles = detail::tiles(detail::output_rows(attrs));
    for (uint32_t i = 0; i < plan.cores; ++i) {
        const auto core = detail::core_at(i, grid.y);
        uint32_t start, count;
        bottleneck_geometry::core_work(output_row_tiles, plan.cores, i, start, count);
        if (detail::has_downsample(attrs)) {
            reader.emplace_runtime_args(core, {tensors.x.buffer(), tensors.weights[0].buffer(),
                                               tensors.biases[0].buffer(), tensors.weights[1].buffer(),
                                               tensors.biases[1].buffer(), tensors.weights[2].buffer(),
                                               tensors.biases[2].buffer(), tensors.weights[3].buffer(),
                                               tensors.biases[3].buffer(), start, count});
        } else {
            reader.emplace_runtime_args(core, {tensors.x.buffer(), tensors.weights[0].buffer(),
                                               tensors.biases[0].buffer(), tensors.weights[1].buffer(),
                                               tensors.biases[1].buffer(), tensors.weights[2].buffer(),
                                               tensors.biases[2].buffer(), 0u, 0u, start, count});
        }
        compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{start, count});
        writer.emplace_runtime_args(
            core, {outputs[0].buffer(), count * output_channel_tiles, start * output_channel_tiles});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(compute));
    program.kernels.push_back(std::move(writer));
    return program;
}
} // namespace ttnn::operations::basic_block
