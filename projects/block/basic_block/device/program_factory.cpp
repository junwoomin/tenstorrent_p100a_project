#include "device_operation.hpp"
#include "host_common.hpp"
#include "timing_host.hpp"
#include "l1_memory.hpp"
#include <cstdlib>
#include <cstdio>
#include <utility>

#ifndef BB_PROFILE
#define BB_PROFILE 0
#endif
#ifndef BB_AUDIT
#define BB_AUDIT 0
#endif

namespace ttnn::operations::basic_block {
using namespace tt::tt_metal;

ProgramDescriptor BasicBlockDeviceOperation::create_descriptor(
    const operation_attributes_t &attrs,
    const tensor_args_t &tensors,
    tensor_return_value_t &outputs) {

    const auto grid = tensors.x.device()->compute_with_storage_grid_size();

    if (attrs.path_override == 0 || attrs.path_override == 1) {
        const auto resident = detail::make_direct_plan(attrs, grid.x * grid.y);
        if (resident.enabled) {
            if (attrs.l1_debug) {
                std::printf("[BasicBlock v6] path=FULL_BLOCK_RESIDENT M_shards=%u N_shards=1 active_cores=%u\n",
                            resident.cores, resident.cores);
                std::fflush(stdout);
            }
            return create_resident_descriptor(attrs, tensors, outputs);
        }
    }
    if (attrs.path_override == 0 || attrs.path_override == 2) {
        const auto sharded = detail::make_sharded_plan(attrs, grid.x * grid.y);
        if (sharded.enabled) return create_sharded_descriptor(attrs, tensors, outputs);
        TT_FATAL(attrs.path_override != 2,
                 "Forced BASIC_BLOCK_PATH=sharded cannot fit requested geometry/N shards. "
                 "requested_N={} required_l1={} available_l1={}\n{}",
                 attrs.requested_n_shards, sharded.required_bytes, sharded.available_bytes,
                 detail::sharded_l1_report(attrs, sharded));
        if (attrs.l1_debug) {
            std::printf("[BasicBlock v6] 2D resident unavailable; selecting legacy chunk streaming.\n");
        }
    }

    const auto plan = detail::make_stream_plan(attrs, grid.x * grid.y);
    const auto all_cores = detail::core_ranges(plan.cores, grid.y);
    const uint32_t pool = detail::select_core_count(grid.x * grid.y, grid.x * grid.y, attrs.max_cores);
    const auto current = detail::query_l1_memory(tensors.x.device(), detail::core_ranges(pool, grid.y));
    TT_FATAL(current.available_bytes >= plan.required_bytes && current.reserved_bytes == attrs.l1_reserved_bytes,
             "L1 allocator changed during basic_block program creation: current_available={} current_reserved={}\n{}",
             current.available_bytes, current.reserved_bytes, detail::l1_report(attrs, plan));
    TT_FATAL(plan.enabled, "basic_block strict L1 plan failed: required_l1={} available_l1={}\n{}",
             plan.required_bytes, plan.available_bytes, detail::l1_report(attrs, plan));
    if (attrs.l1_debug) {
        std::printf("[BasicBlock v6] path=LEGACY_CHUNK_STREAMING M_shards=%u N_shards=1 active_cores=%u\n",
                    plan.cores, plan.cores);
        std::printf("%s", detail::l1_report(attrs, plan).c_str());
        std::printf("Allocator: total=%uB firmware/config_base=%uB persistent_end=%uB "
                    "L1_SMALL=%uB occupied_cap=%uB alignment=%uB\n",
                    current.total_bytes, current.static_base_bytes, current.persistent_end_bytes,
                    current.l1_small_bytes, current.occupied_cap_bytes, current.alignment_bytes);
        std::fflush(stdout);
    }
    const uint32_t hidden = detail::tiles(attrs.channels), cout = detail::tiles(attrs.out_channels);
    ProgramDescriptor program;
    // Every physical allocation appears once. Input, halo and residual operands
    // reuse CB0 in FIFO order; all stage weights/biases reuse CB23. CB20 persists
    // for one bounded patch, CB21 for one processing group; no full-map tensor.
    detail::add_cb(program, 0, plan.operand_slot_tiles * plan.cb_depth, all_cores);
    detail::add_cb(program, 20, plan.ring * hidden, all_cores);
    detail::add_cb(program, 21, plan.processing_tiles * hidden, all_cores);
    detail::add_cb(program, 23, plan.weight_slot_tiles * plan.weight_depth, all_cores);
    detail::add_cb(program, 16, plan.processing_tiles * cout * plan.out_depth, all_cores);
    detail::add_cb(program, 24, 1, all_cores);
    if (detail::has_downsample(attrs)) detail::add_cb(program, 25, 1, all_cores);

    auto reader = detail::kernel("reader.cpp", all_cores);
    reader.config = ReaderConfigDescriptor{};
    detail::accessor(reader, tensors.x); // All input/weight/bias buffers have the same DRAM tile-page ABI.
    reader.named_compile_time_args = {{"has_downsample", uint32_t(detail::has_downsample(attrs))},
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
                                      {"out_channels", attrs.out_channels}};
    reader.defines = {{"BB_PROFILE", attrs.profile ? "1" : "0"}, {"BB_AUDIT", attrs.audit ? "1" : "0"}};
    detail::timing_defines(reader, attrs, 0, 0);
    auto compute = detail::kernel("compute.cpp", all_cores);
    compute.config = detail::compute_config();
    compute.named_compile_time_args = reader.named_compile_time_args;
    compute.defines = reader.defines;
    detail::timing_defines(compute, attrs, 0, 1);
    auto writer = detail::kernel("writer.cpp", all_cores);
    writer.config = WriterConfigDescriptor{};
    writer.named_compile_time_args = reader.named_compile_time_args;
    writer.defines = reader.defines;
    detail::timing_defines(writer, attrs, 0, 2);
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
        writer.emplace_runtime_args(core, {outputs[0].buffer(), count * cout, start * cout});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(compute));
    program.kernels.push_back(std::move(writer));
    return program;
}
} // namespace ttnn::operations::basic_block
