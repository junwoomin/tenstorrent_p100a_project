#include "device_operation.hpp"
#include "host_common.hpp"
#include "timing_host.hpp"
#include "l1_memory.hpp"
#include <cstdio>
#include <utility>

namespace ttnn::operations::basic_block {
using namespace tt::tt_metal;

ProgramDescriptor BasicBlockDeviceOperation::create_sharded_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensors, tensor_return_value_t& outputs) {
    const auto grid = tensors.x.device()->compute_with_storage_grid_size();
    const auto plan = detail::make_sharded_plan(attrs, grid.x * grid.y);
    TT_FATAL(plan.enabled, "2D N-sharded weights and activations do not fit checked L1 budget\n{}",
             detail::sharded_l1_report(attrs, plan));
    const auto all_cores = detail::core_ranges(plan.cores, grid.y);
    const uint32_t pool = detail::select_core_count(grid.x * grid.y, grid.x * grid.y, attrs.max_cores);
    const auto current = detail::query_l1_memory(tensors.x.device(), detail::core_ranges(pool, grid.y));
    TT_FATAL(current.available_bytes >= plan.required_bytes && current.reserved_bytes == attrs.l1_reserved_bytes,
             "L1 allocator changed during 2D sharded program creation: available={} reserved={}\n{}",
             current.available_bytes, current.reserved_bytes, detail::sharded_l1_report(attrs, plan));
    const uint32_t cin = detail::tiles(attrs.in_channels), hidden = detail::tiles(attrs.channels),
                   cout = detail::tiles(attrs.out_channels), total_m = detail::tiles(detail::output_rows(attrs));
    const bool downsample = detail::has_downsample(attrs);
    if (attrs.l1_debug) std::printf("%s", detail::sharded_l1_report(attrs, plan).c_str());

    ProgramDescriptor program;
    // Identical CB sizes and insertion order give all participating workers the
    // same local CB base addresses, required by peer L1 reads. Balanced N tails
    // occupy padded slots; only their actual local channels are read/computed.
    detail::add_cb(program, 0, plan.operand_slot_tiles * plan.cb_depth, all_cores);
    detail::add_cb(program, 20, plan.ring * plan.local_hidden, all_cores);
    detail::add_cb(program, 21, plan.processing_tiles * plan.local_hidden, all_cores);
    detail::add_cb(program, 22, plan.processing_tiles * hidden, all_cores);
    detail::add_cb(program, 23, plan.parameter_tiles, all_cores);
    detail::add_cb(program, 16, plan.processing_tiles * plan.local_cout * plan.out_depth, all_cores);
    detail::add_cb(program, 24, 1, all_cores);
    if (detail::has_downsample(attrs)) detail::add_cb(program, 25, 1, all_cores);
    // Deduce CoreType from the public descriptor. Some UMD releases do not
    // export it in tt::tt_metal, so an unqualified CoreType::WORKER is fragile.
    using WorkerCoreType = decltype(SemaphoreDescriptor{}.core_type);
    for (uint32_t id : {0u, 1u}) {
        program.semaphores.push_back(SemaphoreDescriptor{
            .id = id, .core_type = WorkerCoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }
    auto reader = detail::kernel("sharded_reader.cpp", all_cores);
    reader.config = ReaderConfigDescriptor{};
    detail::accessor(reader, tensors.x);
    reader.named_compile_time_args = {{"has_downsample", uint32_t(downsample)},
        {"stride", attrs.stride}, {"block_tiles", plan.block}, {"max_input_tiles", plan.ring},
        {"processing_tiles", plan.processing_tiles}, {"operand_slot_tiles", plan.operand_slot_tiles},
        {"parameter_tiles", plan.parameter_tiles}, {"n_shards", plan.n_shards},
        {"cb_depth", plan.cb_depth}, {"out_depth", plan.out_depth},
        {"arrival_sem", 0}, {"release_sem", 1}, {"batch_size", attrs.batch_size},
        {"input_height", attrs.input_height}, {"input_width", attrs.input_width},
        {"in_channels", attrs.in_channels}, {"channels", attrs.channels}, {"out_channels", attrs.out_channels}};
    reader.defines = {{"BB_PROFILE", attrs.profile ? "1" : "0"}, {"BB_AUDIT", attrs.audit ? "1" : "0"}};
    detail::timing_defines(reader, attrs, 2, 0);
    auto compute = detail::kernel("sharded_compute.cpp", all_cores);
    compute.config = detail::compute_config();
    compute.named_compile_time_args = reader.named_compile_time_args;
    compute.defines = reader.defines;
    detail::timing_defines(compute, attrs, 2, 1);
    auto writer = detail::kernel("sharded_writer.cpp", all_cores);
    writer.config = WriterConfigDescriptor{};
    writer.named_compile_time_args = reader.named_compile_time_args;
    writer.defines = reader.defines;
    detail::timing_defines(writer, attrs, 2, 2);
    detail::accessor(writer, outputs[0]);

    for (uint32_t m = 0; m < plan.m_shards; ++m) {
        for (uint32_t n = 0; n < plan.n_shards; ++n) {
            const uint32_t index = m * plan.n_shards + n;
            const auto core = detail::core_at(index, grid.y);
            const auto h = bottleneck_geometry::core_shard(total_m, hidden, plan.m_shards, plan.n_shards, m, n);
            const auto o = bottleneck_geometry::core_shard(total_m, cout, plan.m_shards, plan.n_shards, m, n);
            if (downsample) {
                reader.emplace_runtime_args(core, {tensors.x.buffer(), tensors.weights[0].buffer(),
                    tensors.biases[0].buffer(), tensors.weights[1].buffer(), tensors.biases[1].buffer(),
                    tensors.weights[2].buffer(), tensors.biases[2].buffer(), tensors.weights[3].buffer(),
                    tensors.biases[3].buffer(), h.m_start, h.m_count, n, h.n_start, h.n_count, o.n_start, o.n_count});
            } else {
                reader.emplace_runtime_args(core, {tensors.x.buffer(), tensors.weights[0].buffer(),
                    tensors.biases[0].buffer(), tensors.weights[1].buffer(), tensors.biases[1].buffer(),
                    tensors.weights[2].buffer(), tensors.biases[2].buffer(), 0u, 0u,
                    h.m_start, h.m_count, n, h.n_start, h.n_count, o.n_start, o.n_count});
            }
            // Append scalars only after registering the buffer bindings. Their
            // first nine argument positions remain patchable on cache hits.
            auto& args = reader.runtime_args.back().second;
            for (uint32_t peer = 0; peer < plan.n_shards; ++peer) {
                const auto logical = detail::core_at(m * plan.n_shards + peer, grid.y);
                const auto physical = tensors.x.device()->worker_core_from_logical_core(logical);
                args.push_back(physical.x);
                args.push_back(physical.y);
            }
            TT_FATAL(args.size() <= 255, "2D reader runtime arguments exceed device limit");
            compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{
                h.m_start, h.m_count, h.n_start, h.n_count, o.n_start, o.n_count});
            writer.emplace_runtime_args(core, {outputs[0].buffer(), h.m_start, h.m_count, o.n_start, o.n_count});
            if (attrs.l1_debug) {
                const uint64_t w1 = uint64_t(cin) * h.n_count * 2048,
                               w2 = 9ull * hidden * h.n_count * 2048,
                               w3 = uint64_t(hidden) * o.n_count * 2048,
                               wd = downsample ? uint64_t(cin) * o.n_count * 2048 : 0,
                               bias = uint64_t(2 * h.n_count + o.n_count * (downsample ? 2 : 1)) * 2048;
                std::printf("[BB2D core=%u logical=(%zu,%zu) shard=(M%u,N%u)] "
                    "m_start=%u m_count=%u hidden_n_start=%u hidden_n_count=%u "
                    "out_n_start=%u out_n_count=%u W1=%llu W2=%llu W3=%llu Wd=%llu bias=%llu "
                    "parameter_payload=%llu parameter_capacity=%llu input=%llu activation=%llu "
                    "output=%llu required_L1=%llu available_L1=%llu bytes\n",
                    index, core.x, core.y, m, n, h.m_start, h.m_count, h.n_start, h.n_count, o.n_start, o.n_count,
                    (unsigned long long)w1, (unsigned long long)w2, (unsigned long long)w3,
                    (unsigned long long)wd, (unsigned long long)bias, (unsigned long long)(w1 + w2 + w3 + wd + bias),
                    (unsigned long long)plan.weight_bytes, (unsigned long long)plan.operand_bytes,
                    (unsigned long long)(plan.activation_a_bytes + plan.activation_b_bytes + plan.full_b_bytes),
                    (unsigned long long)plan.output_bytes, (unsigned long long)plan.required_bytes,
                    (unsigned long long)plan.available_bytes);
            }
        }
    }
    if (attrs.l1_debug) std::fflush(stdout);
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(compute));
    program.kernels.push_back(std::move(writer));
    return program;
}
} // namespace ttnn::operations::basic_block
