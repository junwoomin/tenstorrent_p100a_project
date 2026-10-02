#include "device_operation.hpp"
#include "host_common.hpp"
#include "l1_memory.hpp"
#include "ttnn/device_operation.hpp"
#include "timing_host.hpp"

#ifndef BB_PROFILE
#define BB_PROFILE 0
#endif
#ifndef BB_AUDIT
#define BB_AUDIT 0
#endif

namespace ttnn::operations::basic_block {
void BasicBlockDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t &attrs,
                                                               const tensor_args_t &tensors) {
    detail::validate_attrs(attrs);
    TT_FATAL(tensors.weights.size() == detail::parameter_count(attrs) &&
                 tensors.biases.size() == detail::parameter_count(attrs),
             "Expected {} weights and biases (conv1, conv2, conv3, optional downsample)",
             detail::parameter_count(attrs));
    detail::validate_matrix(tensors.x, tensors.x, detail::input_rows(attrs), attrs.in_channels);
    const uint32_t in_channels = detail::padded(attrs.in_channels);
    const uint32_t channels = detail::padded(attrs.channels), out_channels = detail::padded(attrs.out_channels);
    const uint32_t weight_rows[] = {in_channels, 9 * channels, channels, in_channels};
    const uint32_t weight_columns[] = {channels, channels, out_channels, out_channels};
    for (uint32_t i = 0; i < detail::parameter_count(attrs); ++i) {
        detail::validate_matrix(tensors.weights[i], tensors.x, weight_rows[i], weight_columns[i]);
        detail::validate_matrix(tensors.biases[i], tensors.x, 32, weight_columns[i]);
    }
}

BasicBlockDeviceOperation::spec_return_value_t
BasicBlockDeviceOperation::compute_output_specs(const operation_attributes_t &attrs, const tensor_args_t &) {
    return {detail::matrix_spec(detail::output_rows(attrs), attrs.out_channels, attrs.output_memory_config)};
}

BasicBlockDeviceOperation::tensor_return_value_t
BasicBlockDeviceOperation::create_output_tensors(const operation_attributes_t &attrs,
                                                 const tensor_args_t &tensors) {
    return {ttnn::create_device_tensor(compute_output_specs(attrs, tensors)[0], tensors.x.device())};
}
} // namespace ttnn::operations::basic_block

namespace ttnn::prim {
BasicBlockReturn basic_block(const Tensor &x,
                             const std::vector<Tensor> &weights,
                             const std::vector<Tensor> &biases,
                             uint32_t in_channels,
                             uint32_t channels,
                             uint32_t out_channels,
                             uint32_t batch_size,
                             uint32_t input_height,
                             uint32_t input_width,
                             uint32_t kernel_size,
                             uint32_t stride,
                             uint32_t padding,
                             uint32_t max_cores,
                             uint32_t block_tiles,
                             uint32_t parameter_cache_tiles) {
    using Op = operations::basic_block::BasicBlockDeviceOperation;
    namespace detail = operations::basic_block::detail;
    auto attrs = detail::attributes(in_channels, channels, out_channels, batch_size, input_height,
                                    input_width, kernel_size, stride, padding, max_cores, block_tiles,
                                    parameter_cache_tiles);
    TT_FATAL(x.storage_type() == StorageType::DEVICE && x.is_allocated(), "Expected allocated device input");
    const auto grid = x.device()->compute_with_storage_grid_size();
    // Snapshot the whole eligible pool, not only min(M, pool): the 2D path can
    // activate extra N workers even when there are few spatial tiles.
    const auto cores = detail::select_core_count(grid.x * grid.y, grid.x * grid.y, attrs.max_cores);
    // Snapshot on every invocation, including cache hits. The bounds are part
    // of attrs and therefore the program cache key. There is no DRAM fallback.
    const auto budget = detail::query_l1_memory(x.device(), detail::core_ranges(cores, grid.y));
    attrs.l1_available_bytes = budget.available_bytes;
    attrs.l1_reserved_bytes = budget.reserved_bytes;
    attrs.l1_cap_bytes = budget.cap_bytes;
    attrs.profile = detail::env_flag("BASIC_BLOCK_PROFILE", BB_PROFILE != 0);
    attrs.audit = detail::env_flag("BASIC_BLOCK_AUDIT", BB_AUDIT != 0);
    detail::configure_timing(attrs);
    attrs.path_override = detail::env_path();
    attrs.requested_n_shards = detail::env_n_shards();
    attrs.l1_debug = detail::env_flag("BASIC_BLOCK_L1_DEBUG");
    // Geometry planning is performed on program creation/cache miss. The memory
    // bounds above remain in every invocation's key, so cache reuse stays safe.

    // Diagnostic eager mode: finish CQ0 before/after the launch. Do not enable
    // this during trace capture; cache misses include compilation in host_ms.
    if (attrs.timing) tt::tt_metal::distributed::Finish(x.device()->mesh_command_queue());
    const auto timing_begin = attrs.timing ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    auto outputs = device_operation::launch<Op>(attrs, Op::tensor_args_t{x, weights, biases});
    if (attrs.timing) {
        tt::tt_metal::distributed::Finish(x.device()->mesh_command_queue());
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - timing_begin).count();
        detail::print_host_timing(attrs, ms);
    }
    return {outputs[0], weights, biases};
}
} // namespace ttnn::prim
