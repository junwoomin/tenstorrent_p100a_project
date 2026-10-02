#pragma once
#include <cstdint>
#include <tuple>
#include <vector>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::basic_block {
// Legacy operation name; the math is a 1x1 -> 3x3 -> 1x1 bottleneck, without BN.
struct BasicBlockDeviceOperation {
    struct operation_attributes_t {
        MemoryConfig output_memory_config;
        uint32_t in_channels, channels, out_channels;
        uint32_t batch_size, input_height, input_width, stride;
        uint32_t max_cores = 0;
        uint32_t block_tiles = 32;
        uint32_t parameter_cache_tiles = 128;
        // Read before every invocation and included in the operation cache key.
        // A cached program must never silently keep a larger obsolete L1 plan.
        uint64_t l1_available_bytes = 0;
        uint64_t l1_reserved_bytes = 0;
        uint64_t l1_cap_bytes = 0;
        bool profile = false;
        bool audit = false;
        bool timing = false;
        uint32_t timing_clock_mhz = 0;
        bool timing_dprint_fmt = false;
        // Environment controls are snapshotted on every invocation so path and
        // shard changes cannot accidentally reuse a previously compiled plan.
        uint32_t path_override = 0; // 0 auto, 1 original v5, 2 sharded, 3 stream
        uint32_t requested_n_shards = 0; // 0 automatic
        bool l1_debug = false;
    };

    struct tensor_args_t {
        const Tensor &x;
        const std::vector<Tensor> &weights;
        const std::vector<Tensor> &biases;
    };

    // The only allocated forward tensor is the final block output [Y].
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    static void validate_on_program_cache_miss(const operation_attributes_t &, const tensor_args_t &);
    static spec_return_value_t compute_output_specs(const operation_attributes_t &, const tensor_args_t &);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t &, const tensor_args_t &);
    static tt::tt_metal::ProgramDescriptor
    create_resident_descriptor(const operation_attributes_t &, const tensor_args_t &, tensor_return_value_t &);
    static tt::tt_metal::ProgramDescriptor
    create_sharded_descriptor(const operation_attributes_t &, const tensor_args_t &, tensor_return_value_t &);
    static tt::tt_metal::ProgramDescriptor
    create_descriptor(const operation_attributes_t &, const tensor_args_t &, tensor_return_value_t &);
};
} // namespace ttnn::operations::basic_block

namespace ttnn::prim {
using BasicBlockReturn = std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>;
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
                             uint32_t parameter_cache_tiles);
} // namespace ttnn::prim
