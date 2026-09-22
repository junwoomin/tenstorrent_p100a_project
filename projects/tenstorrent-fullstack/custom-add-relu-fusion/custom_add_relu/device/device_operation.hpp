#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::custom_add_relu {

struct AddReluDeviceOperation {
    struct operation_attributes_t {
        MemoryConfig output_memory_config;
    };
    struct tensor_args_t {
        const Tensor& a;
        const Tensor& b;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
}  // namespace ttnn::operations::custom_add_relu

namespace ttnn::prim {
Tensor custom_add_relu(const Tensor& a, const Tensor& b);
}
