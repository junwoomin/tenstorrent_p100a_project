#pragma once

#include <vector>
#include "device_operation.hpp"

namespace ttnn::operations::custom_add_relu {

// grad_a and grad_b are equal in value but allocated independently.
struct AddReluBackwardDeviceOperation {
    using operation_attributes_t = AddReluDeviceOperation::operation_attributes_t;
    struct tensor_args_t {
        const Tensor& grad_out;
        const Tensor& saved_output;
    };
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

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
std::vector<Tensor> custom_add_relu_backward(
    const Tensor& grad_out, const Tensor& saved_output);
}
