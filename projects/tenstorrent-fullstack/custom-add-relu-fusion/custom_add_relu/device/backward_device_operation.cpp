#include "backward_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::custom_add_relu {

void AddReluBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    AddReluDeviceOperation::validate_on_program_cache_miss(
        attrs, AddReluDeviceOperation::tensor_args_t{tensors.grad_out, tensors.saved_output});
}

AddReluBackwardDeviceOperation::spec_return_value_t
AddReluBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    const auto spec = AddReluDeviceOperation::compute_output_specs(
        attrs, AddReluDeviceOperation::tensor_args_t{tensors.grad_out, tensors.saved_output});
    return {spec, spec};
}

AddReluBackwardDeviceOperation::tensor_return_value_t
AddReluBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    const auto specs = compute_output_specs(attrs, tensors);
    return {
        create_device_tensor(specs[0], tensors.grad_out.device()),
        create_device_tensor(specs[1], tensors.grad_out.device()),
    };
}
}  // namespace ttnn::operations::custom_add_relu

namespace ttnn::prim {
std::vector<Tensor> custom_add_relu_backward(
    const Tensor& grad_out, const Tensor& saved_output) {
    using Op = operations::custom_add_relu::AddReluBackwardDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{ttnn::DRAM_MEMORY_CONFIG},
        Op::tensor_args_t{grad_out, saved_output});
}
}  // namespace ttnn::prim
