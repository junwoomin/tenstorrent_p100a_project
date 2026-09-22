#include "device_operation.hpp"
#include <tt_stl/assert.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::custom_add_relu {
using namespace tt::tt_metal;

void AddReluDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensors) {
    const auto& a = tensors.a;
    const auto& b = tensors.b;
    TT_FATAL(a.storage_type() == StorageType::DEVICE &&
             b.storage_type() == StorageType::DEVICE,
             "custom_add_relu requires device tensors");
    TT_FATAL(a.is_allocated() && b.is_allocated(), "Input buffers must be allocated");
    TT_FATAL(a.device() == b.device(), "Inputs must be on the same device");
    TT_FATAL(a.dtype() == DataType::BFLOAT16 && b.dtype() == DataType::BFLOAT16,
             "Only BFLOAT16 is supported");
    TT_FATAL(a.layout() == Layout::TILE && b.layout() == Layout::TILE,
             "Only TILE layout is supported");
    TT_FATAL(a.memory_config() == ttnn::DRAM_MEMORY_CONFIG &&
             b.memory_config() == ttnn::DRAM_MEMORY_CONFIG &&
             attributes.output_memory_config == ttnn::DRAM_MEMORY_CONFIG,
             "Only interleaved DRAM memory is supported");
    TT_FATAL(a.logical_shape() == b.logical_shape(),
             "Input shapes must match; broadcasting is not supported");
    TT_FATAL(a.logical_shape().rank() == 4, "This example supports rank-4 inputs");
    TT_FATAL(a.logical_shape() == a.padded_shape() &&
             b.logical_shape() == b.padded_shape(),
             "This example requires inputs without tile padding");
    TT_FATAL(a.logical_shape()[-2] % 32 == 0 && a.logical_shape()[-1] % 32 == 0,
             "The last two dimensions must be multiples of 32");
    TT_FATAL(a.tensor_spec().tile().get_height() == 32 &&
             a.tensor_spec().tile().get_width() == 32 &&
             b.tensor_spec().tile().get_height() == 32 &&
             b.tensor_spec().tile().get_width() == 32,
             "Only standard 32x32 tiles are supported");
    TT_FATAL(a.physical_volume() > 0, "Empty tensors are not supported");
}

AddReluDeviceOperation::spec_return_value_t
AddReluDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensors) {
    return TensorSpec(
        tensors.a.logical_shape(),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE),
                     attributes.output_memory_config));
}

AddReluDeviceOperation::tensor_return_value_t
AddReluDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(attributes, tensors), tensors.a.device());
}
}  // namespace ttnn::operations::custom_add_relu

namespace ttnn::prim {
Tensor custom_add_relu(const Tensor& a, const Tensor& b) {
    using Op = operations::custom_add_relu::AddReluDeviceOperation;
    const auto attributes = Op::operation_attributes_t{ttnn::DRAM_MEMORY_CONFIG};
    const auto tensors = Op::tensor_args_t{a, b};
    return ttnn::device_operation::launch<Op>(attributes, tensors);
}
}  // namespace ttnn::prim
