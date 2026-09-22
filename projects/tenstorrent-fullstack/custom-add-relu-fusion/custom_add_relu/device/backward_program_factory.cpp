#include "backward_device_operation.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::custom_add_relu {
using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor AddReluBackwardDeviceOperation::create_descriptor(
    const operation_attributes_t&, const tensor_args_t& tensors,
    tensor_return_value_t& output) {
    const auto& a = tensors.grad_out;
    auto* a_buffer = a.buffer();
    auto* b_buffer = tensors.saved_output.buffer();
    auto* grad_a_buffer = output.at(0).buffer();
    auto* grad_b_buffer = output.at(1).buffer();

    constexpr uint32_t tile_elements = 32 * 32;
    constexpr uint32_t tile_bytes = tile_elements * 2;
    constexpr uint32_t cb_depth = 2;
    constexpr uint32_t cb_a = CBIndex::c_0;
    constexpr uint32_t cb_b = CBIndex::c_1;
    constexpr uint32_t cb_out = CBIndex::c_16;

    const uint32_t num_tiles = a.physical_volume() / tile_elements;
    const auto grid = a.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, group_1, group_2, tiles_group_1, tiles_group_2] =
        split_work_to_cores(grid, num_tiles);

    ProgramDescriptor desc;
    for (uint32_t cb_id : {cb_a, cb_b, cb_out}) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_depth * tile_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_id,
                .data_format = DataFormat::Float16_b,
                .page_size = tile_bytes,
            }}},
        });
    }

    const std::string kernel_dir =
        "ttnn/cpp/ttnn/operations/eltwise/custom_add_relu/device/kernels/";

    std::vector<uint32_t> reader_compile_args;
    TensorAccessorArgs(*a_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(*b_buffer).append_to(reader_compile_args);

    KernelDescriptor reader;
    reader.kernel_source = kernel_dir + "reader.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = reader_compile_args;
    reader.named_compile_time_args = {{"cb_a", cb_a}, {"cb_b", cb_b}};
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kernel_dir + "backward_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.named_compile_time_args = {
        {"cb_a", cb_a}, {"cb_b", cb_b}, {"cb_out", cb_out}};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
    };

    std::vector<uint32_t> writer_compile_args;
    TensorAccessorArgs(*grad_a_buffer).append_to(writer_compile_args);
    TensorAccessorArgs(*grad_b_buffer).append_to(writer_compile_args);
    KernelDescriptor writer;
    writer.kernel_source = kernel_dir + "backward_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = writer_compile_args;
    writer.named_compile_time_args = {{"cb_out", cb_out}};
    writer.config = WriterConfigDescriptor{};

    uint32_t start_tile = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core{i / grid.y, i % grid.y};
        const uint32_t count = group_1.contains(core) ? tiles_group_1 : tiles_group_2;
        // Buffer* entries enable buffer-address patching on cache hits.
        reader.emplace_runtime_args(core, {a_buffer, b_buffer, count, start_tile});
        compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{count});
        writer.emplace_runtime_args(core, {grad_a_buffer, grad_b_buffer, count, start_tile});
        start_tile += count;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}
}  // namespace ttnn::operations::custom_add_relu
