#pragma once






#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <cstdlib>
#include <limits>
#include <tt_stl/assert.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "device_operation.hpp"
#include "kernels/direct_cache_plan.hpp"
#include "kernels/l1_plan.hpp"
#include "kernels/sharded_plan.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::basic_block::detail {
using namespace tt;
using namespace tt::tt_metal;
using Attributes = BasicBlockDeviceOperation::operation_attributes_t;



inline uint32_t env_path() {
    const char* value = std::getenv("BASIC_BLOCK_PATH");
    const std::string path = value == nullptr ? "auto" : value;
    if (path == "auto") {
        return 0;
    }
    if (path == "v5") {
        return 1;
    }
    if (path == "sharded") {
        return 2;
    }
    if (path == "stream") {
        return 3;
    }
    TT_FATAL(false, "BASIC_BLOCK_PATH must be auto, v5, sharded, or stream; got {}", path);
    return 0;
}



inline uint32_t env_n_shards() {
    const char* value = std::getenv("BASIC_BLOCK_N_SHARDS");
    if (value == nullptr) {
        return 0;
    }
    TT_FATAL(
        *value != '\0',
        "BASIC_BLOCK_N_SHARDS must be a nonnegative integer (0 selects automatically)"
    );
    uint64_t count = 0;
    for (const char* ch = value; *ch; ++ch) {
        TT_FATAL(*ch >= '0' && *ch <= '9', "BASIC_BLOCK_N_SHARDS must be a nonnegative integer");
        count = count * 10 + uint32_t(*ch - '0');
        TT_FATAL(
            count <= std::numeric_limits<uint32_t>::max(), "BASIC_BLOCK_N_SHARDS is too large"
        );
    }
    return uint32_t(count);
}


inline uint32_t tiles(uint32_t n) {
    return n / 32 + (n % 32 != 0);
}



inline uint32_t padded(uint32_t n) {
    return 32 * tiles(n);
}



inline uint32_t output_height(const Attributes& attrs) {
    return (attrs.input_height - 1) / attrs.stride + 1;
}

inline uint32_t output_width(const Attributes& attrs) {
    return (attrs.input_width - 1) / attrs.stride + 1;
}


inline uint32_t input_rows(const Attributes& attrs) {
    return attrs.batch_size * attrs.input_height * attrs.input_width;
}

inline uint32_t output_rows(const Attributes& attrs) {
    return attrs.batch_size * output_height(attrs) * output_width(attrs);
}



inline bool has_downsample(const Attributes& attrs) {
    return attrs.stride != 1 || attrs.in_channels != attrs.out_channels;
}


inline uint32_t parameter_count(const Attributes& attrs) {
    return has_downsample(attrs) ? 4 : 3;
}



inline TensorSpec matrix_spec(uint32_t rows, uint32_t cols, const MemoryConfig& memory) {
    return TensorSpec(
        Shape{1, 1, rows, cols}, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory)
    );
}



inline void validate_attrs(const Attributes& attrs) {
    TT_FATAL(attrs.block_tiles >= 1 && attrs.block_tiles <= 64, "block_tiles must be 1..64");
    TT_FATAL(
        attrs.parameter_cache_tiles >= 1 && attrs.parameter_cache_tiles <= 256,
        "parameter_cache_tiles must be 1..256"
    );
    TT_FATAL(attrs.stride == 1 || attrs.stride == 2, "stride must be 1 or 2");
    TT_FATAL(
        attrs.batch_size && attrs.input_height && attrs.input_width && attrs.in_channels &&
            attrs.channels && attrs.out_channels,
        "All dimensions must be positive"
    );
    TT_FATAL(
        uint64_t(attrs.batch_size) * attrs.input_height * attrs.input_width <= 0x7fffffffULL,
        "Flattened spatial dimension exceeds supported range"
    );
    TT_FATAL(
        attrs.in_channels <= 0x100000 && attrs.channels <= 0x100000 &&
            attrs.out_channels <= 0x100000,
        "Channel dimension exceeds supported range"
    );
    TT_FATAL(
        attrs.output_memory_config == ttnn::DRAM_MEMORY_CONFIG,
        "Only interleaved DRAM output is supported"
    );
    for (const auto [rows, cols] :
         {std::pair{input_rows(attrs), attrs.in_channels},
          std::pair{output_rows(attrs), attrs.out_channels}}) {
        TT_FATAL(
            uint64_t(padded(rows)) * padded(cols) * 2 <= 0xffffffffULL,
            "Input/output tensor exceeds 32-bit byte-address range"
        );
    }
}



inline void validate_matrix(const Tensor& tensor, const Tensor& x, uint32_t rows, uint32_t cols) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.is_allocated(),
        "Expected allocated device tensor"
    );
    TT_FATAL(tensor.device() == x.device(), "All tensors must be on the same device");
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 && tensor.layout() == Layout::TILE,
        "Expected BF16 TILE tensor"
    );
    TT_FATAL(
        tensor.memory_config() == ttnn::DRAM_MEMORY_CONFIG, "Expected interleaved DRAM tensor"
    );
    TT_FATAL(tensor.buffer()->page_size() == 2048, "Expected standard 32x32 BF16 tile pages");
    TT_FATAL(
        uint64_t(padded(rows)) * padded(cols) * 2 <= 0xffffffffULL,
        "Tensor exceeds 32-bit byte-address range"
    );
}


inline CoreRangeSet core_set() {
    return CoreRangeSet({CoreRange({0, 0}, {0, 0})});
}



inline CoreCoord core_at(uint32_t index, uint32_t grid_height) {
    return {index / grid_height, index % grid_height};
}



inline CoreRangeSet core_ranges(uint32_t count, uint32_t grid_height) {
    std::set<CoreRange> ranges;
    for (uint32_t i = 0; i < count; ++i) {
        const auto core = core_at(i, grid_height);
        ranges.emplace(core, core);
    }
    return CoreRangeSet(ranges);
}



inline uint32_t select_core_count(uint32_t jobs, uint32_t available, uint32_t max_cores) {
    const uint32_t limit = max_cores == 0 ? available : std::min(max_cores, available);
    const uint32_t cores = std::min(jobs, limit);
    TT_FATAL(cores > 0, "No compute cores available");
    return cores;
}



inline bottleneck_geometry::DirectCachePlan
make_direct_plan(const Attributes& attrs, uint32_t available) {
    const uint32_t cores = select_core_count(tiles(output_rows(attrs)), available, attrs.max_cores);
    return bottleneck_geometry::direct_cache_plan(
        {attrs.batch_size, attrs.input_height, attrs.input_width, attrs.stride},
        tiles(attrs.in_channels),
        tiles(attrs.channels),
        tiles(attrs.out_channels),
        has_downsample(attrs),
        cores,
        attrs.block_tiles,
        attrs.parameter_cache_tiles,
        uint32_t(attrs.l1_available_bytes / 2048)
    );
}



inline bottleneck_geometry::StreamPlan
make_stream_plan(const Attributes& attrs, uint32_t available) {
    const uint32_t cores = select_core_count(tiles(output_rows(attrs)), available, attrs.max_cores);
    return bottleneck_geometry::streaming_plan(
        {attrs.batch_size, attrs.input_height, attrs.input_width, attrs.stride},
        tiles(attrs.in_channels),
        tiles(attrs.channels),
        tiles(attrs.out_channels),
        cores,
        attrs.block_tiles,
        attrs.parameter_cache_tiles,
        attrs.l1_available_bytes
    );
}



inline bottleneck_geometry::ShardedPlan
make_sharded_plan(const Attributes& attrs, uint32_t available) {
    const uint32_t limit = attrs.max_cores == 0 ? available : std::min(attrs.max_cores, available);
    return bottleneck_geometry::sharded_plan(
        {attrs.batch_size, attrs.input_height, attrs.input_width, attrs.stride},
        tiles(attrs.in_channels),
        tiles(attrs.channels),
        tiles(attrs.out_channels),
        has_downsample(attrs),
        limit,
        attrs.block_tiles,
        attrs.l1_available_bytes,
        attrs.requested_n_shards
    );
}



inline void add_cb(
    ProgramDescriptor& program,
    uint32_t id,
    uint32_t depth = 2,
    const CoreRangeSet& cores = core_set()
) {
    program.cbs.push_back(
        CBDescriptor{
            .total_size = depth * 2048,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id),
                .data_format = tt::DataFormat::Float16_b,
                .page_size = 2048
            }}}
        }
    );
}



inline KernelDescriptor kernel(const std::string& name, const CoreRangeSet& cores = core_set()) {
    KernelDescriptor descriptor;
    descriptor.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/resnet/basic_block/device/kernels/" + name;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = cores;
    return descriptor;
}



inline void accessor(KernelDescriptor& descriptor, const Tensor& tensor) {
    TensorAccessorArgs(*tensor.buffer()).append_to(descriptor.compile_time_args);
}



inline ComputeConfigDescriptor compute_config() {
    return ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false
    };
}



inline Attributes attributes(
    uint32_t in_channels,
    uint32_t channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t padding,
    uint32_t max_cores = 0,
    uint32_t block_tiles = 32,
    uint32_t parameter_cache_tiles = 128
) {
    TT_FATAL(
        kernel_size == 3 && padding == 1, "Middle convolution is fixed at kernel_size=3, padding=1"
    );
    Attributes attrs{
        ttnn::DRAM_MEMORY_CONFIG,
        in_channels,
        channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        stride,
        max_cores,
        block_tiles,
        parameter_cache_tiles
    };
    validate_attrs(attrs);
    return attrs;
}
}
