#pragma once
#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <sstream>
#include <iomanip>
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

inline bool env_flag(const char *name, bool default_value = false) {
    const char *value = std::getenv(name);
    TT_FATAL(value == nullptr || ((value[0] == '0' || value[0] == '1') && value[1] == '\0'),
             "{} must be 0 or 1", name);
    return value == nullptr ? default_value : value[0] == '1';
}

inline uint32_t env_path() {
    const char* value = std::getenv("BASIC_BLOCK_PATH");
    const std::string path = value == nullptr ? "auto" : value;
    if (path == "auto") return 0;
    if (path == "v5") return 1;
    if (path == "sharded") return 2;
    if (path == "stream") return 3;
    TT_FATAL(false, "BASIC_BLOCK_PATH must be auto, v5, sharded, or stream; got {}", path);
    return 0;
}

inline uint32_t env_n_shards() {
    const char* value = std::getenv("BASIC_BLOCK_N_SHARDS");
    if (value == nullptr) return 0;
    TT_FATAL(*value != '\0', "BASIC_BLOCK_N_SHARDS must be a nonnegative integer (0 selects automatically)");
    uint64_t count = 0;
    for (const char* ch = value; *ch; ++ch) {
        TT_FATAL(*ch >= '0' && *ch <= '9', "BASIC_BLOCK_N_SHARDS must be a nonnegative integer");
        count = count * 10 + uint32_t(*ch - '0');
        TT_FATAL(count <= std::numeric_limits<uint32_t>::max(), "BASIC_BLOCK_N_SHARDS is too large");
    }
    return uint32_t(count);
}

inline uint32_t tiles(uint32_t n) {
    return n / 32 + (n % 32 != 0);
}

inline uint32_t padded(uint32_t n) {
    return 32 * tiles(n);
}

inline uint32_t output_height(const Attributes &attrs) {
    return (attrs.input_height - 1) / attrs.stride + 1;
}

inline uint32_t output_width(const Attributes &attrs) {
    return (attrs.input_width - 1) / attrs.stride + 1;
}

inline uint32_t input_rows(const Attributes &attrs) {
    return attrs.batch_size * attrs.input_height * attrs.input_width;
}

inline uint32_t output_rows(const Attributes &attrs) {
    return attrs.batch_size * output_height(attrs) * output_width(attrs);
}

inline bool has_downsample(const Attributes &attrs) {
    return attrs.stride != 1 || attrs.in_channels != attrs.out_channels;
}

inline uint32_t parameter_count(const Attributes &attrs) {
    return has_downsample(attrs) ? 4 : 3;
}

inline TensorSpec matrix_spec(uint32_t rows, uint32_t cols, const MemoryConfig &memory) {
    return TensorSpec(Shape{1, 1, rows, cols},
                      TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory));
}

inline void validate_attrs(const Attributes &attrs) {
    TT_FATAL(attrs.block_tiles >= 1 && attrs.block_tiles <= 64, "block_tiles must be 1..64");
    TT_FATAL(attrs.parameter_cache_tiles >= 1 && attrs.parameter_cache_tiles <= 256,
             "parameter_cache_tiles must be 1..256");
    TT_FATAL(attrs.stride == 1 || attrs.stride == 2, "stride must be 1 or 2");
    TT_FATAL(attrs.batch_size && attrs.input_height && attrs.input_width && attrs.in_channels &&
                 attrs.channels && attrs.out_channels,
             "All dimensions must be positive");
    TT_FATAL(uint64_t(attrs.batch_size) * attrs.input_height * attrs.input_width <= 0x7fffffffULL,
             "Flattened spatial dimension exceeds supported range");
    TT_FATAL(attrs.in_channels <= 0x100000 && attrs.channels <= 0x100000 && attrs.out_channels <= 0x100000,
             "Channel dimension exceeds supported range");
    TT_FATAL(attrs.output_memory_config == ttnn::DRAM_MEMORY_CONFIG,
             "Only interleaved DRAM output is supported");
    for (const auto [rows, cols] :
         {std::pair{input_rows(attrs), attrs.in_channels},
          std::pair{output_rows(attrs), attrs.out_channels}}) {
        TT_FATAL(uint64_t(padded(rows)) * padded(cols) * 2 <= 0xffffffffULL,
                 "Input/output tensor exceeds 32-bit byte-address range");
    }
}

inline void validate_matrix(const Tensor &tensor, const Tensor &x, uint32_t rows, uint32_t cols) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE && tensor.is_allocated(),
             "Expected allocated device tensor");
    TT_FATAL(tensor.device() == x.device(), "All tensors must be on the same device");
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16 && tensor.layout() == Layout::TILE,
             "Expected BF16 TILE tensor");
    TT_FATAL(tensor.memory_config() == ttnn::DRAM_MEMORY_CONFIG, "Expected interleaved DRAM tensor");
    TT_FATAL(tensor.buffer()->page_size() == 2048, "Expected standard 32x32 BF16 tile pages");
    TT_FATAL(uint64_t(padded(rows)) * padded(cols) * 2 <= 0xffffffffULL,
             "Tensor exceeds 32-bit byte-address range");
}

inline CoreRangeSet core_set() {
    return CoreRangeSet({CoreRange({0, 0}, {0, 0})});
}

// 코어를 선택하는 순서는 기존과 동일하다: x=i/grid_height, y=i%grid_height.
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

// 실행 경로 선택과 실제 프로그램 생성이 반드시 같은 계획을 사용한다.
inline bottleneck_geometry::DirectCachePlan make_direct_plan(const Attributes &attrs, uint32_t available) {
    const uint32_t cores = select_core_count(tiles(output_rows(attrs)), available, attrs.max_cores);
    return bottleneck_geometry::direct_cache_plan(
        {attrs.batch_size, attrs.input_height, attrs.input_width, attrs.stride}, tiles(attrs.in_channels),
        tiles(attrs.channels), tiles(attrs.out_channels), has_downsample(attrs), cores, attrs.block_tiles,
        attrs.parameter_cache_tiles, uint32_t(attrs.l1_available_bytes / 2048));
}

inline bottleneck_geometry::StreamPlan make_stream_plan(const Attributes &attrs, uint32_t available) {
    const uint32_t cores = select_core_count(tiles(output_rows(attrs)), available, attrs.max_cores);
    return bottleneck_geometry::streaming_plan(
        {attrs.batch_size, attrs.input_height, attrs.input_width, attrs.stride}, tiles(attrs.in_channels),
        tiles(attrs.channels), tiles(attrs.out_channels), has_downsample(attrs), cores, attrs.block_tiles,
        attrs.parameter_cache_tiles, attrs.l1_available_bytes);
}

inline bottleneck_geometry::ShardedPlan make_sharded_plan(const Attributes& attrs, uint32_t available) {
    const uint32_t limit = attrs.max_cores == 0 ? available : std::min(attrs.max_cores, available);
    return bottleneck_geometry::sharded_plan(
        {attrs.batch_size, attrs.input_height, attrs.input_width, attrs.stride}, tiles(attrs.in_channels),
        tiles(attrs.channels), tiles(attrs.out_channels), has_downsample(attrs), limit, attrs.block_tiles,
        attrs.l1_available_bytes, attrs.requested_n_shards);
}

inline std::string sharded_l1_report(const Attributes& attrs, const bottleneck_geometry::ShardedPlan& p) {
    std::ostringstream out;
    const auto row = [&](const char* name, uint64_t bytes) {
        out << std::left << std::setw(37) << name << ": " << std::right << std::setw(10) << bytes << " B\n";
    };
    out << "============= v6 2D L1 Memory Plan =============\n"
        << "path=2D_N_SHARDED_WEIGHT_RESIDENT M_shards=" << p.m_shards << " N_shards=" << p.n_shards
        << " active_cores=" << p.cores << " block=" << p.block << " ring=" << p.ring
        << " processing=" << p.processing_tiles << "\n"
        << "Logical core index=m*N+n; N varies fastest; logical x=index/grid.y, y=index%grid.y\n"
        << "Uniform padded local widths: hidden=" << p.local_hidden << " output=" << p.local_cout << " tiles\n";
    row("Input / halo / residual CB0", p.operand_bytes);
    row("Local Conv1 activation CB20", p.activation_a_bytes);
    row("Local Conv2 activation CB21", p.activation_b_bytes);
    row("Gathered full Conv2 channels CB22", p.full_b_bytes);
    row("Resident weights and biases CB23", p.weight_bytes);
    row("Final local output CB16", p.output_bytes);
    row("Zero CB24", p.zero_bytes);
    row("DRAM gather scratch CB25", p.dram_scratch_bytes);
    row("Actual CB allocation", p.allocated_bytes);
    row("Safety margin", p.safety_bytes);
    row("Required L1 including margin", p.required_bytes);
    row("Available contiguous L1/core", p.available_bytes);
    row("Allocator reserved base", attrs.l1_reserved_bytes);
    out << "Queue depths: operand=" << p.cb_depth << " output=" << p.out_depth << "\n"
        << "Weight capacities (padding may exceed payload; all bytes counted above):\n";
    row("Conv1 W capacity", p.conv1_weight_bytes);
    row("Conv2 W capacity", p.conv2_weight_bytes);
    row("Conv3 W capacity", p.conv3_weight_bytes);
    row("Shortcut W capacity", p.shortcut_weight_bytes);
    out << "Weights are preloaded once per core per invocation; all M patches reuse CB23.\n"
        << "================================================\n";
    return out.str();
}

inline std::string l1_report(const Attributes &attrs, const bottleneck_geometry::StreamPlan &p) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    const auto row = [&](const char *name, uint64_t bytes) {
        out << std::left << std::setw(39) << name << ": " << std::right << std::setw(10)
            << double(bytes) / 1024 << " KiB (" << bytes << " B)\n";
    };
    out << "================ L1 Memory Plan ================\n";
    out << "strategy: tiled streaming; cores=" << p.cores << " block=" << p.block << " ring=" << p.ring
        << " M=" << p.processing_tiles << " reduction_mask=" << p.reductions << "\n";
    row("Available L1/core", p.available_bytes);
    row("CB metadata / firmware / reserved base", attrs.l1_reserved_bytes);
    row("Upper allocatable address", attrs.l1_cap_bytes);
    row("Input / halo / residual (shared CB0)", p.operand_bytes);
    row("Activation buffer A (CB20)", p.activation_a_bytes);
    row("Activation buffer B (CB21)", p.activation_b_bytes);
    row("Weight+bias workspace (shared CB23)", p.weight_bytes);
    row("Accumulation/final output (CB16)", p.output_bytes);
    row("Zero / other temporary (CB24)", p.zero_bytes);
    row("DRAM gather scratch (CB25)", p.dram_scratch_bytes);
    out << "Residual accumulation: Tensix DST registers, no additional L1 allocation\n";
    out << "Per-stage weight PAYLOAD upper bounds (shared CB23; do not sum):\n";
    row("  Conv1 weights+bias", p.conv1_weight_bytes);
    row("  Conv2 weights+bias", p.conv2_weight_bytes);
    row("  Conv3 weights+bias", p.conv3_weight_bytes);
    row("  Shortcut weights+bias", p.shortcut_weight_bytes);
    out << "Per-phase physical reservations (fixed CB backing remains allocated):\n";
    row("  Phase Conv1", p.phase_conv1_bytes);
    row("  Phase Conv2", p.phase_conv2_bytes);
    row("  Phase Conv3", p.phase_conv3_bytes);
    row("  Phase Add", p.phase_add_bytes);
    row("Peak simultaneous physical allocation", p.required_bytes);
    row("Remaining margin", p.available_bytes > p.required_bytes ? p.available_bytes - p.required_bytes : 0);
    row("Initial single-buffer candidate", p.initial_single_bytes);
    row("Initial double-buffer candidate", p.initial_double_bytes);
    row("Selected geometry single buffers", p.selected_single_bytes);
    row("Selected geometry double buffers", p.selected_double_bytes);
    out << "Queue depths: operand=" << p.cb_depth << " weight=" << p.weight_depth
        << " output=" << p.out_depth << " weight chunk=" << p.weight_chunk << " tiles\n";
    out << "CB backing storage stays allocated for the entire program; pop_front does not free L1.\n"
        << "Auto double-buffer rule: repeated work AND operand <=25%, weight/output <=12.5% of budget.\n"
        << "Double queues permit dataflow/compute overlap; speedup requires a device measurement.\n"
        << "================================================\n";
    return out.str();
}

inline void
add_cb(ProgramDescriptor &program, uint32_t id, uint32_t depth = 2, const CoreRangeSet &cores = core_set()) {
    program.cbs.push_back(
        CBDescriptor{.total_size = depth * 2048,
                     .core_ranges = cores,
                     .format_descriptors = {{CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(id),
                                                                .data_format = tt::DataFormat::Float16_b,
                                                                .page_size = 2048}}}});
}

inline KernelDescriptor kernel(const std::string &name, const CoreRangeSet &cores = core_set()) {
    KernelDescriptor descriptor;
    descriptor.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/resnet/basic_block/device/kernels/" + name;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = cores;
    return descriptor;
}

inline void accessor(KernelDescriptor &descriptor, const Tensor &tensor) {
    TensorAccessorArgs(*tensor.buffer()).append_to(descriptor.compile_time_args);
}

inline ComputeConfigDescriptor compute_config() {
    return ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
}

inline Attributes attributes(uint32_t in_channels,
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
                             uint32_t parameter_cache_tiles = 128) {
    TT_FATAL(kernel_size == 3 && padding == 1, "Middle convolution is fixed at kernel_size=3, padding=1");
    Attributes attrs{ttnn::DRAM_MEMORY_CONFIG,
                     in_channels,
                     channels,
                     out_channels,
                     batch_size,
                     input_height,
                     input_width,
                     stride,
                     max_cores,
                     block_tiles,
                     parameter_cache_tiles};
    validate_attrs(attrs);
    return attrs;
}
} // namespace ttnn::operations::basic_block::detail
