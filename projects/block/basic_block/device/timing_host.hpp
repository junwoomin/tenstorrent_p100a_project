#pragma once
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tt-metalium/distributed.hpp>

namespace ttnn::operations::basic_block::detail {
inline void configure_timing(BasicBlockDeviceOperation::operation_attributes_t& attrs) {
    attrs.timing = env_flag("BASIC_BLOCK_TIMING");
    if (!attrs.timing) return;
    const char* clock = std::getenv("BASIC_BLOCK_CLOCK_MHZ");
    char* end = nullptr;
    errno = 0;
    const unsigned long mhz = clock ? std::strtoul(clock, &end, 10) : 0;
    TT_FATAL(clock && end != clock && *end == '\0' && errno == 0 && mhz >= 1 && mhz <= 10000,
             "BASIC_BLOCK_TIMING requires BASIC_BLOCK_CLOCK_MHZ=<actual integer Tensix MHz>. "
             "Do not assume a GPU/DRAM/PCIe clock or hard-code an unverified frequency.");
    attrs.timing_clock_mhz = static_cast<uint32_t>(mhz);
    attrs.timing_dprint_fmt = env_flag("BASIC_BLOCK_DPRINT_FMT");
    const char* cores = std::getenv("TT_METAL_DPRINT_CORES");
    TT_FATAL(cores && *cores, "Set TT_METAL_DPRINT_CORES='(0,0)' before starting Python for timing output");
}
inline void timing_defines(tt::tt_metal::KernelDescriptor& kernel,
                           const BasicBlockDeviceOperation::operation_attributes_t& attrs,
                           uint32_t path, uint32_t role) {
    kernel.defines = {
        {"BB_PROFILE", attrs.profile ? "1" : "0"},
        {"BB_AUDIT", attrs.audit ? "1" : "0"},
        {"BB_TIMING", attrs.timing ? "1" : "0"},
        {"BB_TIMING_CLOCK_MHZ", std::to_string(attrs.timing_clock_mhz)},
        {"BB_TIMING_DPRINT_FMT", attrs.timing_dprint_fmt ? "1" : "0"},
        {"BB_TIMING_PATH", std::to_string(path)},
        {"BB_TIMING_ROLE", std::to_string(role)}};
}
inline void print_host_timing(const BasicBlockDeviceOperation::operation_attributes_t& a, double ms) {
    const char* name = std::getenv("BASIC_BLOCK_TIMING_LABEL");
    std::printf("[BB_HOST] %s = %.4f ms | B=%u H=%u W=%u Cin=%u C=%u Cout=%u stride=%u "
                "| synchronized CQ0 host elapsed; includes launch/print overhead and compilation on cache miss\n",
                name && *name ? name : "basic_block", ms,
                a.batch_size, a.input_height, a.input_width, a.in_channels, a.channels, a.out_channels, a.stride);
    std::fflush(stdout);
}
} // namespace ttnn::operations::basic_block::detail
