#pragma once

// Host kernel defines are part of the TT-Metal kernel descriptor hash. Debug
// builds must also enable the runtime profiler / device-print service.
#ifndef BB_PROFILE
#define BB_PROFILE 0
#endif
#ifndef BB_AUDIT
#define BB_AUDIT 0
#endif

#if BB_PROFILE
#if __has_include("api/debug/profiler.h")
#include "api/debug/profiler.h"
#else
#include "tools/profiler/kernel_profiler.hpp"
#endif
#define BB_STREAM_ZONE_IF(condition, name, ...) \
    do { \
        if (condition) { DeviceZoneScopedN(name); __VA_ARGS__; } \
        else { __VA_ARGS__; } \
    } while (false)
#else
#define BB_STREAM_ZONE_IF(condition, name, ...) do { (void)(condition); __VA_ARGS__; } while (false)
#endif

#if BB_AUDIT
#include "api/debug/dprint.h"
#endif

namespace bottleneck_stream {
#if BB_PROFILE && defined(PROFILE_KERNEL)
// Same wall-clock registers as the pinned official kernel_profiler.hpp. The
// high/low/high retry also handles the low-word rollover during long kernels.
inline uint64_t profile_clock() {
#if defined(ARCH_QUASAR)
    return kernel_profiler::quasar_read_wall_clock_64();
#else
    volatile tt_reg_ptr uint32_t *clock =
        reinterpret_cast<volatile tt_reg_ptr uint32_t *>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t high, low, again;
    do {
        high = clock[kernel_profiler::WALL_CLOCK_HIGH_INDEX];
        low = clock[kernel_profiler::WALL_CLOCK_LOW_INDEX];
        again = clock[kernel_profiler::WALL_CLOCK_HIGH_INDEX];
    } while (high != again);
    return (uint64_t(high) << 32) | low;
#endif
}
struct CycleScope {
    uint64_t &total;
    uint64_t begin;
    explicit CycleScope(uint64_t &value) : total(value), begin(profile_clock()) {}
    ~CycleScope() { total += profile_clock() - begin; }
};
#else
struct CycleScope { explicit CycleScope(uint64_t &) {} };
#endif

// These are per-core, per-RISC elapsed cycles including waits in each scope.
// Output one accumulated value per stage so traces cannot grow with image size.
// Do not add different cores/RISCs or combine overlapped reader/compute cycles.
struct StageTotals {
    uint64_t conv1 = 0, conv2 = 0, shortcut = 0, conv3 = 0, residual = 0, total = 0;
    void print_compute() const {
#if BB_PROFILE && defined(PROFILE_KERNEL)
        DeviceTimestampedData("BB_TOTAL_CONV1_CYCLES", conv1);
        DeviceTimestampedData("BB_TOTAL_CONV2_CYCLES", conv2);
        DeviceTimestampedData("BB_TOTAL_SHORTCUT_CYCLES", shortcut);
        DeviceTimestampedData("BB_TOTAL_CONV3_CYCLES", conv3);
        DeviceTimestampedData("BB_TOTAL_RESIDUAL_CYCLES", residual);
        DeviceTimestampedData("BB_TOTAL_COMPUTE_CYCLES", total);
#endif
    }
};

// These count actual payload bytes at this source's DRAM issue sites, not NoC
// packet overhead or an external bus trace. No intermediate address exists in
// the forward reader/writer ABI. L1 gather traffic is counted separately.
struct Traffic {
#if BB_AUDIT
    uint64_t input = 0, weight = 0, local = 0, output = 0;
    void input_read(uint32_t n) { input += n; }
    void weight_read(uint32_t n) { weight += n; }
    void local_read(uint32_t n) { local += n; }
    void output_write(uint32_t n) { output += n; }
    void print_reader() const {
#ifdef DEVICE_PRINT
        DPRINT("[BB_AUDIT reader] Input DRAM read={} B; Weight DRAM read={} B; "
               "Intermediate DRAM read=0 B; Intermediate DRAM write=0 B; L1 gather/copy={} B\n",
               input, weight, local);
#else
        DPRINT << "[BB_AUDIT reader] Input DRAM read=" << input << " B; Weight DRAM read=" << weight
               << " B; Intermediate DRAM read=0 B; Intermediate DRAM write=0 B; L1 gather/copy="
               << local << " B" << ENDL();
#endif
    }
    void print_writer() const {
#ifdef DEVICE_PRINT
        DPRINT("[BB_AUDIT writer] Final output DRAM write={} B; "
               "Intermediate DRAM read=0 B; Intermediate DRAM write=0 B\n", output);
#else
        DPRINT << "[BB_AUDIT writer] Final output DRAM write=" << output
               << " B; Intermediate DRAM read=0 B; Intermediate DRAM write=0 B" << ENDL();
#endif
    }
#else
    void input_read(uint32_t) {}
    void weight_read(uint32_t) {}
    void local_read(uint32_t) {}
    void output_write(uint32_t) {}
    void print_reader() const {}
    void print_writer() const {}
#endif
};
} // namespace bottleneck_stream
