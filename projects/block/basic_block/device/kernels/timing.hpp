#pragma once
#include <cstdint>

#ifndef BB_TIMING
#define BB_TIMING 0
#endif
#ifndef BB_TIMING_CLOCK_MHZ
#define BB_TIMING_CLOCK_MHZ 0
#endif
#ifndef BB_TIMING_DPRINT_FMT
#define BB_TIMING_DPRINT_FMT 0
#endif

#if defined(TRISC_UNPACK)
#define BB_TIMING_COMPUTE_ROLE "unpack"
#elif defined(TRISC_MATH)
#define BB_TIMING_COMPUTE_ROLE "math"
#elif defined(TRISC_PACK)
#define BB_TIMING_COMPUTE_ROLE "pack"
#else
#define BB_TIMING_COMPUTE_ROLE "compute"
#endif

#ifndef BB_TIMING_PATH
#define BB_TIMING_PATH 0
#endif
#if BB_TIMING_PATH == 1
#define BB_TIMING_PATH_NAME "resident"
#elif BB_TIMING_PATH == 2
#define BB_TIMING_PATH_NAME "sharded"
#else
#define BB_TIMING_PATH_NAME "stream"
#endif
#ifndef BB_TIMING_ROLE
#define BB_TIMING_ROLE 0
#endif
#if BB_TIMING_ROLE == 0
#define BB_TIMING_ROLE_NAME "reader"
#elif BB_TIMING_ROLE == 2
#define BB_TIMING_ROLE_NAME "writer"
#else
#define BB_TIMING_ROLE_NAME BB_TIMING_COMPUTE_ROLE
#endif

#if BB_TIMING
#include "api/debug/dprint.h"
static_assert(BB_TIMING_CLOCK_MHZ > 0, "Set BASIC_BLOCK_CLOCK_MHZ to the measured Tensix clock in MHz");
#endif

namespace bb_timing {
enum class Metric : uint32_t {
    InputLoad, InputBarrier, WeightLoad, WeightBarrier,
    Routing, RoutingPlan, RoutingBarrier, ShortcutLoad, ShortcutBarrier,
    ActivationWait, CbWait, PeerSync, A2Routing, A2Barrier,
    Conv1, Conv2, Conv3, Shortcut, Residual, InputWait, WeightWait, DstWait,
    Matmul, Pack, Conv1Matmul, Conv2Matmul, Conv3Matmul, ShortcutMatmul,
    OutputWrite, OutputBarrier, OutputWait, Count
};

#if BB_TIMING
// Same Tensix wall-clock low register used by the official device profiler.
// Independent of PROFILE_KERNEL. Each individual interval must be < 2^32
// cycles. Unsigned subtraction tolerates one rollover during an interval.
inline uint32_t clock32() {
    asm volatile("" ::: "memory");
    const uint32_t value = *reinterpret_cast<volatile tt_reg_ptr uint32_t *>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory");
    return value;
}
struct Counters {
    uint64_t cycles[static_cast<uint32_t>(Metric::Count)]{};
    uint32_t seen = 0;
    uint32_t begin = clock32();

    void line(uint32_t start, uint32_t count, uint32_t lane, Metric metric, uint64_t value) const {
        const uint64_t us = value / uint64_t(BB_TIMING_CLOCK_MHZ);
        const uint32_t whole = uint32_t(us / 1000), frac = uint32_t(us % 1000);
#if BB_TIMING_DPRINT_FMT
#define BB_LINE(label) DPRINT("[BB_MS " BB_TIMING_PATH_NAME " " BB_TIMING_ROLE_NAME " M={}+{} N={}] " label " = {}.{}{}{} ms\n", start, count, lane, whole, frac/100, (frac/10)%10, frac%10)
#else
#define BB_LINE(label) DPRINT << "[BB_MS " BB_TIMING_PATH_NAME " " BB_TIMING_ROLE_NAME " M=" << start << "+" << count << " N=" << lane << "] " label " = " << whole << "." << frac/100 << (frac/10)%10 << frac%10 << " ms" << ENDL()
#endif
        switch (metric) {
        case Metric::InputLoad: BB_LINE("Input load incl barrier"); break;
        case Metric::InputBarrier: BB_LINE("  input read barrier [subset]"); break;
        case Metric::WeightLoad: BB_LINE("Weight load incl barrier"); break;
        case Metric::WeightBarrier: BB_LINE("  weight read barrier [subset]"); break;
        case Metric::Routing: BB_LINE("Conv1->Conv2 routing incl plan+barrier"); break;
        case Metric::RoutingPlan: BB_LINE("  routing make_runs [subset]"); break;
        case Metric::RoutingBarrier: BB_LINE("  routing read barrier [subset]"); break;
        case Metric::ShortcutLoad: BB_LINE("Shortcut input gather/load"); break;
        case Metric::ShortcutBarrier: BB_LINE("  shortcut read barrier [subset]"); break;
        case Metric::ActivationWait: BB_LINE("Activation ready wait"); break;
        case Metric::CbWait: BB_LINE("CB capacity/ready wait"); break;
        case Metric::PeerSync: BB_LINE("Peer sync wait"); break;
        case Metric::A2Routing: BB_LINE("Conv2->Conv3 routing"); break;
        case Metric::A2Barrier: BB_LINE("  A2 read barrier [subset]"); break;
        case Metric::Conv1: BB_LINE("Conv1 stage [inclusive]"); break;
        case Metric::Conv2: BB_LINE("Conv2 stage [inclusive]"); break;
        case Metric::Conv3: BB_LINE("Conv3 reduction span [inclusive]"); break;
        case Metric::Shortcut: BB_LINE("Shortcut span [inclusive]"); break;
        case Metric::Residual: BB_LINE("Residual add+ReLU+pack span"); break;
        case Metric::InputWait: BB_LINE("  input ready wait [subset]"); break;
        case Metric::WeightWait: BB_LINE("  weight ready wait [subset]"); break;
        case Metric::DstWait: BB_LINE("  DST acquire/wait [subset]"); break;
        case Metric::Matmul: BB_LINE("  matmul issue loop [subset]"); break;
        case Metric::Pack: BB_LINE("  pack issue loop [subset]"); break;
        case Metric::Conv1Matmul: BB_LINE("  Conv1 matmul issue [subset]"); break;
        case Metric::Conv2Matmul: BB_LINE("  Conv2 matmul issue [subset]"); break;
        case Metric::Conv3Matmul: BB_LINE("  Conv3 matmul issue [subset]"); break;
        case Metric::ShortcutMatmul: BB_LINE("  Shortcut matmul issue [subset]"); break;
        case Metric::OutputWrite: BB_LINE("Output DRAM write incl barrier"); break;
        case Metric::OutputBarrier: BB_LINE("  output write barrier [subset]"); break;
        case Metric::OutputWait: BB_LINE("Output ready wait"); break;
        case Metric::Count: BB_LINE("RISC total before print"); break;
        }
#undef BB_LINE
    }
    void print(const char*, const char*, uint32_t start, uint32_t count, uint32_t lane = 0) const {
        const uint32_t elapsed = clock32() - begin; // snapshot BEFORE any DPRINT
#if BB_TIMING_DPRINT_FMT
        DPRINT("[BB_SHAPE " BB_TIMING_PATH_NAME " " BB_TIMING_ROLE_NAME " M={}+{} N={}] B={} H={} W={} Cin={} C={} Cout={} stride={} clock_MHz={}\n",
               start, count, lane,
               get_named_compile_time_arg_val("batch_size"), get_named_compile_time_arg_val("input_height"),
               get_named_compile_time_arg_val("input_width"), get_named_compile_time_arg_val("in_channels"),
               get_named_compile_time_arg_val("channels"), get_named_compile_time_arg_val("out_channels"),
               get_named_compile_time_arg_val("stride"), BB_TIMING_CLOCK_MHZ);
#else
        DPRINT << "[BB_SHAPE " BB_TIMING_PATH_NAME " " BB_TIMING_ROLE_NAME " M=" << start << "+" << count << " N=" << lane
               << "] B=" << get_named_compile_time_arg_val("batch_size")
               << " H=" << get_named_compile_time_arg_val("input_height")
               << " W=" << get_named_compile_time_arg_val("input_width")
               << " Cin=" << get_named_compile_time_arg_val("in_channels")
               << " C=" << get_named_compile_time_arg_val("channels")
               << " Cout=" << get_named_compile_time_arg_val("out_channels")
               << " stride=" << get_named_compile_time_arg_val("stride")
               << " clock_MHz=" << BB_TIMING_CLOCK_MHZ << ENDL();
#endif
        line(start, count, lane, Metric::Count, elapsed);
#define BB_PRINT_METRIC(id, label) \
        if (seen & (uint32_t(1) << static_cast<uint32_t>(Metric::id))) \
            line(start, count, lane, Metric::id, cycles[static_cast<uint32_t>(Metric::id)])
        BB_PRINT_METRIC(InputLoad, "Input load incl barrier");
        BB_PRINT_METRIC(InputBarrier, "  input read barrier [subset]");
        BB_PRINT_METRIC(WeightLoad, "Weight load incl barrier");
        BB_PRINT_METRIC(WeightBarrier, "  weight read barrier [subset]");
        BB_PRINT_METRIC(Routing, "Conv1->Conv2 routing incl plan+barrier");
        BB_PRINT_METRIC(RoutingPlan, "  routing make_runs [subset]");
        BB_PRINT_METRIC(RoutingBarrier, "  routing read barrier [subset]");
        BB_PRINT_METRIC(ShortcutLoad, "Shortcut input gather/load");
        BB_PRINT_METRIC(ShortcutBarrier, "  shortcut read barrier [subset]");
        BB_PRINT_METRIC(ActivationWait, "Activation ready wait");
        BB_PRINT_METRIC(CbWait, "CB capacity/ready wait");
        BB_PRINT_METRIC(PeerSync, "Peer sync wait");
        BB_PRINT_METRIC(A2Routing, "Conv2->Conv3 routing");
        BB_PRINT_METRIC(A2Barrier, "  A2 read barrier [subset]");
        BB_PRINT_METRIC(Conv1, "Conv1 stage [inclusive]");
        BB_PRINT_METRIC(Conv2, "Conv2 stage [inclusive]");
        BB_PRINT_METRIC(Conv3, "Conv3 reduction span [inclusive]");
        BB_PRINT_METRIC(Shortcut, "Shortcut span [inclusive]");
        BB_PRINT_METRIC(Residual, "Residual add+ReLU+pack span");
        BB_PRINT_METRIC(InputWait, "  input ready wait [subset]");
        BB_PRINT_METRIC(WeightWait, "  weight ready wait [subset]");
        BB_PRINT_METRIC(DstWait, "  DST acquire/wait [subset]");
        BB_PRINT_METRIC(Matmul, "  matmul issue loop [subset]");
        BB_PRINT_METRIC(Pack, "  pack issue loop [subset]");
        BB_PRINT_METRIC(Conv1Matmul, "  Conv1 matmul issue [subset]");
        BB_PRINT_METRIC(Conv2Matmul, "  Conv2 matmul issue [subset]");
        BB_PRINT_METRIC(Conv3Matmul, "  Conv3 matmul issue [subset]");
        BB_PRINT_METRIC(ShortcutMatmul, "  Shortcut matmul issue [subset]");
        BB_PRINT_METRIC(OutputWrite, "Output DRAM write incl barrier");
        BB_PRINT_METRIC(OutputBarrier, "  output write barrier [subset]");
        BB_PRINT_METRIC(OutputWait, "Output ready wait");
#undef BB_PRINT_METRIC
    }
};
struct Scope {
    Counters& counters;
    Metric metric;
    uint32_t begin;
    Scope(Counters& c, Metric m) : counters(c), metric(m), begin(clock32()) {}
    ~Scope() {
        const uint32_t elapsed = clock32() - begin;
        const uint32_t i = static_cast<uint32_t>(metric);
        counters.cycles[i] += elapsed;
        counters.seen |= uint32_t(1) << i;
    }
};
#else
struct Counters { void print(const char*, const char*, uint32_t, uint32_t, uint32_t = 0) const {} };
struct Scope { Scope(Counters&, Metric) {} };
#endif
} // namespace bb_timing

#define BB_TIME(counters, metric, ...) \
    do { bb_timing::Scope bb_timing_scope_(counters, bb_timing::Metric::metric); __VA_ARGS__; } while (false)
