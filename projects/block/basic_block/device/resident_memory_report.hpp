#pragma once
#include "host_common.hpp"
namespace ttnn::operations::basic_block::detail {
inline std::string resident_l1_report(const Attributes &a, const bottleneck_geometry::DirectCachePlan &p) {
    const uint64_t ci=tiles(a.in_channels), h=tiles(a.channels), co=tiles(a.out_channels), bytes=2048;
    std::ostringstream s;
    s << std::fixed << std::setprecision(2);
    const auto row=[&](const char *label, uint64_t value) {
        s << std::left << std::setw(39) << label << ": " << std::right << std::setw(10)
          << double(value)/1024 << " KiB (" << value << " B)\n";
    };
    s << "================ L1 Memory Plan ================\n"
      << "strategy: small parameter residency; cores=" << p.cores << " block=" << p.block << " ring=" << p.ring << "\n";
    row("Available L1/core",a.l1_available_bytes);
    row("CB metadata / firmware / reserved base",a.l1_reserved_bytes);
    row("Input / source ring (CB22)",p.ring*ci*bytes);
    row("Activation buffer A (CB20)",p.ring*h*bytes);
    row("Activation buffer B (CB21)",p.block*h*bytes);
    row("Halo window FIFO (CB11)",18*p.queue_depth*h*bytes);
    row("All parameters (CB23)",p.parameters*bytes);
    row("  Conv1 weights+bias",(ci*h+h)*bytes);
    row("  Conv2 weights+bias",(9*h*h+h)*bytes);
    row("  Conv3 weights+bias",(h*co+co)*bytes);
    row("  Shortcut weights+bias",has_downsample(a)?(ci*co+co)*bytes:0);
    row("Accumulation/final output (CB16)",2*p.queue_depth*co*bytes);
    row("Residual input FIFO (CB9)",a.stride==2?2*p.queue_depth*ci*bytes:0);
    row("Zero / other temporary (CB24)",bytes);
    const uint64_t peak=uint64_t(p.l1_tiles)*bytes;
    row("Phase Conv1 (physical reservation)",peak);
    row("Phase Conv2 (physical reservation)",peak);
    row("Phase Conv3 (physical reservation)",peak);
    row("Phase Add (physical reservation)",peak);
    row("Peak simultaneous physical allocation",peak);
    row("Remaining margin",a.l1_available_bytes>peak?a.l1_available_bytes-peak:0);
    row("Same geometry single FIFO candidate",uint64_t(p.single_tiles)*bytes);
    row("Same geometry double FIFO candidate",uint64_t(p.double_tiles)*bytes);
    s << "Selected resident FIFO depth=" << p.queue_depth << "\n";
    s << "Parameters are small and reused across this core's patches. No stage DRAM path.\n"
      << "Both strategies select FIFO depth from reuse opportunity and actual L1 budget.\n"
      << "================================================\n";
    return s.str();
}
}
