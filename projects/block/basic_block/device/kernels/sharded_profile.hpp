#pragma once
#include "stream_profile.hpp"

namespace bottleneck_sharded {
// Payload accounting, not NoC packet overhead or hardware bus counters.
// Pull reads have no software send issue on the owner. The owner's send total
// mirrors the exact identical demand from the other N peers at its local-read
// sites. Group barriers guarantee all peers finish those requests.
struct Traffic {
#if BB_AUDIT
    uint64_t input = 0, weight = 0, local = 0, send = 0, recv = 0, output = 0;
    void input_read(uint32_t bytes) { input += bytes; }
    void weight_read(uint32_t bytes) { weight += bytes; }
    void local_read(uint32_t bytes) { local += bytes; }
    void activation_read(uint32_t owner, uint32_t lane, uint32_t peers, uint32_t bytes) {
        if (owner == lane) {
            local += bytes;
            send += uint64_t(peers - 1) * bytes;
        } else {
            recv += bytes;
        }
    }
    void output_write(uint32_t bytes) { output += bytes; }
    void print_reader() const {
#ifdef DEVICE_PRINT
        DPRINT("[BB_V6] BB_INPUT_DRAM_READ_BYTES={} BB_WEIGHT_DRAM_READ_BYTES={} "
               "BB_ACTIVATION_LOCAL_BYTES={} BB_ACTIVATION_NOC_SEND_BYTES={} "
               "BB_ACTIVATION_NOC_RECV_BYTES={}\n", input, weight, local, send, recv);
#else
        DPRINT << "[BB_V6] BB_INPUT_DRAM_READ_BYTES=" << input
               << " BB_WEIGHT_DRAM_READ_BYTES=" << weight
               << " BB_ACTIVATION_LOCAL_BYTES=" << local
               << " BB_ACTIVATION_NOC_SEND_BYTES=" << send
               << " BB_ACTIVATION_NOC_RECV_BYTES=" << recv << ENDL();
#endif
    }
    void print_writer() const {
#ifdef DEVICE_PRINT
        DPRINT("[BB_V6] BB_OUTPUT_DRAM_WRITE_BYTES={}\n", output);
#else
        DPRINT << "[BB_V6] BB_OUTPUT_DRAM_WRITE_BYTES=" << output << ENDL();
#endif
    }
#else
    void input_read(uint32_t) {}
    void weight_read(uint32_t) {}
    void local_read(uint32_t) {}
    void activation_read(uint32_t, uint32_t, uint32_t, uint32_t) {}
    void output_write(uint32_t) {}
    void print_reader() const {}
    void print_writer() const {}
#endif
};

struct ReaderTotals {
    uint64_t preload = 0, preload_barrier = 0, publish_a1 = 0, halo = 0, recv_a2 = 0, sync = 0;
    void print() const {
#if BB_PROFILE && defined(PROFILE_KERNEL)
        DeviceTimestampedData("BB_WEIGHT_PRELOAD_CYCLES", preload);
        DeviceTimestampedData("BB_WEIGHT_PRELOAD_BARRIER_CYCLES", preload_barrier);
        DeviceTimestampedData("BB_CONV1_ACTIVATION_BROADCAST_CYCLES", publish_a1);
        DeviceTimestampedData("BB_CONV2_HALO_PREPARE_CYCLES", halo);
        DeviceTimestampedData("BB_CONV2_ACTIVATION_RECV_CYCLES", recv_a2);
        DeviceTimestampedData("BB_ACTIVATION_SYNC_CYCLES", sync);
#endif
    }
};
} // namespace bottleneck_sharded
