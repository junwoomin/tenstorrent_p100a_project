#pragma once
#include <cstdint>

// Reader-private CB25: one allocated 2048-byte page. The aligned landing
// region uses <=127 bytes even if CB placement is only 16-byte aligned.
// Callers hold the reservation for the invocation; compute never accesses it.
namespace bb_dram_gather {
constexpr uint32_t scratch_cb = 25;
constexpr uint32_t scratch_bytes = 2048;
struct TrafficBytes { uint32_t dram = 0, local = 0; };
inline TrafficBytes read_face_rows(uint64_t from, uint32_t to, uint32_t length, uint32_t landing) {
    ASSERT(length && length % 32 == 0);
    if ((from & uint64_t(63)) == (to & uint32_t(63))) {
        noc_async_read(from, to, length);
        return {length, 0}; // Caller's existing final barrier completes this read.
    }
    ASSERT(landing && (landing & 63) == 0);
    TrafficBytes bytes;
    for (uint32_t offset = 0; offset < length; offset += 32) {
        const uint64_t row = from + offset;
        // A face row is 32 bytes and starts at offset 0 or 32 in a 64-byte
        // DRAM line. The containing line stays within the 2048-byte tile.
        noc_async_read(row & ~uint64_t(63), landing, 64);
        noc_async_read_barrier(); // DRAM data must arrive before local copy.
        noc_async_read(get_noc_addr(landing + uint32_t(row & 63)), to + offset, 32);
        noc_async_read_barrier(); // Do not overwrite landing while read in flight.
        bytes.dram += 64;
        bytes.local += 32;
    }
    return bytes;
}
} // namespace bb_dram_gather
