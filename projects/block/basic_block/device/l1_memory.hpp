#pragma once






#include <algorithm>
#include <cstdint>
#include <limits>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt_stl/assert.hpp>



#if __has_include("impl/allocator/allocator.hpp")
#include "impl/allocator/allocator.hpp"
#define BB_HAS_ALLOCATOR_INTERNALS 1
#elif __has_include("tt_metal/impl/allocator/allocator.hpp")
#include "tt_metal/impl/allocator/allocator.hpp"
#define BB_HAS_ALLOCATOR_INTERNALS 1
#else
#define BB_HAS_ALLOCATOR_INTERNALS 0
#endif

namespace ttnn::operations::basic_block::detail {


struct L1MemorySnapshot {
    uint32_t available_bytes = 0;
    uint32_t reserved_bytes = 0;
};

namespace l1_detail {

inline uint32_t checked_u32(uint64_t value) {
    TT_FATAL(
        value <= std::numeric_limits<uint32_t>::max(), "L1 address/size exceeds uint32: {}", value
    );
    return static_cast<uint32_t>(value);
}




template <class Device>
void include_device_occupancy(
    const Device* device, const tt::tt_metal::CoreRangeSet& cores, uint64_t& lower, uint64_t& upper
) {
    using namespace tt::tt_metal;
    if (const auto occupied = device->lowest_occupied_compute_l1_address(); occupied.has_value()) {
        upper = std::min(upper, uint64_t(*occupied));
    }
#if BB_HAS_ALLOCATOR_INTERNALS

    if constexpr (requires { device->allocator_impl()->persistent_l1().high_water_mark(cores); }) {
        lower = std::max(
            lower, uint64_t(device->allocator_impl()->persistent_l1().high_water_mark(cores))
        );
    }

    if constexpr (requires {
                      device->allocator_impl()->get_lowest_occupied_l1_address(uint32_t{0});
                  }) {
        const auto& alloc = device->allocator_impl();
        for (const auto& range : cores.ranges()) {
            for (const auto& core : range) {
                for (const auto bank :
                     alloc->get_bank_ids_from_logical_core(BufferType::L1, core)) {
                    if (const auto occupied = alloc->get_lowest_occupied_l1_address(bank);
                        occupied.has_value()) {
                        upper = std::min(upper, uint64_t(*occupied));
                    }
                }
            }
        }
    }
#else
    (void)cores;
    (void)lower;
#endif
}
}




template <class Device>
L1MemorySnapshot query_l1_memory(const Device* device, const tt::tt_metal::CoreRangeSet& cores) {
    using namespace tt::tt_metal;
    TT_FATAL(device != nullptr, "Cannot query L1 budget without a device");
    TT_FATAL(
        BB_HAS_ALLOCATOR_INTERNALS,
        "Strict L1 budgeting requires tt-metal allocator implementation headers. "
        "Build this operation in the matching tt-metal source tree; public-only SDK headers cannot "
        "reveal the persistent L1 arena and are not accepted."
    );
    if constexpr (requires {
                      device->get_active_sub_device_manager_id();
                      device->get_default_sub_device_manager_id();
                  }) {
        TT_FATAL(
            device->get_active_sub_device_manager_id() ==
                device->get_default_sub_device_manager_id(),
            "basic_block strict L1 plan currently requires the default sub-device manager"
        );
    }
    const auto& alloc = device->allocator();
    const uint64_t total =
        std::min(uint64_t(device->l1_size_per_core()), uint64_t(alloc->get_worker_l1_size()));
    const uint64_t base = alloc->get_base_allocator_addr(HalMemType::L1);

    uint64_t small = 0;
#if BB_HAS_ALLOCATOR_INTERNALS
    if constexpr (requires { device->allocator_impl()->get_config().l1_small_size; }) {
        small = device->allocator_impl()->get_config().l1_small_size;
    } else {
        TT_FATAL(
            false,
            "Cannot read l1_small_size from this SDK allocator; strict L1 budgeting is unavailable"
        );
    }
#endif


    const uint64_t alignment = alloc->get_alignment(BufferType::DRAM);
    TT_FATAL(
        small <= total && base <= total && alignment != 0,
        "Invalid L1 allocator bounds: total={} base={} L1_SMALL={} alignment={}",
        total,
        base,
        small,
        alignment
    );
    TT_FATAL(
        2048 % alignment == 0,
        "Unsupported CB alignment {}: BF16-tile allocation accounting requires it to divide 2048",
        alignment
    );
    uint64_t lower = base;
    uint64_t occupied_cap = total;
    l1_detail::include_device_occupancy(device, cores, lower, occupied_cap);
    if constexpr (requires { device->get_devices(); }) {
        for (const auto* physical : device->get_devices()) {
            l1_detail::include_device_occupancy(physical, cores, lower, occupied_cap);
        }
    }
    lower = ((lower + alignment - 1) / alignment) * alignment;
    const uint64_t cap = std::min(total - small, occupied_cap);
    const uint64_t available = cap > lower ? cap - lower : 0;
    return {
        .available_bytes = l1_detail::checked_u32(available),
        .reserved_bytes = l1_detail::checked_u32(lower),
    };
}
}
