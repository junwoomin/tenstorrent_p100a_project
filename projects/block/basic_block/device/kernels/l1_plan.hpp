#pragma once






#include <algorithm>
#include <cstdint>
#include <limits>
#include "block_geometry.hpp"

namespace bottleneck_geometry {
constexpr uint64_t bf16_tile_bytes = 2048;



struct StreamPlan {
    bool enabled = false;
    uint32_t cores = 0, block = 0, ring = 0;
    uint32_t processing_tiles = 2, cb_depth = 2, weight_depth = 2, out_depth = 2;
    uint32_t weight_chunk = 1, operand_slot_tiles = 0, weight_slot_tiles = 0;
    uint64_t operand_bytes = 0, activation_a_bytes = 0, activation_b_bytes = 0;
    uint64_t weight_bytes = 0, output_bytes = 0, zero_bytes = bf16_tile_bytes;
    uint64_t required_bytes = 0, available_bytes = 0;
};



inline uint32_t maximum_patch_tiles(const Geometry& g, uint32_t cores, uint32_t block) {
    uint32_t ring = 0;
    const uint32_t total = ceil32(g.output_rows());
    for (uint32_t core = 0; core < cores; ++core) {
        uint32_t start, count;
        core_work(total, cores, core, start, count);
        for (uint32_t done = 0; done < count; done += block) {
            ring = std::max(
                ring, patch(g, start + done, std::min(block, count - done)).input_mt_count
            );
        }
    }
    return ring;
}



inline void account_stream_plan(StreamPlan& p, uint32_t cin, uint32_t hidden, uint32_t cout) {
    const uint64_t m = p.processing_tiles, bytes = bf16_tile_bytes;
    const uint64_t operand_slot = m * std::max<uint64_t>(cin, uint64_t(9) * hidden);
    p.operand_slot_tiles =
        operand_slot > std::numeric_limits<uint32_t>::max() ? 0 : uint32_t(operand_slot);
    p.weight_slot_tiles = p.weight_chunk + 1;
    p.operand_bytes = operand_slot * p.cb_depth * bytes;
    p.activation_a_bytes = uint64_t(p.ring) * hidden * bytes;
    p.activation_b_bytes = m * hidden * bytes;
    p.weight_bytes = uint64_t(p.weight_slot_tiles) * p.weight_depth * bytes;
    p.output_bytes = m * cout * p.out_depth * bytes;
    p.required_bytes = p.operand_bytes + p.activation_a_bytes + p.activation_b_bytes +
                       p.weight_bytes + p.output_bytes + p.zero_bytes;
    const uint64_t max_cb = std::max(
        {p.operand_bytes,
         p.activation_a_bytes,
         p.activation_b_bytes,
         p.weight_bytes,
         p.output_bytes,
         p.zero_bytes}
    );
    p.enabled = p.operand_slot_tiles && max_cb <= std::numeric_limits<uint32_t>::max() &&
                p.required_bytes <= p.available_bytes;
}




inline StreamPlan streaming_plan(
    const Geometry& g,
    uint32_t cin,
    uint32_t hidden,
    uint32_t cout,
    uint32_t cores,
    uint32_t requested_block,
    uint32_t parameter_limit,
    uint64_t available_bytes
) {
    StreamPlan p;
    p.available_bytes = available_bytes;
    if (!cores || !requested_block || !parameter_limit || !cin || !hidden || !cout || !g.batch ||
        !g.height || !g.width || (g.stride != 1 && g.stride != 2)) {
        return p;
    }
    p.cores = std::min(cores, ceil32(g.output_rows()));
    p.block = std::min(requested_block, (ceil32(g.output_rows()) + p.cores - 1) / p.cores);
    p.weight_chunk = std::min(parameter_limit, std::max(cin, 9 * hidden));
    const auto update = [&] {
        p.ring = maximum_patch_tiles(g, p.cores, p.block);
        account_stream_plan(p, cin, hidden, cout);
    };
    update();


    const uint32_t max_core_work = (ceil32(g.output_rows()) + p.cores - 1) / p.cores;
    p.cb_depth = p.operand_bytes <= available_bytes / 4 &&
                         (p.ring > p.processing_tiles || p.block > p.processing_tiles)
                     ? 2
                     : 1;
    p.weight_depth = p.weight_bytes <= available_bytes / 8 &&
                             (std::max(cin, 9 * hidden) > p.weight_chunk || hidden > 1 || cout > 1)
                         ? 2
                         : 1;
    p.out_depth =
        p.output_bytes <= available_bytes / 8 && max_core_work > p.processing_tiles ? 2 : 1;
    update();
    if (p.enabled) {
        return p;
    }



    while (p.block > 1) {
        p.block = std::max(1u, p.block / 2);

        update();
        if (p.enabled) {
            return p;
        }
    }

    p.cb_depth = 1;
    update();
    if (p.enabled) {
        return p;
    }

    p.weight_depth = 1;
    update();
    if (p.enabled) {
        return p;
    }

    p.out_depth = 1;
    update();
    if (p.enabled) {
        return p;
    }
    while (p.weight_chunk > 1) {
        p.weight_chunk = std::max(1u, p.weight_chunk / 2);

        update();
        if (p.enabled) {
            return p;
        }
    }
    p.processing_tiles = 1;

    update();
    return p;
}
}
