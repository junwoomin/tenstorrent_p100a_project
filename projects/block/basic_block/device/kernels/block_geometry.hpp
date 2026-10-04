#pragma once






#include <cstdint>

namespace bottleneck_geometry {

inline uint32_t ceil32(uint32_t n) {
    return n / 32 + (n % 32 != 0);
}

inline uint32_t min_u(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}



struct Geometry {
    uint32_t batch, height, width, stride;

    uint32_t out_height() const {
        return (height - 1) / stride + 1;
    }

    uint32_t out_width() const {
        return (width - 1) / stride + 1;
    }

    uint32_t input_rows() const {
        return batch * height * width;
    }

    uint32_t output_rows() const {
        return batch * out_height() * out_width();
    }



    uint32_t center(uint32_t position) const {
        const uint32_t area = out_height() * out_width();
        const uint32_t image = position / area, local = position % area;
        return (image * height + (local / out_width()) * stride) * width +
               (local % out_width()) * stride;
    }
};



struct Patch {
    uint32_t input_mt_begin, input_mt_count, output_mt_begin, output_mt_count;
};




inline Patch patch(const Geometry& g, uint32_t output_mt_begin, uint32_t output_mt_count) {
    const uint32_t first = g.center(output_mt_begin * 32);
    const uint32_t last =
        g.center(min_u((output_mt_begin + output_mt_count) * 32, g.output_rows()) - 1);

    const uint32_t begin = first > g.width + 1 ? first - g.width - 1 : 0;
    const uint32_t end = min_u(last + g.width + 2, g.input_rows());
    const uint32_t start_tile = begin / 32;
    return {start_tile, ceil32(end) - start_tile, output_mt_begin, output_mt_count};
}



inline void
core_work(uint32_t total, uint32_t cores, uint32_t core, uint32_t& start, uint32_t& count) {
    const uint32_t q = total / cores, r = total % cores;
    count = q + (core < r);
    start = core * q + min_u(core, r);
}



inline uint32_t channel_shard_owner(uint32_t channel, uint32_t total, uint32_t shards) {
    if (!shards || channel >= total) {
        return shards;
    }
    const uint32_t q = total / shards, r = total % shards;
    const uint32_t wide = (q + 1) * r;
    return channel < wide ? channel / (q + 1) : r + (channel - wide) / q;
}



struct CoreShard {
    uint32_t m_start = 0, m_count = 0;
    uint32_t n_start = 0, n_count = 0;
};



inline CoreShard core_shard(
    uint32_t total_m,
    uint32_t total_n,
    uint32_t m_shards,
    uint32_t n_shards,
    uint32_t m_index,
    uint32_t n_index
) {
    CoreShard shard;
    if (!m_shards || !n_shards || m_index >= m_shards || n_index >= n_shards) {
        return shard;
    }
    core_work(total_m, m_shards, m_index, shard.m_start, shard.m_count);
    core_work(total_n, n_shards, n_index, shard.n_start, shard.n_count);
    return shard;
}
}
