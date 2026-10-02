#pragma once
#include "block_geometry.hpp"
#include <limits>

// Small-parameter residency is an internal strategy of the SAME default op.
// If it cannot fit, the caller selects tiled L1 streaming, never DRAM stages.
namespace bottleneck_geometry {
struct DirectCachePlan {
    bool enabled = false;
    uint32_t cores = 0, block = 0, ring = 0, parameters = 0, l1_tiles = 0;
    uint32_t queue_depth = 1, single_tiles = 0, double_tiles = 0;
};
inline uint64_t parameter_count(uint32_t ci, uint32_t h, uint32_t co, bool downsample) {
    return uint64_t(ci)*h+h+uint64_t(9)*h*h+h+uint64_t(h)*co+co+
           (downsample ? uint64_t(ci)*co+co : 0);
}
inline DirectCachePlan direct_cache_plan(const Geometry &g, uint32_t ci, uint32_t h,
                                         uint32_t co, bool downsample, uint32_t cores,
                                         uint32_t requested_block, uint32_t parameter_limit,
                                         uint32_t budget_tiles) {
    DirectCachePlan p;
    p.cores=cores;
    const uint64_t parameters=parameter_count(ci,h,co,downsample);
    if (parameters>parameter_limit || !cores || !requested_block) return p;
    p.parameters=uint32_t(parameters);
    const uint32_t m_tiles=ceil32(g.output_rows());
    const uint32_t max_work=(m_tiles+cores-1)/cores;
    for (uint32_t block=min_u(requested_block,max_work); block; block/=2) {
        uint32_t ring=0;
        for (uint32_t core=0;core<cores;++core) {
            uint32_t start,count;
            core_work(m_tiles,cores,core,start,count);
            for (uint32_t done=0;done<count;done+=block) {
                const auto patch_=patch(g,start+done,min_u(block,count-done));
                ring=ring>patch_.input_mt_count?ring:patch_.input_mt_count;
            }
        }
        const uint64_t fixed=uint64_t(ring)*(ci+h)+uint64_t(block)*h+parameters+1;
        const uint64_t queues=uint64_t(18)*h+uint64_t(2)*co+(g.stride==2?uint64_t(2)*ci:0);
        const uint64_t single=fixed+queues, dual=fixed+2*queues;
        // More than one group and modest extra memory justify trying overlap.
        const uint32_t preferred=(max_work>2 && queues*4<=budget_tiles)?2:1;
        const uint32_t depth=(preferred==2 && dual<=budget_tiles)?2:1;
        const uint64_t needed=depth==2?dual:single;
        if (needed<=budget_tiles && dual<=std::numeric_limits<uint32_t>::max()) {
            p.enabled=true;p.block=block;p.ring=ring;p.l1_tiles=uint32_t(needed);
            p.queue_depth=depth;p.single_tiles=uint32_t(single);p.double_tiles=uint32_t(dual);
            return p;
        }
    }
    return p;
}
}
