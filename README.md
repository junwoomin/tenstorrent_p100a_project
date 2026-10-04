# Tenstorrent P100a — ResNet50 Custom Block

[한국어](README_ko.md)

Current snapshot: **GPT-generated stable version (v7)**, based on the owner's `basic_block_v7_no_debug(1).zip`. “Stable” describes the version currently used by the owner for the reported workload; broader correctness and training validation are not established by this timing record.

The project explores ResNet bottleneck fusion, core partitioning, L1 buffer planning and data movement using TT-Metalium on P100a (Blackhole). The current custom operation is a **1×1 → 3×3 → 1×1 bottleneck without BatchNorm**.

## Current measurement

Input: **batch 8, 3 channels, 256 × 256** — logical NCHW shape `[8, 3, 256, 256]`.

| Version | Reported TOTAL |
|---|---:|
| Custom block + TTNN | 17.8934 ms |
| TTNN only | 15.3131 ms |

The custom run is **2.5803 ms (16.85%) slower** by the reported TOTAL. Some early blocks are faster, while `layer2_0`, `layer3_0` and especially `layer4_0` have higher latency. These are timing observations, not an established bottleneck diagnosis.

The displayed layer sums differ from TOTAL: **17.2348 ms** for custom and **14.9168 ms** for TTNN. The reason is unresolved. The output ends at `avgpool`; an FC classifier is not listed. Warmup, sample count, synchronization, SDK revision and matched numerical validation were not supplied.

## Code and results

- [Source and layer timings](projects/block/README.md)
- [Comment-free source ZIP](projects/block/basic_block_v7_gpt_stable.zip)
- [Timing record](projects/block/results/resnet50_layer_timing.json)
- [Source hashes and offline checks](projects/block/results/source_manifest.json)

The current tree replaces the previous VGG and Block snapshot; earlier versions remain in Git history. This update removes source comments and replaces documentation/results. It does not introduce a kernel optimization or claim a reproduced device run.

