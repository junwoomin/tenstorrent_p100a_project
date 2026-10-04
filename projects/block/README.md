# P100a Custom Bottleneck — GPT-generated stable version (v7)

[한국어](README_ko.md) · [Project overview](../../README.md)

This is the owner's currently used **GPT-generated stable version**, supplied as `basic_block_v7_no_debug(1).zip`. All C++ and CMake comments have been removed. Code tokens, string literals and line boundaries are preserved across all **33 source files**; the cleaned ZIP has also been checked against the published sources. These are offline checks, not an SDK build or hardware validation.

## Operation and integration

The API is named `basic_block`, but implements a bottleneck: **1×1 Conv + ReLU → 3×3 Conv + ReLU → 1×1 Conv → identity/projection residual Add + ReLU**, without BatchNorm. The source includes resident, M/N-sharded and stream execution paths, host L1 planning, reader/compute/writer kernels, Python bindings and backward code. Backward and model training are outside the supplied timing evidence.

The operation is integrated into TTNN; this is not a standalone build:
1. Place `basic_block/` under `ttnn/cpp/ttnn/operations/experimental/resnet/` in a compatible TT-Metal source checkout.
2. Register its directory in the parent CMake configuration and connect `ttnn::operations::basic_block::bind_basic_block` to the experimental Python module.
3. Rebuild TTNN and ensure the rebuilt host library and matching JIT kernel sources are loaded.

The input to this operation is a flattened NHWC BF16 TILE device tensor, distinct from the full model's logical NCHW input. Weights and biases must follow the operation's packed DRAM tensor contract. Parent build/binding changes, the full ResNet50 runner and the TTNN-only runner were not included in the uploaded archive. The SDK revision for this run is not recorded.

This source accepts `BASIC_BLOCK_PATH=auto|v5|sharded|stream` and `BASIC_BLOCK_N_SHARDS` (0 selects automatically). These exact values replace the previous snapshot's instructions.

## ResNet50 layer timing

**P100a; batch 8; 3-channel 256 × 256 input.** Values below are the owner's supplied timing output, received on 2026-10-04, and have not been independently rerun.

| Layer | Custom + TTNN (ms) | TTNN only (ms) |
|---|---:|---:|
| conv1 | 1.6188 | 1.5755 |
| pool1 | 0.3698 | 0.4116 |
| to_tile | 0.1380 | — |
| layer1_0 | 0.4864 | 1.6389 |
| layer1_1 | 0.5296 | 1.0819 |
| layer1_2 | 0.5209 | 1.0694 |
| layer2_0 | 1.4406 | 1.1583 |
| layer2_1 | 0.5379 | 0.6933 |
| layer2_2 | 0.5369 | 0.6855 |
| layer2_3 | 0.5044 | 0.6905 |
| layer3_0 | 1.8605 | 0.9054 |
| layer3_1 | 0.6299 | 0.5807 |
| layer3_2 | 0.5975 | 0.5812 |
| layer3_3 | 0.5977 | 0.5613 |
| layer3_4 | 0.5974 | 0.5471 |
| layer3_5 | 0.6014 | 0.5507 |
| layer4_0 | 3.4890 | 0.8672 |
| layer4_1 | 0.9544 | 0.5400 |
| layer4_2 | 0.9492 | 0.5355 |
| avgpool | 0.2745 | 0.2428 |
| **Reported TOTAL** | **17.8934** | **15.3131** |
| Sum of displayed layers | 17.2348 | 14.9168 |

“—” means that the TTNN output did not separately list `to_tile`, not that conversion took zero time. The displayed layer sums do not equal the reported TOTAL; the cause is unresolved. The listed scope ends at `avgpool`, with no FC timing.

The reported TOTAL is **16.85% slower** for custom. Early `layer1_*` and `layer2_1–3` improve, while later blocks and the stage-transition blocks have regressions. `layer4_0` alone differs by **+2.6218 ms**. Numerical equivalence, BatchNorm handling in the TTNN baseline, warmup, sample count, synchronization and timing statistic have not been recorded, so the table is a development measurement rather than a validated speedup comparison.

## Artifacts

- [Comment-free ZIP](basic_block_v7_gpt_stable.zip)
- [Published source](basic_block/)
- [Timing values](results/resnet50_layer_timing.json)
- [Original timing outputs](results/custom_timing.txt) · [TTNN output](results/ttnn_timing.txt)
- [Original/cleaned source hashes and checks](results/source_manifest.json)

Next: record matched correctness and timing conditions, explain the TOTAL discrepancy, then profile `layer4_0`, `layer3_0` and `layer2_0` before choosing a performance change.

