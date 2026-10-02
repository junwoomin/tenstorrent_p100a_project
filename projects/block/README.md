# Custom Bottleneck Block on P100a

[한국어](README_ko.md)

Building on the TTNN experience from VGG11, this project develops dedicated kernels for a ResNet bottleneck block inside TTNN using TT-Metalium. The goal is to design the block's computation, data movement, and L1 buffer use directly.

Implemented a TT-Metalium operation that combines 1×1 Conv + ReLU, 3×3 Conv + ReLU, 1×1 Conv, identity/projection shortcut, and residual Add + ReLU. The API retains the name `basic_block`; its computation is a ResNet-style bottleneck without BatchNorm.

## Work completed

- Implemented CPU geometry/L1 planning and device reader, compute, and writer kernels.
- Kept intermediate forward activations in L1 circular buffers; inputs, parameters, and final output use DRAM.
- Added resident, M/N-sharded resident, and chunk-stream execution paths with L1 capacity checks.
- Implemented Python bindings, a recomputing backward path, and diagnostic timing/traffic instrumentation.
- Isolated a stride-2 projection correctness error to DRAM/L1 address congruence, then fixed stream/sharded gathers using an aligned landing buffer accounted for in the L1 plan.

## Baseline source

`basic_block/` contains the **2026-09-30 forward correctness-fixed baseline**, preserved from the source associated with the successful P100a validation. The eight-file correction matches the applied patch; source and tested-library hashes are in [results/source_manifest.json](results/source_manifest.json). Later static-routing and v7 refactoring variants are excluded from this baseline.

Validated workload: batch 8, H=W=32, Cin=512, hidden=256, Cout=1024, stride=2; BF16 TILE, HiFi4, FP32 accumulation, seed 42. Stream and sharded passed dense/sparse projection checks and small control cases. Dense comparison used `rtol=0.05`, `atol=0.02`; custom vs Torch maximum error was 0.0078125.

This stability designation applies to the tested forward workloads. Backward code is retained, but its related gather alignment risk and full-model training remain unverified.

## Measured baseline

Each row is a separate session with 50 warmups and 200 synchronized eager-forward samples per candidate. Values are median milliseconds, including host dispatch and completion; compilation, uploads, and weight preparation are excluded.

| Forced path | TTNN separate | TTNN builtin fused | Custom |
|---|---:|---:|---:|
| Stream | 0.7458 | 0.6821 | 1.2735 |
| Sharded | 0.7047 | 0.6382 | 2.2793 |

The corrected custom block is slower than builtin fused on this workload. Older measurements with different shapes or incorrect outputs are not presented as this baseline's speedup. [Validation summary](results/validation_summary.json) and [stream](results/stream_samples.json)/[sharded](results/sharded_samples.json) raw samples preserve the supporting results.

## Integration

Validated environment: P100a, firmware 19.13.1, KMD 2.10.0, Python 3.12.14, TT-Metal `02fef3de6ebbf2e3d4a02452932d647b7e06c7af` with local integration changes.

This is an in-tree TTNN operation. Place `basic_block/` under `ttnn/cpp/ttnn/operations/experimental/resnet/`, register it in the parent CMake, and call `ttnn::operations::basic_block::bind_basic_block` from the experimental Python binding module. Rebuild TTNN and ensure the new host library and matching JIT kernel sources are loaded together. Parent registration changes and a standalone benchmark harness are not included.

The operation expects flattened NHWC BF16 TILE matrices, packed weights/biases, and interleaved DRAM storage. `[W1, W2, W3, Wshortcut]` and corresponding biases use the original packing contract; shortcut parameters are used for projection. `BASIC_BLOCK_PATH=auto|resident|sharded|stream` selects the path. This repository cleanup preserves kernel bytes and does not rerun device tests.
