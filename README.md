# Tenstorrent P100a Projects

[한국어](README_ko.md)

Implementation and optimization work on the Tenstorrent P100a (Blackhole), using TTNN, TTML, and TT-Metalium.

| Project | Work |
|---|---|
| [VGG11](projects/vgg/README.md) | Frozen TTNN backbone, TTML classifier training, and runtime/memory configuration tuning. |
| [Custom Block](projects/block/README.md) | Fused convolution/residual bottleneck with resident, sharded, and stream execution paths. |

Each project contains its source and a short record of the work and results.
