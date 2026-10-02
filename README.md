# Tenstorrent P100a Projects

[한국어](README_ko.md)

A progression from learning the basic TTNN configuration and execution flow with VGG11 to developing dedicated ResNet bottleneck kernels with TT-Metalium on the Tenstorrent P100a (Blackhole).

| Project | Work |
|---|---|
| [VGG11](projects/vgg/README.md) | First TTNN project: understand the basic configuration and execution flow, then tune VGG11 and train a TTML classifier. |
| [Custom Block](projects/block/README.md) | Dedicated ResNet bottleneck kernel development, with fused convolution/residual operations and resident, sharded, and stream paths. |

Each project contains its source and a short record of the work and results.
