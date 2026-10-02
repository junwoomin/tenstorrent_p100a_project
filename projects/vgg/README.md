# VGG11 on P100a

[한국어](README_ko.md)

Run an ImageNet-pretrained VGG11 backbone with TTNN and train a three-layer TTML classifier on Oxford-IIIT Pet (37 classes). The backbone is frozen; the classifier is trained.

## Work completed

- Built the dataset split, NHWC preprocessing, training/validation, checkpoint, latency, and power measurement pipeline.
- Reused prepared convolution weights and biases across calls.
- Tuned `act_block_h_override`, activation double buffering, and HEIGHT_SHARDED placement for Conv1, Conv2, and Conv4.
- Explored L1_SMALL configuration and removed unnecessary explicit memory conversions. The final Conv/MaxPool outputs and configuration tensors use DRAM.
- Separated input conversion/upload from timed device forward execution.

## Reported results

Batch 8, 224 × 224 NHWC, BF16, 10 epochs, seed 42. Validation uses a class-stratified 80:20 split of trainval, rather than the official test split.

| Metric | Training | Validation |
|---|---:|---:|
| Accuracy | 98.64% | 82.88% |
| Loss | 0.0511 | 0.6611 |
| Backbone forward mean | 5.85 ms | 5.98 ms |
| Classifier forward mean | 0.27 ms | 0.31 ms |
| Forward sum per batch | 6.11 ms | 6.30 ms |
| Sampled device power mean | 65.9 W | 63.4 W |

These are historical run results. Forward time sums separately synchronized backbone/classifier measurements and excludes preprocessing, upload, loss, backward, and optimizer steps. Power is the mean of hwmon samples, not whole-system power. Earlier timings used different measurement boundaries and are not used as a speedup ratio.

## Source and execution

`src/` contains the dataset, dtype configuration, TTNN VGG, TTML wrapper, training script, and dataset tests. The previously duplicated VGG sources are consolidated here without changing execution logic.

Requires the original source-built TTNN/TTML environment (Python 3.12) and P100a. The training script uses the custom TTML `open_device(..., l1_small_size=0)` API; the corresponding TTML binding must be available. The VGG run's exact TT-Metal revision was not recorded.

```bash
cd projects/vgg/src
python -m unittest test_dataset -v
python train_ttml.py
```

Check `DATA_ROOT`, `DOWNLOAD`, and `WEIGHTS` in `train_ttml.py`. Set `DOWNLOAD=True` if the dataset is absent. `WEIGHTS=None` downloads pretrained weights. Checkpoints are written to `run/ttml/` relative to the working directory.
