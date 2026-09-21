# VGG11 on P100a — 최종 구현

## 개요

P100a에서 ImageNet 사전학습 VGG11 backbone을 TTNN으로 실행하고 TTML 3층 분류기를 학습하는 최종 구현이다. [VGG11 최적화 프로젝트](../../tenstorrent-fullstack/vgg11-ttml-optimization/README.md)의 최종 Python 소스 6개를 `src/`에 수록하였다.

본 구현은 고정 backbone을 이용한 전이학습이며 VGG 전체를 처음부터 학습한 결과가 아니다. 최적화 과정, 메모리 배치 실험 및 측정 한계는 기존 최적화 문서에서 관리한다.

## 최종 보고 결과

| 지표 | 학습 | 검증 |
|---|---:|---:|
| Loss | 0.0511 | 0.6611 |
| 정확도 | 98.64% | 82.88% |
| Backbone 순전파 평균 | 5.85 ms | 5.98 ms |
| Classifier 순전파 평균 | 0.27 ms | 0.31 ms |
| 순전파 합계 / batch | **6.11 ms** | **6.30 ms** |
| hwmon 전력 표본 평균 | 65.9 W | 63.4 W |

사용자가 제공한 실행 결과이며 이번 파일 정리에서 재측정하지 않았다. 전체 소요 시간은 초기 준비를 제외하고 학습·검증·저장을 포함한 166.3초이다. 검증은 Oxford-IIIT Pet trainval의 품종별 80:20 내부 분할이며 공식 test split 결과가 아니다.

순전파 합계는 별도로 동기화하여 측정한 backbone과 classifier 평균의 합이다. 입력 전처리·H2D 전송, loss, backward 및 optimizer step을 포함하지 않는다. 전력은 순전파 이후 hwmon 표본의 평균이며 전체 PC 소비전력이 아니다.

## 최종 설정

Batch 8, 입력 224 × 224 NHWC, BF16, 10 epochs, seed 42를 사용한다. Conv1은 block override 64와 activation double buffering을 사용하며 Conv1·2·4에 HEIGHT_SHARDED를 명시한다. Conv와 MaxPool의 config tensor 및 출력은 DRAM으로 지정하고 L1_SMALL은 0이다.

해당 workload에서는 불필요한 메모리 변환을 반복하지 않을 때 L1 출력과 DRAM 출력의 큰 성능 차이가 관찰되지 않았다. MaxPool L1 출력 후 별도 DRAM 변환은 추가 지연을 발생시켰고, 전체 activation의 L1 배치는 용량 제약을 받았다. 이는 L1과 DRAM의 물리적 성능이 동일하다는 의미가 아니다.

## 실행

[TTML 빌드 환경](../../tenstorrent-fullstack/ttml-tt-train-build/README.md)과 [L1_SMALL API 패치](../../tenstorrent-fullstack/ttml-l1-small-config/README.md)를 준비한 뒤 저장소 루트에서 실행한다.

```bash
conda activate ttml
cd projects/ai-from-scratch/vgg/src
python -m unittest test_dataset -v
python train_ttml.py
```

DATA_ROOT와 DOWNLOAD 설정을 확인한다. 데이터가 없으면 DOWNLOAD=True로 변경한다. WEIGHTS=None이면 사전학습 가중치를 내려받는다. Checkpoint는 실행 디렉터리 기준 `run/ttml/`에 저장된다.

| 파일 | 역할 |
|---|---|
| [src/dataset.py](src/dataset.py) | 데이터 분할 및 NHWC 전처리 |
| [src/dtype_config.py](src/dtype_config.py) | dtype/layout 설정 |
| [src/test_dataset.py](src/test_dataset.py) | 데이터 분할 및 shape 검증 |
| [src/vgg.py](src/vgg.py) | TTNN 연산 및 prepared weight 재사용 |
| [src/vgg_ttml.py](src/vgg_ttml.py) | 고정 backbone과 TTML 분류기 |
| [src/train_ttml.py](src/train_ttml.py) | 학습·검증·시간·전력 측정 |

소스는 기존 최종본과 동일한 주석 제거본이다. 이번 작업에서 실행 로직을 변경하지 않았으며 P100a 실행은 재검증하지 않았다.

## 실험 기록

- [최적화 이력과 상세 측정 조건](../../tenstorrent-fullstack/vgg11-ttml-optimization/README.md)
- [레이어별 프로파일링](../../tenstorrent-fullstack/vgg11-ttml-optimization/LAYERWISE_PROFILING.md)
- [메모리 병목 분석](../../tenstorrent-fullstack/vgg11-ttml-optimization/MEMORY_BOTTLENECK_EVIDENCE.md)
