# ResNet TTNN / TTML Optimization on P100a

## 개요

본 프로젝트는 P100a에서 ResNet backbone을 TTNN으로 실행하고 TTML 분류기를 학습하는 전이학습 실험이다. 기존 ResNet baseline의 Python 소스 24개를 실행 구조 그대로 `src/`에 복사하였다. 이후 레이어별 프로파일링과 메모리 배치 실험을 본 경로에서 기록한다.

Backbone은 고정하며 3층 분류기만 학습한다. ResNet 전체의 역전파 학습 결과가 아니다. PyTorch 참조 모델과 독립 TTNN 모델, 그래프 비교 및 가중치 변환 코드를 포함한다.

## 초기 보고 결과

| 항목 | 사용자 보고값 |
|---|---:|
| Latency | 약 10 ms |
| 전력 | 약 55 W |

두 값은 사용자가 제공한 초기 기준값이며 본 저장소 정리 과정에서 재측정하지 않았다. 학습·검증 구분, backbone 단독 또는 분류기 포함 여부와 측정 당시의 정확한 설정은 원본 로그를 추가하여 확정해야 한다. 정확도, loss, p50/p95 및 반복 측정 분산은 아직 보고되지 않았다. VGG 결과와 직접적인 가속 배율을 계산하지 않는다.

## 현재 코드 기본 설정

다음은 첨부 코드의 기본값이다. 위 보고값의 실행 조건이 모두 확인되었다는 뜻은 아니다.

| 항목 | 기본값 |
|---|---|
| 모델 | ResNet-50; 18/50/101 선택 가능 |
| 장치 | P100a, device 0 |
| 입력 | batch 8, 224 × 224, NHWC |
| 정밀도 | BF16, ROW_MAJOR preset |
| 데이터 | Oxford-IIIT Pet, 37개 품종 |
| 학습 | 고정 pretrained backbone + 3층 TTML 분류기, hidden 512 |
| Epoch / seed | 10 / 42 |
| Optimizer | AdamW, lr 0.001, weight decay 0.0001 |
| Warm-up | 학습·검증 각 epoch의 처음 10 batches |
| L1_SMALL | 0 |
| Conv / MaxPool config tensor | DRAM |
| Conv / Pool 기본 출력 | DRAM |

`optimization_config.py`의 자동 규칙은 연산의 채널·kernel·stride 조건에 따라 적용된다. VGG 이름의 규칙도 포함되어 있으므로 이름만으로 ResNet에서 미적용이라고 판단하지 않는다. 실제 레이어 설정은 defaults, 자동 규칙, 명시적 override를 반영한 값으로 확인해야 한다.

## 측정 범위

`train_ttml.py`는 입력의 TTNN 변환과 H2D 전송을 타이머 밖에서 수행한다. 장치 동기화 후 backbone과 classifier의 순전파를 각각 측정하고 평균을 합산한다. Loss, backward, optimizer step, metric 회수 및 checkpoint 저장은 이 합계에 포함하지 않는다. 따라서 순전파 합계를 학습 step 전체 지연시간으로 해석하지 않는다.

전력은 warm-up 이후 순전파가 끝난 시점의 hwmon 표본 평균이다. 전체 PC 전력이나 시간 가중 평균 에너지를 의미하지 않는다. 약 10 ms와 55 W가 이 출력의 어느 구간에서 취득되었는지는 로그와 함께 추가 기록한다.

## 실행

[TTML 빌드 환경](../ttml-tt-train-build/README.md) 및 코드에서 사용하는 [L1_SMALL API 패치](../ttml-l1-small-config/README.md)를 준비한다.

저장소 루트에서:

```bash
conda activate ttml
cd projects/tenstorrent-fullstack/resnet-ttml-optimization/src
python train_ttml.py
```

`train_ttml.py`에서 DATA_ROOT, DOWNLOAD, WEIGHTS, RESNET_DEPTH를 확인한다. 기본 DOWNLOAD는 False이며 WEIGHTS=None이면 사전학습 가중치를 사용한다. Checkpoint는 현재 작업 디렉터리의 `run/resnet50_ttml/`에 저장된다. 모델 깊이를 변경하면 저장 경로도 변경된다.

## 구조 및 검증 도구

| 경로 | 역할 |
|---|---|
| [src/train_ttml.py](src/train_ttml.py) | 학습·검증 및 시간·전력 측정 |
| [src/resnet_ttml.py](src/resnet_ttml.py) | 고정 backbone과 TTML 분류기 |
| [src/models/](src/models/) | PyTorch/TTNN ResNet 및 연산 래퍼 |
| [src/compat/](src/compat/) | 그래프 비교, BN fusion 및 가중치 매핑, forward 비교 |
| [src/optimization_config.py](src/optimization_config.py) | Conv/Pool 기본값과 자동 설정 규칙 |
| [src/layer_factory.py](src/layer_factory.py) | 연산 생성 및 설정 적용 |
| [src/dataset.py](src/dataset.py) | 데이터 분할과 전처리 |
| [src/dtype_config.py](src/dtype_config.py) | dtype/layout preset |
| [src/tests/](src/tests/) | CPU 참조 및 fake TTNN 기반 테스트 |

`src/`에서 다음 명령을 사용할 수 있다.

```bash
python -m unittest discover -s tests -t . -v
python test_model_compatibility.py --depth 50 --image-size 224 --batch-size 8 --cpu-weights
python test_model_compatibility.py --depth 50 --image-size 224 --batch-size 8 --pretrained --device
```

CPU 및 fake TTNN 검증은 실제 P100a 실행 검증을 대체하지 않는다. 이번 작업은 기존 주석 제거본을 그대로 복사한 것으로 실행 로직을 변경하지 않았다. 장치 실행과 성능 재측정은 수행하지 않았다.

## 최적화 기록

| 단계 | 변경 사항 | Latency | 전력 | 상태 |
|---|---|---:|---:|---|
| Baseline | 첨부 ResNet 코드 | 약 10 ms | 약 55 W | 사용자 보고; 상세 로그 추가 예정 |

추가 실험은 [EXPERIMENTS.md](EXPERIMENTS.md)에 기록한다. 실제 측정 전의 최적화 계획을 완료 결과로 기재하지 않는다.

## 관련 프로젝트

- [VGG 최적화 과정과 측정 범위](../vgg11-ttml-optimization/README.md)
- [VGG 최종 구현](../../ai-from-scratch/vgg/README.md)
- [기존 ResNet baseline 소스](../resnet-baseline/)
