# ResNet TTNN / TTML Optimization on P100a

## 개요

본 프로젝트는 P100a에서 ResNet backbone을 TTNN으로 실행하고 TTML 분류기를 학습하는 실행 및 분류기 학습 실험이다. `x(2).zip`의 Python 소스 10개로 `src/`의 baseline을 교체하였다. 이후 레이어별 프로파일링과 메모리 배치 실험을 본 경로에서 기록한다.

Backbone 가중치는 TTNN의 난수 생성으로 초기화한 뒤 고정하며 3층 분류기만 학습한다. 현재 baseline에는 사전학습 가중치 로딩 및 PyTorch 그래프 비교 경로가 없다. 따라서 사전학습 backbone을 이용한 전이학습 또는 ResNet 전체 역전파 학습으로 기술하지 않는다. Residual 경로의 `add_relu`는 `ttnn.add`와 `ttnn.relu`를 순서대로 호출하며 custom fused kernel은 사용하지 않는다.

## 보고 결과

현재 `x(2).zip` baseline의 지연시간·전력·정확도 결과는 아직 제공되지 않았다.

| 버전 | Latency | 전력 | 상태 |
|---|---:|---:|---|
| 이전 x(1).zip | 약 9.5 ms | 기존 보고 약 55 W | 과거 사용자 보고; 동일 실행의 측정값인지는 미확인 |
| 현재 x(2).zip | 미보고 | 미보고 | 새 baseline |

기존 수치는 이력으로 보존하며 새 코드의 실측 결과로 간주하지 않는다.

## 현재 코드 기본 설정

다음은 첨부 코드의 기본값이다. 실제 측정 결과는 실행 로그와 함께 추가한다.

| 항목 | 기본값 |
|---|---|
| 모델 | ResNet-50; 18/50/101 선택 가능 |
| 장치 | P100a, device 0 |
| 입력 | batch 8, 224 × 224, NHWC |
| 정밀도 | BF16, ROW_MAJOR preset |
| 데이터 | Oxford-IIIT Pet, 37개 품종 |
| 학습 | 무작위 초기화 후 고정한 backbone + 3층 TTML 분류기, hidden 512 |
| Epoch / seed | 10 / 42 |
| Optimizer | AdamW, lr 0.001, weight decay 0.0001 |
| Warm-up | 학습·검증 각 epoch의 처음 10 batches |
| L1_SMALL | 0 |
| Conv / MaxPool config tensor | DRAM |
| Conv / Pool 기본 출력 | DRAM |

`optimization_config.py`의 자동 규칙은 연산의 채널·kernel·stride 조건에 따라 적용된다. VGG 이름의 규칙도 포함되어 있으므로 이름만으로 ResNet에서 미적용이라고 판단하지 않는다. 실제 레이어 설정은 defaults, 자동 규칙, 명시적 override를 반영한 값으로 확인해야 한다.

## 측정 범위

`train_ttml.py`는 입력의 TTNN 변환과 H2D 전송을 타이머 밖에서 수행한다. 장치 동기화 후 backbone과 classifier의 순전파를 각각 측정하고 평균을 합산한다. Loss, backward, optimizer step, metric 회수 및 checkpoint 저장은 이 합계에 포함하지 않는다. 따라서 순전파 합계를 학습 step 전체 지연시간으로 해석하지 않는다.

전력은 warm-up 이후 순전파가 끝난 시점의 hwmon 표본 평균이다. 전체 PC 전력이나 시간 가중 평균 에너지를 의미하지 않는다. 현재 baseline의 결과는 측정 구간과 원본 로그를 함께 기록한다.

## 실행

[TTML 빌드 환경](../ttml-tt-train-build/README.md) 및 코드에서 사용하는 [L1_SMALL API 패치](../ttml-l1-small-config/README.md)를 준비한다.

저장소 루트에서:

```bash
conda activate ttml
cd projects/tenstorrent-fullstack/resnet-ttml-optimization/src
python train_ttml.py
```

`train_ttml.py`에서 DATA_ROOT, DOWNLOAD, RESNET_DEPTH를 확인한다. 기본 DOWNLOAD는 False이다. Backbone은 무작위 초기화되며 WEIGHTS 설정은 제거되었다. Checkpoint는 현재 작업 디렉터리의 `run/resnet50_ttml/`에 저장된다. 모델 깊이를 변경하면 저장 경로도 변경된다.

## 소스 구성

| 경로 | 역할 |
|---|---|
| [src/models/resnet_ttnn.py](src/models/resnet_ttnn.py) | TTConv2d, residual block 및 ResNet-18/50/101 |
| [src/models/ttnn_config.py](src/models/ttnn_config.py) | Conv/Pool 설정 적용 |
| [src/models/residual.py](src/models/residual.py) | Add 후 ReLU 실행 |
| [src/models/__init__.py](src/models/__init__.py) | 모델 패키지 초기화 |
| [src/resnet.py](src/resnet.py) | 모델 export |
| [src/resnet_ttml.py](src/resnet_ttml.py) | 고정 backbone과 TTML 분류기 |
| [src/train_ttml.py](src/train_ttml.py) | 학습·검증·시간·전력 측정 |
| [src/dataset.py](src/dataset.py) | 데이터 분할과 전처리 |
| [src/dtype_config.py](src/dtype_config.py) | dtype/layout preset |
| [src/optimization_config.py](src/optimization_config.py) | 연산 설정과 자동 규칙 |

이전 baseline의 `compat/`, 참조 모델, layer factory 및 호환성 검사 스크립트 등 새 압축에 없는 소스는 제거하였다. 주석과 docstring 제거 후 10개 Python 파일의 문법 및 정규화 AST 동일성을 검사하였다. 실제 P100a 실행과 성능 재측정은 수행하지 않았다.

Checkpoint에는 분류기 가중치와 설정이 저장되며 backbone 가중치는 포함되지 않는다. 동일 backbone 복원을 위한 별도 저장·복원 경로는 현재 구현에 없다.

## 최적화 기록

새 코드를 기준으로 후속 실험을 [EXPERIMENTS.md](EXPERIMENTS.md)에 기록한다. 이전 9.5 ms 결과와 비교할 때 가중치 초기화 방식 및 측정 조건의 차이를 명시한다.

## 관련 프로젝트

- [VGG 최적화 과정과 측정 범위](../vgg11-ttml-optimization/README.md)
- [VGG 최종 구현](../../ai-from-scratch/vgg/README.md)
- [기존 ResNet baseline 소스](../resnet-baseline/)
