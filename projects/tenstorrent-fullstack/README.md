# Tenstorrent Full Stack

P100a에서 모델 실행과 학습을 검증하고, compiler·runtime·operator·kernel 계층까지 추적하는 프로젝트입니다.

## 프로젝트

| 프로젝트 | 경로 | 상태 |
|---|---|---|
| P100a 구조 및 소프트웨어 스택 학습 문서 | [study](study/README.md) | 하드웨어·컴파일러·런타임 정리 완료 |
| P100a TTML / tt-train 소스 빌드 | [ttml-tt-train-build](ttml-tt-train-build/README.md) | 빌드 과정 기록 완료 |
| P100a TTML L1_SMALL 설정 패치 | [ttml-l1-small-config](ttml-l1-small-config/README.md) | 소스 수정 및 실험 구성 완료 |
| VGG11 TTML runtime 최적화 | [vgg11-ttml-optimization](vgg11-ttml-optimization/README.md) | prepared tensor·레이어별 block tuning 및 측정 경계 분리, 최종본 통합: 순전파 학습 6.11 / 검증 6.30 ms, 검증 정확도 82.88% |
| ResNet TTNN / TTML 최적화 | [resnet-ttml-optimization](resnet-ttml-optimization/README.md) | Baseline 코드 및 실험 기록 양식: 사용자 보고 약 10 ms / 55 W |
| TTML 최소 학습 예제 | 추가 예정 | 진행 예정 |
| TT-XLA 실행 경로 분석 | 추가 예정 | 진행 예정 |
| TT-NN custom operator | 추가 예정 | 진행 예정 |
| TT-Metalium custom kernel | 추가 예정 | 진행 예정 |

## 실행 경로

### TTML Training

```text
Application / Model
  → TTML / tt-train
  → Autograd / Optimizer
  → TTNN
  → TT-Metal
  → P100a
```

### TT-XLA Compilation

```text
PyTorch
  → Torch-XLA
  → StableHLO
  → TT-MLIR
  → TTNN / TT-Metal
  → P100a
```

두 경로를 분리해서 기록하고, 최종적으로 동일하거나 유사한 모델의 실행 결과와 개발 난이도를 비교합니다.

## 핵심 분석 항목

- PyTorch CPU/GPU baseline
- TTML training correctness
- TT-XLA compile 및 실행 결과
- TT-NN operator coverage
- custom TT-Metalium operator/kernel
- 정확도 오차와 compile time
- training/inference latency
- SRAM/DRAM traffic
- core utilization

## 진행 원칙

- 실행 환경과 software stack 버전을 반드시 기록합니다.
- 성공 로그뿐 아니라 빌드 및 런타임 오류 해결 과정도 남깁니다.
- 서로 다른 precision의 이론 성능을 직접 동일시하지 않습니다.
- 정확도와 수치 일치 여부를 확인한 뒤 성능을 비교합니다.
