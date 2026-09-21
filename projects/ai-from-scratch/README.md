# AI From Scratch

딥러닝 모델을 단순 호출하지 않고 핵심 구조, 학습 과정, 평가 및 성능 측정을 직접 구현하는 트랙입니다.

## 프로젝트

| 순서 | 프로젝트 | 핵심 주제 | 상태 |
|---:|---|---|---|
| 1 | [VGG](vgg/README.md) | CNN 구조 및 P100a 전이학습 최종 구현 | 최종 코드 수록; 순전파 학습 6.11 / 검증 6.30 ms |
| 2 | [ResNet](resnet/README.md) | residual connection과 downsampling | 시작 전 |
| 3 | [YOLO](yolo/README.md) | object detection과 NMS | 시작 전 |
| 4 | [U-Net](unet/README.md) | semantic segmentation과 skip connection | 시작 전 |
| 5 | [Multi-task](multitask/README.md) | shared backbone 기반 detection + segmentation | 시작 전 |

## 권장 진행 순서

`VGG → ResNet → YOLO → U-Net → Multi-task`

VGG는 CNN의 기본 흐름을 확인하는 짧은 프로젝트로 진행하고, 첫 번째 심화 프로젝트는 ResNet-18로 설정합니다.

## 공통 완료 조건

각 프로젝트는 다음 조건을 충족해야 완료로 처리합니다.

- [ ] 모델 핵심 구조 직접 구현
- [ ] tensor shape unit test
- [ ] parameter 및 FLOPs 검증
- [ ] 학습·평가 명령 제공
- [ ] 정확도 지표 기록
- [ ] latency 및 peak memory 측정
- [ ] profiler 기반 병목 분석
- [ ] 결과와 한계 문서화

## 프로젝트 내부 표준 구조

코드 구현을 시작할 때 각 프로젝트 폴더 안에 다음 구조를 만듭니다.

```text
src/
tests/
configs/
scripts/
benchmarks/
reports/
README.md
```

빈 폴더만 미리 만들지 않고 실제 코드나 결과가 생길 때 함께 추가합니다.
