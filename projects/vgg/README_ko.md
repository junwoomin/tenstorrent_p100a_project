# P100a VGG11

[English](README.md)

ImageNet 사전학습 VGG11 backbone을 TTNN으로 실행하고 Oxford-IIIT Pet 37개 품종을 분류하는 3층 TTML 분류기를 학습했습니다. Backbone은 고정하고 분류기만 학습합니다.

## 수행한 작업

- 데이터 분할, NHWC 전처리, 학습·검증, checkpoint 저장, 시간·전력 측정 파이프라인 구현
- 준비한 Conv weight·bias를 호출마다 재사용
- `act_block_h_override`, activation double buffering, Conv1·2·4의 HEIGHT_SHARDED 설정 조정
- L1_SMALL 설정 실험과 불필요한 명시적 메모리 변환 제거. 최종 Conv·MaxPool 출력과 configuration tensor는 DRAM 사용
- 입력 변환·전송과 장치 순전파 측정 구간 분리

## 보고된 결과

Batch 8, 입력 224 × 224 NHWC, BF16, 10 epochs, seed 42. 검증은 trainval의 품종별 80:20 내부 분할입니다. 공식 test split 결과는 아닙니다.

| 지표 | 학습 | 검증 |
|---|---:|---:|
| 정확도 | 98.64% | 82.88% |
| Loss | 0.0511 | 0.6611 |
| Backbone 순전파 평균 | 5.85 ms | 5.98 ms |
| Classifier 순전파 평균 | 0.27 ms | 0.31 ms |
| 배치당 순전파 합계 | 6.11 ms | 6.30 ms |
| 장치 전력 표본 평균 | 65.9 W | 63.4 W |

기존 실행에서 보고된 결과입니다. 순전파 시간은 각각 동기화한 backbone·classifier 측정값의 합이며 전처리·전송·loss·backward·optimizer step은 제외합니다. 전력은 hwmon 표본 평균으로 전체 PC 전력이 아닙니다. 초기 실험은 측정 범위가 달라 최종값과 가속 배율로 비교하지 않습니다.

## 코드와 실행

`src/`에 데이터셋, dtype 설정, TTNN VGG, TTML wrapper, 학습 코드와 데이터셋 테스트를 모았습니다. 기존 중복 VGG 코드는 실행 로직 변경 없이 이 경로로 통합했습니다.

기존 소스 빌드 TTNN·TTML 환경(Python 3.12)과 P100a가 필요합니다. 학습 코드는 사용자 패치인 TTML `open_device(..., l1_small_size=0)` API를 사용하므로 해당 binding이 필요합니다. VGG 실행 당시의 정확한 TT-Metal revision은 기록되지 않았습니다.

```bash
cd projects/vgg/src
python -m unittest test_dataset -v
python train_ttml.py
```

`train_ttml.py`의 `DATA_ROOT`, `DOWNLOAD`, `WEIGHTS`를 확인합니다. 데이터가 없으면 `DOWNLOAD=True`로 설정합니다. `WEIGHTS=None`이면 사전학습 가중치를 다운로드합니다. Checkpoint는 실행 디렉터리 기준 `run/ttml/`에 저장됩니다.
