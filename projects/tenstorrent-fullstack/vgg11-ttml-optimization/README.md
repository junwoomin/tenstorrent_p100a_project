# VGG11 TTNN / TTML Optimization on P100a

## 개요

본 프로젝트는 Tenstorrent P100a에서 ImageNet 사전학습 VGG11 backbone을 TTNN으로 실행하고, TTML의 3층 분류기만 학습하는 전이학습 실험이다. 입력 처리와 장치 연산의 측정 범위를 분리하고, prepared weight/bias 재사용, 레이어별 block 및 shard 설정을 검토하였다. 전체 VGG backbone의 역전파 학습이나 custom Metalium kernel 구현은 수행하지 않았다.

첨부된 최종 6개 소스 파일을 기준으로 통합하였다. 과거 문서는 실험 이력이며, 현재 실행 설정은 이 README 및 src/를 기준으로 한다.

최종 구현은 [AI From Scratch / VGG](../../ai-from-scratch/vgg/README.md)에서도 실행할 수 있다. 해당 경로의 `src/`에는 본 프로젝트 최종 소스와 동일한 6개 Python 파일을 수록하였다. 이 폴더는 최적화 이력과 분석 자료를 유지한다.

## 최종 보고 결과

다음 값은 사용자가 제공한 실행 로그이다. 저장소 정리 환경에서 재측정하지 않았다.

| 지표 | 학습 | 검증 |
|---|---:|---:|
| Loss | 0.0511 | 0.6611 |
| 정확도 | 98.64% | 82.88% |
| Backbone 순전파 평균 | 5.85 ms | 5.98 ms |
| Classifier 순전파 평균 | 0.27 ms | 0.31 ms |
| 순전파 합계 / batch | **6.11 ms** | **6.30 ms** |
| hwmon 전력 표본 평균 | 65.9 W | 63.4 W |

전체 소요 시간은 **166.3초(2.77분)**이며 학습·검증·checkpoint 저장을 포함하고 데이터 로더 생성, 가중치 준비 등 초기 준비는 제외한다. 표시된 부분 시간과 합계의 0.01 ms 차이는 각 값의 반올림으로 발생할 수 있다.

검증 정확도는 Oxford-IIIT Pet의 공식 test split 결과가 아니다. 코드에서 trainval을 품종별 80:20으로 분리한 내부 검증 결과이다. 학습·검증 정확도 차이는 15.76%p이며, 일반화 성능 개선은 별도 과제로 남긴다. 이 결과만으로 최적화 전후 수치 동등성을 증명하지 않는다.

## 최종 실행 설정

| 항목 | 값 |
|---|---|
| 장치 | P100a / Blackhole, device 0 |
| 입력 | batch 8, 224 × 224, NHWC |
| 정밀도 | bfloat16 preset |
| 학습 | 고정 VGG11 + 학습 가능한 512 → 512 → 37 분류기 |
| Epoch / seed | 10 / 42 |
| Optimizer | AdamW, lr 0.001, weight decay 0.0001 |
| Warm-up | 각 epoch의 학습·검증에서 각각 처음 10 batches 제외 |
| L1_SMALL | 0 (사용자 패치된 TTML API 인자 사용) |
| Conv1 block override | **64** |
| Conv1 activation double buffer | **True** |
| Conv2–8 block override | 0 |
| HEIGHT_SHARDED 명시 | Conv1·Conv2·Conv4 |
| reshard / deallocate activation | Conv1만 True |
| Weight double buffer | 모두 False |
| Conv / Pool config tensor | DRAM |
| Conv 출력 memory_config | **DRAM_MEMORY_CONFIG** |
| MaxPool 출력 memory_config | **DRAM_MEMORY_CONFIG** |
| Prepared weight/bias | 첫 호출에서 준비 후 재사용 |

HEIGHT_SHARDED 지정과 출력 DRAM 배치는 별도 설정이다. 최종 코드를 “모든 activation을 L1에 유지한 모델”이라고 표현하지 않는다. 과거 L1 출력 실험과 구분한다. 또한 일반적인 block 크기 축소가 shard 크기나 코어 배분 축소를 곧바로 의미하지 않는다.

## 후속 관찰: 출력 위치와 명시적 메모리 변환

본 workload의 후속 실험에서는 불필요한 메모리 변환을 반복하지 않는 경우 L1 출력과 DRAM 출력 사이에 큰 지연시간 차이가 관찰되지 않았다. 반면 MaxPool 출력을 L1에 생성한 뒤 별도의 `to_memory_config(..., ttnn.DRAM_MEMORY_CONFIG)`로 DRAM으로 옮기는 경로에서는 추가 지연이 관찰되었다. 모든 중간 activation을 L1에 유지하려는 구성은 L1 용량 제약을 받았다.

| 비교 항목 | 관찰 및 최종 선택 |
|---|---|
| L1 출력과 DRAM 출력 | 불필요한 변환이 없을 때 큰 성능 차이를 관찰하지 못함 |
| MaxPool L1 출력 후 별도 DRAM 변환 | 명시적 변환을 추가한 경로에서 지연 증가 |
| 전체 activation의 L1 유지 | 실험 구성에서 L1 공간 부족 |
| 최종 구성 | Conv·MaxPool 호출에 DRAM 출력을 지정하고 후속의 별도 DRAM 변환 호출 제거 |

이에 따라 최종 구현에서는 저장 위치를 일률적으로 L1으로 바꾸기보다, 필요한 출력 메모리 구성을 연산 호출에서 직접 지정하여 연산 사이의 불필요한 명시적 메모리 변환을 최소화하였다. 출력 DRAM 지정이 내부 L1 사용이나 모든 데이터 이동을 제거한다는 의미는 아니다.

이 관찰은 해당 모델과 실행 구성에 한정된다. L1과 DRAM의 물리적 대역폭이 같다는 결론이나 L1 배치가 항상 불필요하다는 일반화는 지지하지 않는다. 별도 변환 경로의 지연에는 데이터 복사뿐 아니라 allocation, layout/shard 변환 및 dispatch 비용 등이 포함될 수 있으며 각각의 기여도는 확인하지 않았다.

후속 비교의 조건별 수치와 반복 측정 로그는 제공되지 않았으므로 정성적 관찰로 기록한다. 기존 8.3 → 7.3 ms는 L1 및 sharding 등의 결합 설정 결과로 보존하며, 이를 L1 배치 단독 효과나 NoC 이동 감소의 증거로 해석하지 않는다.

## 측정 방법과 한계

- Dataset에서 이미지 정규화, CHW → HWC 변환 및 host dtype 변환을 수행한다.
- 각 batch의 TTNN 변환 및 H2D 전송은 타이머 시작 전에 수행한다.
- 장치를 동기화한 뒤 backbone을 측정하고, 다시 동기화한 뒤 classifier를 측정한다.
- 합계는 두 구간 평균의 합이다. 중간 동기화가 없는 단일 전체-forward 측정과 동일하지 않다.
- Loss, backward, optimizer step, metric D2H 및 저장은 6.11/6.30 ms에 포함되지 않는다. “학습 배치 전체 시간”이 아니다.
- 배치 평균은 표본 수 가중 평균이 아닌 batch 단위 평균이다. 마지막 부분 batch도 포함된다.
- 전력은 warm-up 이후 각 batch 순전파 측정이 끝난 시점의 hwmon 값을 읽어 산술평균한다. 전체 PC 전력, 시간 가중 평균, 순전파 구간 에너지 또는 최대 전력이 아니다.
- 전력 센서는 hwmon 이름과 정렬 순서로 선택하므로 다중 장치에서는 PCI 주소 기반 매핑을 별도로 검증해야 한다.
- 평균 latency만 보고되었으며 p50/p95와 반복 실행 간 분산은 아직 측정되지 않았다.

## 최적화 이력

| 단계 | 보고 latency | 해석 |
|---|---:|---|
| 초기 구성 | 약 28 ms | 입력 변환·전송 포함 |
| Prepared tensor 재사용 | 약 21 ms | 입력 변환·전송 포함 |
| Block 자동 선택 | 약 18 ms | 입력 변환·전송 포함 |
| 입력 경로를 측정 밖으로 분리 | 약 10 ms | 측정 범위 정정 |
| Conv1 block 256 등 레이어 설정 | 약 8.3 ms | device forward 실험 |
| Conv1 L1 및 HEIGHT_SHARDED 실험 | 약 7.3 ms | 결합 설정 |
| Conv1 block 128 + Conv1·2·4 shard | 약 6.6 ms | 결합 설정 |
| 최종 첨부 소스 | 학습 6.11 / 검증 6.30 ms | backbone·classifier 분리 측정 합계 |

28 → 6.11 ms를 동일 조건의 가속 배율로 계산하지 않는다. 측정 범위 및 구성 변화가 포함되어 있다. 최종 소스에는 Conv1 block 64, activation double buffering, DRAM 출력, 공통 Conv 인자 사전 구성 등 여러 변경이 존재하며, 개별 기여도는 분리 검증하지 않았다. 재분배·NoC 이동 감소 역시 profiler로 확인되지 않은 가설이다.

## 실행 및 파일

기존 [TTML 빌드 환경](../ttml-tt-train-build/README.md)과 [L1_SMALL API 패치](../ttml-l1-small-config/README.md)를 준비한다. 최종 소스는 Python 3.12 환경을 기준으로 하며 정확한 tt-metal commit, 패키지 및 firmware 버전은 실행 환경에서 추가 기록해야 한다.

```bash
conda activate ttml
cd projects/tenstorrent-fullstack/vgg11-ttml-optimization/src
python -m unittest test_dataset -v
python train_ttml.py
```

데이터가 없으면 train_ttml.py의 DOWNLOAD를 True로 변경한다. WEIGHTS=None이면 ImageNet VGG11 가중치를 내려받는다. 그 외 환경 경로는 빌드 문서에 따른다.

| 파일 | 역할 |
|---|---|
| src/dataset.py | 데이터 다운로드·품종별 분할·NHWC 전처리 |
| src/dtype_config.py | dtype/layout 프리셋, 기본 BF16 |
| src/test_dataset.py | 분할·부분 batch·NHWC 검증 |
| src/vgg.py | TTNN Conv/Pool/Linear 및 prepared tensor 재사용 |
| src/vgg_ttml.py | 고정 backbone과 TTML classifier |
| src/train_ttml.py | 학습·검증·시간/전력 측정·checkpoint |

checkpoint는 run/ttml/의 backbone.pt, best.pt, last.pt에 저장한다. BF8/FP32 프리셋은 소스에 남겨 두었지만 위 결과는 BF16에만 해당하며 다른 프리셋의 실행 성공을 보증하지 않는다.

## 검증 상태

주석과 docstring을 제거했으며 제거 전후 AST에서 실행 로직의 동일성을 검사하였다. 유일한 동작 수정은 test_dataset.py의 오래된 NCHW 기대 shape를 실제 NHWC (2, 32, 32, 3)으로 정정한 것이다. 6개 파일 모두 문법 검사를 통과했다. 정리 환경에 torch 및 Tenstorrent 장치가 없어 단위 테스트와 장치 실행은 수행하지 못했다.

## 실험 자료

- [레이어별 프로파일·block·shard 실험](LAYERWISE_PROFILING.md)
- [L1_SMALL 실험](L1_SMALL_TUNING.md)
- [메모리 대역폭 측정 및 병목 가설](MEMORY_BOTTLENECK_EVIDENCE.md)
- [대역폭 벤치마크 코드](benchmarks/p100a-bandwidth/README.md)

본 단계는 VGG runtime 설정 최적화의 정리 지점이다. 향후 실험은 동일한 측정 범위에서 정확도 동등성, 반복 측정 분산 및 개별 옵션의 기여도를 검증하는 데 초점을 둔다.
