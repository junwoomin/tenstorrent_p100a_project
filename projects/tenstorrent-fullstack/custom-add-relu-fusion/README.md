# Custom Add + ReLU Fusion Benchmark on P100a

## 목적과 범위

본 실험은 residual block에서 사용하는 Add와 ReLU를 대상으로, 개별 TTNN operation 실행과 built-in fusion 및 직접 구현한 custom fusion의 지연시간을 비교한다. 간단한 custom operation의 구현 및 실행 성능을 확인하는 단계이며, ResNet block 및 전체 모델 적용은 후속 실험이다.

첨부 소스에는 forward/backward C++ 커널, nanobind 바인딩 및 TTML autograd 연결 코드가 포함된다. 아래 결과는 **forward 연산만** 대상으로 한다. Backward 정확도·성능 및 전체 ResNet 가속은 본 결과에서 검증하지 않았다.

## 비교 조건

장치는 P100a이며 입력은 BF16, TILE layout, interleaved DRAM이다. 다음 shape는 사용자가 제공한 tensor shape 그대로 기록하며 이미지의 NCHW/NHWC 의미를 별도로 가정하지 않는다. 이 연산은 동일 shape의 원소별 계산이다.

```python
separate = ttnn.relu(ttnn.add(a, b))

builtin = ttnn.add(
    a, b,
    activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
)

custom = ttnn.custom_add_relu(a, b)
```

사용자 제공 median을 기록하였다. 측정에 사용한 Python benchmark 스크립트는 이번 압축에 포함되지 않았다. Warm-up 횟수, 반복 횟수, 동기화 위치, host timer 또는 device profiler 사용 여부, software/firmware 버전 및 정확도 비교 로그는 추가 기록이 필요하다. 따라서 아래 시간을 순수 device kernel 실행 시간으로 단정하지 않는다.

## 결과

단위는 ms이며 speedup은 동일 행의 median 비율로 계산하였다.

| Shape | Add | ReLU | Separate | Built-in fused | Custom fused |
|---|---:|---:|---:|---:|---:|
| (1, 64, 640, 640) | 0.4540 | 0.3349 | 0.7453 | 0.4538 | 0.4295 |
| (8, 64, 640, 640) | 3.3039 | 2.3814 | 5.5919 | 3.1979 | 3.1853 |
| (1, 64, 320, 320) | 0.1517 | 0.0999 | 0.2191 | 0.1429 | 0.1378 |
| (8, 64, 320, 320) | 0.8477 | 0.6623 | 1.4295 | 0.8405 | 0.8222 |
| (1, 64, 160, 160) | 0.0859 | 0.0474 | 0.0977 | 0.0758 | 0.0514 |
| (8, 64, 160, 160) | 0.2416 | 0.1734 | 0.4051 | 0.2555 | 0.2180 |

| Shape | Separate / Built-in | Separate / Custom | Built-in / Custom |
|---|---:|---:|---:|
| (1, 64, 640, 640) | 1.64× | 1.74× | 1.06× |
| (8, 64, 640, 640) | 1.75× | 1.76× | 1.00× |
| (1, 64, 320, 320) | 1.53× | 1.59× | 1.04× |
| (8, 64, 320, 320) | 1.70× | 1.74× | 1.02× |
| (1, 64, 160, 160) | 1.29× | 1.90× | 1.47× |
| (8, 64, 160, 160) | 1.59× | 1.86× | 1.17× |

원시 median 표는 [CSV](results/forward_medians.csv)로도 제공한다. 단독 Add와 ReLU의 median 합은 연속 실행한 Separate의 median과 동일한 통계량이 아니므로 합산하여 대체하지 않는다.

## 커널 구성

| 구성 | 역할 |
|---|---|
| [device_operation.cpp](custom_add_relu/device/device_operation.cpp) | 입력 검증 및 출력 tensor 생성 |
| [program_factory.cpp](custom_add_relu/device/program_factory.cpp) | tile 작업 분배, Circular Buffer 및 reader/compute/writer 구성 |
| [reader.cpp](custom_add_relu/device/kernels/reader.cpp) | 입력 A/B의 tile을 읽어 CB에 공급 |
| [compute.cpp](custom_add_relu/device/kernels/compute.cpp) | Add → ReLU → pack 수행 |
| [writer.cpp](custom_add_relu/device/kernels/writer.cpp) | 결과 tile 저장 |
| [nanobind](custom_add_relu/custom_add_relu_nanobind.cpp) | Python forward/backward API 연결 |
| [TTML bridge](custom_add_relu/python/_custom_add_relu_autograd.py) | TTML 입력의 autograd 연결 |

Compute는 각 core에 배정된 tile을 순회하면서 `add_tiles()`, `relu_tile()`, `pack_tile()`를 실행한다. Add 결과를 별도 DRAM tensor로 저장했다가 ReLU 입력으로 다시 읽는 경로를 제거한다. Reader, compute, writer는 별개의 장치 커널이며 이를 하나의 fused operation/program으로 구성한다.

멀티코어 분배는 `split_work_to_cores(grid, num_tiles)`에서 결정한다. Host는 각 core에 처리할 tile 수와 시작 tile을 전달하며, 각 core의 루프는 자기 구간을 처리한다. CB depth는 2이며 입력 2개와 출력 1개 CB를 구성한다.

현재 지원 범위는 동일 장치·동일 rank-4 shape, BF16, 기본 32×32 TILE, padding 없는 interleaved DRAM 입력 및 출력이다. 마지막 두 차원은 32의 배수여야 한다. Broadcast, scalar 입력 및 sharded 입력을 지원하는 범용 binary operation과는 지원 범위가 다르다.

## 분석

### Fusion 효과

Built-in fusion은 Separate 대비 **1.29–1.75×**, custom fusion은 **1.59–1.90×**의 median 비율을 보였다. 측정된 모든 shape에서 fusion의 지연시간이 낮았다.

Custom 구현에서는 중간 tensor의 DRAM 저장·재읽기를 제거하고 하나의 operation으로 실행한다. 이 구조는 메모리 트래픽과 operation 실행 비용 감소를 설명하는 근거이지만, 각 비용의 기여도는 별도로 측정하지 않았다.

### Custom과 built-in의 차이

`(1, 64, 160, 160)`에서 built-in은 0.0758 ms, custom은 0.0514 ms로 약 **1.47×** 차이가 나타났다. 반면 `(8, 64, 640, 640)`에서는 각각 3.1979 ms와 3.1853 ms로 custom의 지연시간이 약 **0.39%** 낮았다.

작은 입력에서 차이가 더 커지는 경향은 고정 실행 비용 차이와 양립한다. 다만 이 결과만으로 Python dispatch, validation 또는 program setup이 원인이라고 확정할 수 없다. Core 배분, 데이터 이동, cache 상태, 구현 경로 및 측정 변동도 영향을 줄 수 있다. 큰 입력에서 DRAM transfer가 지배적이라는 설명 역시 profiler로 확인할 가설이다.

Custom의 제한된 지원 범위는 범용 구현과의 차이지만, 실제로 어느 처리 경로가 생략되었고 몇 μs를 줄였는지는 별도 profiling이 필요하다. 특히 0.39% 차이는 반복 실행 분산이 없으므로 유의한 우위로 주장하지 않는다.

## 재현 및 검증 상태

[설치 연결 안내](INTEGRATION.md)를 참고한다. 첨부 README가 언급하던 `install.py`, `tests/`, `examples/`, benchmark 실행 스크립트는 실제 압축에 없어 실행 가능한 파일로 안내하지 않는다.

소스는 첨부본을 보존하였다. 이번 저장소 정리에서는 Python bridge의 문법만 검사했으며 네이티브 빌드, P100a 실행 및 수치 정확도 검사를 수행하지 않았다. 사용자 보고 median과 코드 검토 결과를 구분한다. 첨부 문서에 적힌 참조 tt-metal commit은 `58648d44443223041148125843ca7c7d4780463e`이나 실제 측정에 사용한 commit인지는 확인되지 않았다.

## 다음 단계: ResNet block

[ResNet baseline](../resnet-ttml-optimization/README.md)의 residual Add → ReLU를 대상으로 Separate, built-in fused, custom fused를 비교할 예정이다. 현재 baseline의 residual 함수는 변경하지 않았다.

- 동일 입력·가중치에서 출력 오차 및 정확도를 비교한다.
- 실제 residual tensor의 layout·memory·padding 조건을 확인한다.
- 필요한 변환 비용까지 포함한 block latency와 전체 모델 latency를 측정한다.
- 연산 단독 이득이 block 및 전체 모델에서도 유지되는지 확인한다.
- Backward 적용 시 gradient 정확도와 TTML 그래프 연결을 별도 검증한다.

단독 microbenchmark 결과를 ResNet 전체 가속으로 확대 해석하지 않는다.
