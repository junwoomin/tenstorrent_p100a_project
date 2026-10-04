# P100a Custom Bottleneck — GPT로 만든 안정 버전(v7)

[English](README.md) · [프로젝트 소개](../../README_ko.md)

사용자가 현재 사용 중인 **GPT로 만든 안정 버전**이며, 제공된 `basic_block_v7_no_debug(1).zip`을 기준으로 합니다. C++와 CMake 주석을 제거했습니다. **소스 33개**에서 코드 토큰·문자열·줄 경계가 유지되는지 확인했으며, 정리된 ZIP과 공개 소스의 내용도 대조했습니다. 이는 오프라인 확인이며 SDK 빌드나 장치 검증은 수행하지 않았습니다.

## 연산과 통합

API 이름은 `basic_block`이지만 실제 연산은 **1×1 Conv + ReLU → 3×3 Conv + ReLU → 1×1 Conv → identity/projection residual Add + ReLU**인 bottleneck입니다. BatchNorm은 포함하지 않습니다. Resident·M/N-sharded·stream 실행 경로, 호스트 L1 계획, reader·compute·writer 커널, Python binding, backward 코드가 포함되어 있습니다. 이번 측정 기록은 backward나 모델 학습의 검증 결과를 포함하지 않습니다.

TTNN 내부에 통합하는 연산 소스이며 독립 빌드 프로젝트는 아닙니다.
1. 호환되는 TT-Metal 소스의 `ttnn/cpp/ttnn/operations/experimental/resnet/` 아래에 `basic_block/`을 배치합니다.
2. 상위 CMake에 디렉터리를 등록하고 experimental Python module에 `ttnn::operations::basic_block::bind_basic_block`을 연결합니다.
3. TTNN을 재빌드하고 새 호스트 라이브러리와 같은 버전의 JIT 커널 소스가 함께 로드되는지 확인합니다.

이 연산의 입력은 펼친 NHWC BF16 TILE 장치 텐서입니다. 전체 모델의 논리적 NCHW 입력과 구분되며, 가중치·bias는 연산이 기대하는 packed DRAM 텐서 형식으로 준비해야 합니다. 업로드된 ZIP에는 상위 빌드·binding 등록 변경, 전체 ResNet50 실행 코드, TTNN 비교 실행 코드가 포함되지 않았습니다. 이번 측정의 SDK 리비전도 기록되지 않았습니다.

현재 소스가 허용하는 경로 옵션은 `BASIC_BLOCK_PATH=auto|v5|sharded|stream`이며, `BASIC_BLOCK_N_SHARDS`의 0은 자동 선택입니다. 이전 기준본의 옵션 설명을 현재 소스에 맞게 교체했습니다.

## ResNet50 레이어별 시간

**P100a, 배치 8, 3채널 256 × 256 입력.** 아래 값은 2026-10-04에 사용자가 제공한 실측 출력이며, 이번 정리 과정에서 장치 실행을 다시 수행하지 않았습니다.

| 레이어 | Custom + TTNN (ms) | TTNN만 사용 (ms) |
|---|---:|---:|
| conv1 | 1.6188 | 1.5755 |
| pool1 | 0.3698 | 0.4116 |
| to_tile | 0.1380 | — |
| layer1_0 | 0.4864 | 1.6389 |
| layer1_1 | 0.5296 | 1.0819 |
| layer1_2 | 0.5209 | 1.0694 |
| layer2_0 | 1.4406 | 1.1583 |
| layer2_1 | 0.5379 | 0.6933 |
| layer2_2 | 0.5369 | 0.6855 |
| layer2_3 | 0.5044 | 0.6905 |
| layer3_0 | 1.8605 | 0.9054 |
| layer3_1 | 0.6299 | 0.5807 |
| layer3_2 | 0.5975 | 0.5812 |
| layer3_3 | 0.5977 | 0.5613 |
| layer3_4 | 0.5974 | 0.5471 |
| layer3_5 | 0.6014 | 0.5507 |
| layer4_0 | 3.4890 | 0.8672 |
| layer4_1 | 0.9544 | 0.5400 |
| layer4_2 | 0.9492 | 0.5355 |
| avgpool | 0.2745 | 0.2428 |
| **출력된 TOTAL** | **17.8934** | **15.3131** |
| 표시된 레이어 합계 | 17.2348 | 14.9168 |

“—”는 TTNN 출력에 `to_tile`이 별도로 표시되지 않았다는 뜻입니다. 변환 시간이 0이라는 의미는 아닙니다. 표시된 레이어 합계와 TOTAL이 다르며 차이의 원인은 미확인입니다. 표시된 측정 범위는 `avgpool`까지이고 FC 시간은 없습니다.

출력된 TOTAL 기준으로 커스텀 버전이 **16.85% 느립니다**. `layer1_*`와 `layer2_1–3`은 개선됐지만, 이후 블록과 단계 전환 블록에서는 지연이 증가했습니다. `layer4_0`만 비교해도 **+2.6218 ms**입니다. 두 구현의 수치 일치, TTNN 비교본의 BatchNorm 처리, warmup·표본 수·동기화·시간 통계 방식이 기록되지 않아 검증된 가속 성과로 제시하지 않습니다.

## 자료

- [주석을 제거한 ZIP](basic_block_v7_gpt_stable.zip)
- [공개 소스](basic_block/)
- [측정값](results/resnet50_layer_timing.json)
- [제공된 커스텀 시간 출력](results/custom_timing.txt) · [TTNN 시간 출력](results/ttnn_timing.txt)
- [원본·정리본 해시와 확인 내역](results/source_manifest.json)

다음 작업은 두 구현의 정확성과 측정 조건을 맞춰 기록하고 TOTAL 차이를 확인한 뒤, `layer4_0`, `layer3_0`, `layer2_0`을 프로파일링하여 최적화 방향을 정하는 것입니다.

