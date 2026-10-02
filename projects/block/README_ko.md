# P100a Custom Bottleneck Block

[English](README.md)

1×1 Conv + ReLU → 3×3 Conv + ReLU → 1×1 Conv에 identity/projection shortcut과 residual Add + ReLU를 결합한 TT-Metalium 연산을 구현했습니다. API 이름은 `basic_block`이지만 실제 구조는 BatchNorm 없는 ResNet형 bottleneck입니다.

## 수행한 작업

- CPU의 형상·L1 계획과 장치 reader·compute·writer 커널 구현
- 순전파 중간 activation은 L1 circular buffer에 유지하고 입력·파라미터·최종 출력은 DRAM 사용
- Resident, M/N-sharded resident, chunk-stream 세 경로와 L1 용량 검사 구현
- Python binding, activation 재계산 방식 backward, 시간·traffic 진단 계측 구현
- Stride 2 projection 정확성 오류를 DRAM·L1 주소 정렬 조건으로 좁히고, stream·sharded gather에 정렬된 임시 버퍼를 적용해 L1 계획에 반영

## 기준 코드

`basic_block/`에는 **2026-09-30 순전파 정확성 수정 기준본**을 넣었습니다. P100a 검증에 연결된 소스를 그대로 보존했으며 8개 파일의 수정은 실제 적용한 patch와 일치합니다. 소스·검증 라이브러리 해시는 [results/source_manifest.json](results/source_manifest.json)에 기록했습니다. 이후 static-routing 실험본과 v7 리팩토링본은 이 기준본에 포함하지 않았습니다.

검증 조건: batch 8, H=W=32, Cin=512, hidden=256, Cout=1024, stride=2; BF16 TILE, HiFi4, FP32 accumulation, seed 42. Stream·sharded의 dense·sparse projection 검사와 작은 control case를 통과했습니다. Dense 허용 오차는 `rtol=0.05`, `atol=0.02`이며 custom 대 Torch 최대 오차는 0.0078125였습니다.

안정 기준은 검증한 순전파 형상에 한정합니다. Backward 코드는 보존했지만 유사한 gather 정렬 위험과 전체 모델 학습은 아직 검증하지 않았습니다.

## 실측 기준값

각 행은 별도 세션에서 후보별 warmup 50회·측정 200회로 얻은 중앙값(ms)입니다. 동기화한 eager 순전파의 host dispatch·완료를 포함하며 컴파일·입력 전송·가중치 준비는 제외합니다.

| 강제 실행 경로 | TTNN separate | TTNN builtin fused | Custom |
|---|---:|---:|---:|
| Stream | 0.7458 | 0.6821 | 1.2735 |
| Sharded | 0.7047 | 0.6382 | 2.2793 |

이 형상에서 수정된 custom은 builtin fused보다 느렸습니다. 형상이 다르거나 출력이 틀렸던 과거 결과를 현재 기준본의 가속 성과로 쓰지 않았습니다. [검증 요약](results/validation_summary.json)과 [stream](results/stream_samples.json)·[sharded](results/sharded_samples.json) 원시 표본을 함께 보존했습니다.

## 통합 방법

검증 환경은 P100a, firmware 19.13.1, KMD 2.10.0, Python 3.12.14, TT-Metal `02fef3de6ebbf2e3d4a02452932d647b7e06c7af`와 로컬 통합 변경입니다.

TTNN 내부에 넣는 연산 코드입니다. `basic_block/`을 `ttnn/cpp/ttnn/operations/experimental/resnet/`에 배치하고 상위 CMake에 등록합니다. Experimental Python binding module에서 `ttnn::operations::basic_block::bind_basic_block`을 호출하도록 연결합니다. TTNN을 재빌드하고 새 host library와 해당 JIT kernel source가 함께 로드되는지 확인합니다. 상위 등록 변경과 독립 benchmark harness는 포함하지 않았습니다.

입력은 펼친 NHWC BF16 TILE 행렬이며 가중치·bias는 원래 packing 계약을 따르는 interleaved DRAM tensor입니다. `[W1, W2, W3, Wshortcut]`과 해당 bias를 사용하고 projection일 때 shortcut 파라미터를 사용합니다. `BASIC_BLOCK_PATH=auto|resident|sharded|stream`으로 경로를 선택합니다. 이번 저장소 정리에서는 커널 바이트를 보존했으며 장치 테스트를 다시 실행하지 않았습니다.
