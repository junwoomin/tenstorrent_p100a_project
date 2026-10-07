# Tenstorrent P100a — ResNet50 Custom Block

[English](README.md)

현재 기준본은 **GPT로 만든 안정 버전(v7)**입니다. 사용자가 제공한 `basic_block_v7_no_debug(1).zip`을 기준으로 정리했습니다. 여기서 안정 버전은 아래 입력 조건에서 사용자가 현재 사용 중인 버전을 뜻하며, 이 시간 기록만으로 다양한 형상의 정확성이나 학습 안정성을 확인한 것은 아닙니다.

P100a(Blackhole)에서 TT-Metalium으로 ResNet bottleneck을 결합하고, 코어 분할·L1 버퍼 배치·데이터 이동을 다루는 프로젝트입니다. 현재 커스텀 연산은 **BatchNorm 없는 1×1 → 3×3 → 1×1 bottleneck**입니다.

## 현재 측정 결과

입력: **배치 8, 3채널, 256 × 256** — 논리적 NCHW 형상 `[8, 3, 256, 256]`.

| 버전 | 출력된 TOTAL |
|---|---:|
| Custom block + TTNN | 17.8934 ms |
| TTNN만 사용 | 15.3131 ms |

출력된 TOTAL 기준으로 커스텀 버전이 **2.5803 ms, 약 16.85% 느립니다**. 일부 초기 블록은 빨라졌지만 `layer2_0`, `layer3_0`, 특히 `layer4_0`의 지연이 큽니다. 이는 레이어 시간의 관찰이며, 내부 병목의 원인은 아직 확정하지 않았습니다.

표시된 레이어의 합계는 커스텀 **17.2348 ms**, TTNN **14.9168 ms**로 TOTAL과 다릅니다. 차이의 원인은 미확인입니다. 제공된 출력은 `avgpool`까지이며 FC 분류기는 표시되지 않았습니다. Warmup·측정 횟수·동기화 방식·SDK 리비전·두 구현의 수치 일치 검증 결과는 제공되지 않았습니다.

## 코드와 결과

- [소스 설명과 레이어별 시간](projects/block/README_ko.md)
- [주석을 제거한 소스 ZIP](projects/block/basic_block_v7_gpt_stable.zip)
- [측정값 기록](projects/block/results/resnet50_layer_timing.json)
- [소스 해시와 오프라인 확인 내역](projects/block/results/source_manifest.json)

기존 VGG와 이전 Block 자료는 현재 트리에서 정리했으며 이전 버전은 Git 이력에 남아 있습니다. 이번 변경은 소스 주석 제거와 문서·결과 교체입니다. 커널 최적화를 추가하거나 장치 실행을 재현한 결과는 아닙니다.

## P100a 벤치마크 분석

- [Benchmark K 분석·Excel·그래프](분석/README.md): BF16 DRAM 원소 연산 440조건·6,600표본의 코어 확장성, pipeline 효과, Host 변동성과 측정 한계.

