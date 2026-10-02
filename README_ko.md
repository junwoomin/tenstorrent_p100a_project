# Tenstorrent P100a 프로젝트

[English](README.md)

Tenstorrent P100a(Blackhole)에서 VGG11으로 TTNN의 기본 구성과 실행 흐름을 익힌 뒤, TT-Metalium으로 ResNet bottleneck 전용 커널을 직접 개발하는 작업을 정리했습니다.

| 프로젝트 | 수행한 작업 |
|---|---|
| [VGG11](projects/vgg/README_ko.md) | 처음 TTNN을 사용하며 기본 구성·실행 흐름 확인, VGG11 설정 최적화와 TTML 분류기 학습 |
| [Custom Block](projects/block/README_ko.md) | ResNet bottleneck 전용 커널 개발, Conv·residual 결합과 resident·sharded·stream 실행 경로 구현 |

각 프로젝트에 코드와 작업·결과 요약을 정리했습니다.
