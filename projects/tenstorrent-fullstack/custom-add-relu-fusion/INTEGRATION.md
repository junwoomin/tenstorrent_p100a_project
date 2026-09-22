# TTNN 소스 트리 연결

본 디렉터리는 standalone CMake 프로젝트가 아니다. 첨부에는 자동 설치 스크립트가 없으므로 호환되는 tt-metal checkout에 수동 연결한다. 아래는 첨부 소스의 연결 지점이며 이번 환경에서 빌드를 실행하지 않았다.

## 1. 소스 배치

이 프로젝트의 `custom_add_relu/`를 tt-metal checkout의 다음 경로로 복사한다.

`ttnn/cpp/ttnn/operations/eltwise/custom_add_relu/`

Program factory가 해당 상대 경로에서 장치 커널을 찾으므로 폴더 이름과 위치를 유지한다.

## 2. CMake

상위 `ttnn/cpp/ttnn/operations/eltwise/CMakeLists.txt`에 한 번 추가한다.

```cmake
add_subdirectory(custom_add_relu)
```

첨부 CMake는 host 소스를 `ttnncpp`, nanobind 소스를 `ttnn` target에 연결한다.

## 3. Python 네이티브 바인딩

`ttnn/cpp/ttnn-nanobind/__init__.cpp`의 include 영역에 추가한다.

```cpp
#include "ttnn/operations/eltwise/custom_add_relu/custom_add_relu_nanobind.hpp"
```

다른 operation submodule 등록과 같은 초기화 범위에 추가한다.

```cpp
auto m_custom_add_relu = mod.def_submodule(
    "custom_add_relu", "Custom fused Add and ReLU operation");
ttnn::operations::custom_add_relu::bind_custom_add_relu(m_custom_add_relu);
```

기존 옵션으로 tt-metal을 다시 빌드하고 Python 프로세스를 재시작한다. 프로젝트의 [TTML 빌드 문서](../ttml-tt-train-build/README.md)를 참고한다. 중복 등록은 추가하지 않는다.

## 4. 공개 Python 이름과 선택적 TTML bridge

TTNN의 operation 자동 등록 후 `ttnn.custom_add_relu`와 `ttnn.custom_add_relu_backward`가 노출되는지 확인한다. 소스 버전에 따라 최상위 export 연결을 조정해야 한다.

```python
import ttnn
assert callable(ttnn.custom_add_relu)
assert callable(ttnn.custom_add_relu_backward)
```

Raw TTNN 입력의 forward에는 TTML bridge가 필요하지 않다. TTML 자동 역전파를 사용할 때는 `python/_custom_add_relu_autograd.py`를 checkout의 `ttnn/ttnn/`에 복사한 뒤, 네이티브 operation 등록 이후 다음을 호출한다.

```python
import ttnn
from ttnn._custom_add_relu_autograd import install

assert install(ttnn)
```

TTNN 초기화에 영구 연결하려면 같은 호출을 operation 등록 이후 위치에 추가해야 한다. 호환되는 `ttml.autograd.Function`이 필요하다. TTML Tensor를 raw 값으로 꺼내 전달하면 자동 그래프 연결이 유지되지 않는다.

## 5. Backward 범위

수동 backward는 `ttnn.custom_add_relu_backward(grad_out, saved_output)`로 호출한다. 두 gradient는 저장된 출력의 양수 여부를 이용한 `grad_out * (saved_output > 0)`이다. BF16 출력 반올림 및 0에서의 미분 규칙에 유의한다. 두 입력 gradient는 별도 버퍼를 갖는다.

이번 benchmark 결과는 forward에만 해당한다. 장치 forward 정확도, gradient 비교 및 실제 TTML 학습 검증 로그는 별도로 확보해야 한다.
