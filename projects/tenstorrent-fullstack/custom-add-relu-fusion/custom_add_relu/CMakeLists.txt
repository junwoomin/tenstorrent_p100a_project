# Included by ttnn/cpp/ttnn/operations/eltwise/CMakeLists.txt.
# This is an in-tree TTNN operation, not a standalone CMake project.
target_sources(
    ttnncpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/device/device_operation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device/program_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device/backward_device_operation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device/backward_program_factory.cpp
)

target_sources(
    ttnncpp
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/custom_add_relu.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/device/device_operation.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/device/backward_device_operation.hpp
)

if(TARGET ttnn)
    target_sources(
        ttnn
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/custom_add_relu_nanobind.cpp
    )
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/device/kernels/
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/custom_add_relu/device/kernels
    COMPONENT ttnn-runtime
    FILES_MATCHING PATTERN "*.cpp"
)
