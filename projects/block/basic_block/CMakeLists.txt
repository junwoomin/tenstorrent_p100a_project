






target_sources(
    ttnncpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/device/device_operation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device/program_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device/resident_program_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device/sharded_program_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device/stream_program_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device/backward_device_operation.cpp
)

target_sources(
    ttnncpp
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/basic_block.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/device/device_operation.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/device/host_common.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/device/backward_device_operation.hpp
)


if(TARGET ttnn)
    target_sources(
        ttnn
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/basic_block_nanobind.cpp
    )
endif()



install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/device/kernels/
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/resnet/basic_block/device/kernels
    COMPONENT ttnn-runtime
    FILES_MATCHING PATTERN "*.cpp" PATTERN "*.hpp"
)
