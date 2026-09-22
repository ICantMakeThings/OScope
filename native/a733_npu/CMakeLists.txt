cmake_minimum_required(VERSION 3.16)
project(a733_npu LANGUAGES CXX)

if(NOT A733_MODEL_ZOO_DIR)
    message(FATAL_ERROR "A733_MODEL_ZOO_DIR must point to the official model-zoo root")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(RUNTIME_DIR "${A733_MODEL_ZOO_DIR}/common/npuruntime")
set(A733_RUNTIME_DIR "${RUNTIME_DIR}/lib_linux_aarch64/A733")

add_library(a733_npu SHARED
    src/a733_npu.cpp
    ${RUNTIME_DIR}/npulib.cpp
    ${RUNTIME_DIR}/npu_util.cpp
)
target_include_directories(a733_npu PRIVATE
    include
    ${RUNTIME_DIR}
    ${RUNTIME_DIR}/include
)
target_compile_definitions(a733_npu PRIVATE CREATE_NETWORK_FROM_MEMORY=1)
target_compile_options(a733_npu PRIVATE -O3 -fPIC)
target_link_directories(a733_npu PRIVATE ${A733_RUNTIME_DIR})
target_link_libraries(a733_npu PRIVATE NBGlinker VIPhal dl pthread m)
set_target_properties(a733_npu PROPERTIES
    OUTPUT_NAME a733_npu
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN/native/a733_npu/.deps/model-zoo/common/npuruntime/lib_linux_aarch64/A733"
)
